import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class TokenEmbedding(nn.Module):
    
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model, kernel_size=3, padding=padding, padding_mode='circular', bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
    
    def forward(self, x):
        """
        Forward pass for the token embedding layer.
        Args:
            x: Input tensor of shape (batch_size, input_dim, seq_length)
        :return x: 
            Output tensor of shape (batch_size, output_dim, seq_length)
        """
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)  # Change shape to (batch_size, input_dim, seq_length)
        return x

class PositionEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        """
        Initializes the PositionEmbedding layer with sinusoidal position encodings.

        :param outputDim: Dimension of the output embedding
        :param max_len: Maximum length of the input sequence for which position embeddings are computed
        """
        super(PositionEmbedding, self).__init__()
        self.outputDim = d_model
        self.maxlen = max_len

        pe = torch.zeros(self.maxlen, self.outputDim).float()
        pe.requires_grad = False

        position = torch.arange(0, self.maxlen).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, self.outputDim, 2).float() * -(np.log(10000.0) / self.outputDim)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Forward pass for the position embedding layer.

        :param x: Input tensor of shape (batch_size, seq_length, output_dim)
        :return x: Output tensor with position embeddings added
        """
        return self.pe[:, :x.size(1)]
        

class EmbeddingBlock(nn.Module):
    # Data embedding = Position embedding + Token embedding
    def __init__(self, c_in, d_model, dropout=0.1):
        """
        Initializes the EmbeddingBlock with input and output dimensions.

        :param inputDim: Dimension of the input data
        :param outputDim: Dimension of the output embedding
        """
        super(EmbeddingBlock, self).__init__()
        self.inputDim = c_in
        self.outputDim = d_model
        self.tokenEmbedding = TokenEmbedding(c_in, d_model)
        self.positionEmbedding = PositionEmbedding(d_model)
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, x):
        """
        Forward pass for the embedding block.
        
        :param x: Input tensor of shape (batch_size, seq_length, input_dim)
        :return: Output tensor of shape (batch_size, seq_length, output_dim)
        """
        x = self.tokenEmbedding(x) + self.positionEmbedding(x)
        return self.dropout(x)
    
class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        """
        Initializes the InceptionBlock with multiple convolutional kernels.

        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param num_kernels: Number of different kernel sizes to use
        :param init_weight: Whether to initialize weights (default: True)

        """
        super(InceptionBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []
        for i in range(self.num_kernels):
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i))
        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()
    
    def _initialize_weights(self):
        """
        Initializes the weights of the convolutional layers using Kaiming normal initialization.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass for the Inception block.

        :param x: Input tensor of shape (batch_size, in_channels, seq_length)
        :return res: Output tensor of shape (batch_size, out_channels * num_kernels, seq_length)
        """
        res_list = []
        for i in range(self.num_kernels):
            res_list.append(self.kernels[i](x))
        res = torch.stack(res_list, dim=-1).mean(dim=-1)
        return res
        

class TimesBlock(nn.Module):

    def __init__(self, configs):
        """
        Initializes the TimesBlock with the given configurations.

        :param configs: Configuration object containing model parameters such as seq_len, pred_len, d_model, d_ff, num_kernels, and top_k
        """
        super(TimesBlock, self).__init__()
        self.seq_len = configs['training']['seq_len']
        self.pred_len = configs['training']['pred_len']
        self.k = configs['timesnet']['top_k']
        self.conv = nn.Sequential(
            InceptionBlock(configs['timesnet']['d_model'], configs['timesnet']['d_ff'], num_kernels=configs['timesnet']['num_kernels']),
            nn.GELU(),
            InceptionBlock(configs['timesnet']['d_ff'], configs['timesnet']['d_model'], num_kernels=configs['timesnet']['num_kernels'])
        )
    
    def forward(self, x):
        """
        Forward pass for the TimesBlock.

        :param x: Input tensor of shape (batch_size, seq_len, d_model)
        :return res: Output tensor of shape (batch_size, pred_len, d_model)
        """
        B, T, N = x.size()
        period_list, period_weight = self.FFTforPeriod(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]
            if T % period != 0:
                length = ((T // period) + 1) * period
                padding = torch.zeros((B, length - T, N)).to(x.device)
                out = torch.cat((x, padding), dim=1)
            else:
                length = T
                out = x
            out = out.reshape(B, length // period, period, N).permute(0, 3, 1, 2).contiguous()
            out = self.conv(out)
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :T, :])
        res = torch.stack(res, dim=-1)
        period_weight = F.softmax(period_weight, dim=-1)
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, dim=-1)
        res = res + x 
        return res         


    def FFTforPeriod(self, x, k=2):
        """
        Computes the Fast Fourier Transform (FFT) to identify the top k periods in the input time series data.

        :param x: Input tensor of shape (batch_size, seq_len, input_dim)
        :param k: Number of top periods to identify
        :return period: List of top k periods identified from the FFT
        :return frequency_list: Tensor containing the average frequency magnitudes for the top k periods
        """
        xf = torch.fft.rfft(x.float(), dim=1)
        frequency_list = abs(xf).mean(0).mean(-1)
        frequency_list[0] = 0
        _, top_list = torch.topk(frequency_list, k)
        top_list = top_list.detach().cpu().numpy()
        period = x.shape[1] // top_list
        return period, abs(xf).mean(-1)[:, top_list]
    

class TimesNet(nn.Module):
    def __init__(self, configs):
        """
        Initializes the TimesNet model with the given configurations.

        :param configs: Configuration object containing model parameters such as seq_len, pred_len, d_model, output_dim, and top_k
        """
        super(TimesNet, self).__init__()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.configs = configs
        self.seq_len = configs['training']['seq_len']
        self.label_len = configs['training']['label_len']
        self.pred_len = configs['training']['pred_len']

        self.model = nn.ModuleList([TimesBlock(configs) for _ in range(configs['timesnet']['e_layers'])])
        self.embedding = EmbeddingBlock(configs['timesnet']['enc_in'], configs['timesnet']['d_model'], configs['training']['dropout'])
        self.layer = configs['timesnet']['e_layers']
        self.layer_norm = nn.LayerNorm(configs['timesnet']['d_model'])
        
        self.predict_linear = nn.Linear(self.seq_len, self.pred_len + self.seq_len)
        
        # for classification task(buy, sell, hold)
        # Action Block
        self.activation = F.gelu
        self.dropout = nn.Dropout(p=self.configs['training']['dropout'])
        hidden_dim = configs['timesnet']['d_model'] * configs['training']['seq_len'] // 4  # 可調整縮放比例
        self.projection1 = nn.Linear(configs['timesnet']['d_model']*configs['training']['seq_len'], hidden_dim, bias=True)
        self.projection2 = nn.Linear(hidden_dim, configs['timesnet']['num_class'], bias=True)
    
    def classification(self, x_enc, x_mark_enc=None):
        enc_out = self.embedding(x_enc)  # [B, T, C]

        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))  # [B, T, C]

        output = self.activation(enc_out)
        output = self.dropout(output)

        if x_mark_enc is not None:
            output = output * x_mark_enc.unsqueeze(-1)  # 加權時間資訊

        output = output.reshape(output.shape[0], -1)  # [B, T*C]
        output = self.projection1(output)
        output = self.projection2(output)
        # print(f"Output : {output}")
        return output  # [B, num_class]

    def forward(self, x_enc, x_mark_enc=None):
        return self.classification(x_enc, x_mark_enc)      

# main function to test the model

    