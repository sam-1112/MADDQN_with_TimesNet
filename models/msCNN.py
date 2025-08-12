import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class InputBlock:
    def __init__(self, tradeData, Qvalue_sub_list):
        self.tradeData = tradeData
        self.Qvalue_sub_list = Qvalue_sub_list
    
    def getInput(self, block='MultiScaleBlock'):
        """ 
        Prepare the input tensor by processing trade data and Q-value sub-tensors.
        
        :return input_data: Processed tensor based on block type
        """
        if block == 'MultiScaleBlock':
            # 對於 MultiScaleBlock，只返回市場數據
            # 因為市場數據和子代理Q值的維度不兼容，無法直接連接
            return self.tradeData
        elif block == 'ActionBlock':
            # 對於 ActionBlock，返回展平的子代理Q值
            if isinstance(self.Qvalue_sub_list, list) and len(self.Qvalue_sub_list) > 0:
                # 如果是列表，連接所有Q值
                if len(self.Qvalue_sub_list) == 1:
                    qvalues = self.Qvalue_sub_list[0]
                else:
                    qvalues = torch.cat(self.Qvalue_sub_list, dim=-1)
            else:
                # 如果不是列表，直接使用
                qvalues = self.Qvalue_sub_list[0] if isinstance(self.Qvalue_sub_list, list) else self.Qvalue_sub_list
            
            # 確保有正確的batch維度
            if qvalues.dim() == 2:  # (n_agents, 3)
                qvalues = qvalues.unsqueeze(0)  # (1, n_agents, 3)
            
            # 展平為 (batch_size, n_agents * 3)
            batch_size = qvalues.size(0)
            flattened_qvalues = qvalues.view(batch_size, -1)
            return flattened_qvalues
        else:
            raise ValueError("Unsupported block type. Use 'MultiScaleBlock' or 'ActionBlock'.")

class MultiScaleBlock(nn.Module):
    def __init__(self, input_channels=5, sequence_length=10):
        super(MultiScaleBlock, self).__init__()
        
        # 第一個模塊：單尺度特徵提取器
        # 使用一維卷積技術提取時域特徵，輸出5維特徵
        self.single_scale = nn.Conv1d(
            in_channels=input_channels,  # 5個參數 (OHLCV)
            out_channels=5,  # 堆疊成5維特徵
            kernel_size=3,
            padding=1
        )
        
        # 第二個模塊：中等尺度特徵提取器
        # 將數據視為單通道二維圖像，使用3×3卷積核的二維卷積
        self.medium_scale = nn.Conv2d(
            in_channels=1,  # 單通道
            out_channels=2,  # 2個輸出通道
            kernel_size=(3, 3),
            padding=(1, 1)
        )
        
        # 第三個模塊：全局尺度特徵提取器
        # 考慮單通道圖像數據，使用5×5卷積核大小進行全局特徵提取
        self.global_scale = nn.Conv2d(
            in_channels=1,  # 單通道
            out_channels=1,  # 1個輸出通道
            kernel_size=(5, 5),
            padding=(2, 2)
        )
        
        self.sequence_length = sequence_length
        self.input_channels = input_channels
    
    def forward(self, x):
        """
        輸入: (batch_size, 1, sequence_length, input_channels)
        輸出: (batch_size, 8, sequence_length, input_channels) - 8通道二維特徵層
        """
        batch_size = x.size(0)
        
        # 移除通道維度以便1D卷積: (batch_size, sequence_length, input_channels)
        x_for_1d = x.squeeze(1)  # (batch_size, sequence_length, input_channels)
        
        # 轉置為1D卷積期望的格式: (batch_size, input_channels, sequence_length)
        x_for_1d = x_for_1d.transpose(1, 2)  # (batch_size, input_channels, sequence_length)
        
        # 第一個模塊：單尺度特徵提取 (1D卷積)
        single_features = F.relu(self.single_scale(x_for_1d))  # (batch_size, 5, sequence_length)
        
        # 將1D卷積結果重塑為2D格式以便連接
        # (batch_size, 5, sequence_length) -> (batch_size, 5, sequence_length, input_channels)
        single_features_2d = single_features.unsqueeze(-1).expand(
            -1, -1, -1, self.input_channels
        )  # (batch_size, 5, sequence_length, input_channels)
        
        # 第二個模塊：中等尺度特徵提取 (2D卷積，3×3核)
        medium_features = F.relu(self.medium_scale(x))  # (batch_size, 2, sequence_length, input_channels)
        
        # 第三個模塊：全局尺度特徵提取 (2D卷積，5×5核)
        global_features = F.relu(self.global_scale(x))  # (batch_size, 1, sequence_length, input_channels)
        
        # 連接三個模塊的輸出形成8通道二維特徵層
        # 5 (single) + 2 (medium) + 1 (global) = 8 通道
        multi_scale_output = torch.cat([
            single_features_2d,  # (batch_size, 5, sequence_length, input_channels)
            medium_features,     # (batch_size, 2, sequence_length, input_channels)
            global_features      # (batch_size, 1, sequence_length, input_channels)
        ], dim=1)  # (batch_size, 8, sequence_length, input_channels)
        
        return multi_scale_output


class ECABlock2D(nn.Module):
    """2D版本的ECA Block，適用於圖像數據"""
    def __init__(self, channels, gamma=2, b=1):
        super(ECABlock2D, self).__init__()
        # 自適應核大小
        kernel_size = int(abs(math.log2(channels) / gamma + b / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, 
                             padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # x: (batch_size, channels, height, width)
        batch_size, channels, height, width = x.size()
        
        # 全局平均池化: (batch_size, channels, 1, 1)
        y = self.avg_pool(x)
        
        # 重塑為1D卷積格式: (batch_size, 1, channels)
        y = y.squeeze(-1).transpose(-1, -2)
        
        # 1D卷積
        y = self.conv(y)
        
        # 恢復形狀並應用sigmoid: (batch_size, channels, 1, 1)
        y = y.transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        
        # 廣播相乘
        return x * y.expand_as(x)

class BackBoneBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(BackBoneBlock, self).__init__()
        # 第一層卷積 + MaxPooling
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 第二層卷積
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding)
        
        # ECA Block
        self.eca = ECABlock2D(channels=out_channels)
        
        # 第三層卷積
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding)
        
        # 線性層（用於最後的特徵處理）
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear1 = nn.Linear(out_channels, out_channels)
        self.linear2 = nn.Linear(out_channels, out_channels)
    
    def forward(self, x):
        # 第一個 Conv3×3 + MaxPooling
        x = F.relu(self.conv1(x))
        x = self.maxpool(x)
        
        # 第二個 Conv3×3
        x = F.relu(self.conv2(x))
        
        # ECA Block
        x = self.eca(x)
        
        # 第三個 Conv3×3
        x = F.relu(self.conv3(x))
        
        # 全局平均池化並通過線性層
        pooled = self.adaptive_pool(x)  # (batch_size, out_channels, 1, 1)
        pooled = pooled.view(pooled.size(0), -1)  # (batch_size, out_channels)
        
        features = F.relu(self.linear1(pooled))
        features = self.linear2(features)
        
        return features

class ActionBlock(nn.Module):
    def __init__(self, backbone_features_size, subagent_qvalues_size, hidden_size=64):
        super(ActionBlock, self).__init__()
        self.backbone_features_size = backbone_features_size
        self.subagent_qvalues_size = subagent_qvalues_size
        self.hidden_size = hidden_size
        
        # Action Block 有兩個線性層
        total_input_size = backbone_features_size + subagent_qvalues_size
        self.linear1 = nn.Linear(total_input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 3)  # 3個動作
        self.dropout = nn.Dropout(0.1)
        
        # 添加輸入正規化
        self.input_norm = nn.LayerNorm(total_input_size)
        
        # print(f"ActionBlock initialized: backbone={backbone_features_size}, subagent={subagent_qvalues_size}, total={total_input_size}")
    
    def forward(self, backbone_features, subagent_qvalues):
        # 連接backbone特徵和子代理Q值
        combined = torch.cat([backbone_features, subagent_qvalues], dim=1)
        
        # 輸入正規化
        combined = self.input_norm(combined)
        
        # 通過線性層
        x = F.relu(self.linear1(combined))
        x = self.dropout(x)
        x = self.linear2(x)
        
        # 限制輸出範圍避免爆炸
        # x = torch.clamp(x, min=-10.0, max=10.0)
        
        # print(f"ActionBlock forward: output range [{x.min().item():.4f}, {x.max().item():.4f}]")
        return x

class MSCNN(nn.Module):
    def __init__(self, configs):
        super(MSCNN, self).__init__()
        
        # 從配置中獲取參數
        multi_scale_config = configs['mscnn']['multi_scale']
        backbone_config = configs['mscnn']['backbone']
        action_config = configs['mscnn']['action']
        
        # 時序數據的維度信息
        self.sequence_length = configs.get('seq_len', 10)
        self.feature_dim = configs['env']['features']  # 5個特徵 (OHLCV)
        
        # Multi-Scale Block - 根據論文精確實現
        self.multi_scale_block = MultiScaleBlock(
            input_channels=self.feature_dim,  # 5個參數
            sequence_length=self.sequence_length
        )
        
        # Multi-Scale Block輸出8個通道
        multi_scale_output_channels = 8
        
        # Backbone Block (2D卷積)
        self.backbone_block = BackBoneBlock(
            multi_scale_output_channels,  # 8通道輸入
            backbone_config['out_channels']
        )
        
        # 不在初始化時創建ActionBlock，而是在第一次forward時動態創建
        self.backbone_output_channels = backbone_config['out_channels']
        self.action_hidden_size = action_config['hidden_size']
        self.action_block = None
        
        # 計算子代理Q值的固定尺寸
        # 假設有 risk_agent + return_agent 個子代理，每個輸出3個Q值
        n_risk_agents = configs.get('env', {}).get('risk_agent', 2)
        n_return_agents = configs.get('env', {}).get('return_agent', 2)
        total_agents = n_risk_agents + n_return_agents
        subagent_qvalues_size = total_agents * 3  # 每個代理3個Q值
        
        # print(f"MSCNN Init - Backbone output: {self.backbone_output_channels}")
        # print(f"MSCNN Init - Subagent Q-values size: {subagent_qvalues_size}")
        # print(f"MSCNN Init - Total agents: {total_agents} (risk: {n_risk_agents}, return: {n_return_agents})")
        
        # 🔧 修復：創建固定的 ActionBlock 作為正式的子模塊
        self.action_block = ActionBlock(
            backbone_features_size=self.backbone_output_channels,
            subagent_qvalues_size=subagent_qvalues_size,
            hidden_size=self.action_hidden_size
        )
        
        # 添加輸出標準化和限制
        self.output_norm = nn.LayerNorm(3)
        self.output_scale = nn.Parameter(torch.tensor(1.0))  # 可學習的輸出縮放因子
        
        # 初始化所有權重
        self.apply(self._init_weights)
        
        # print("MSCNN initialized with fixed ActionBlock and output normalization")
    
    def _init_weights(self, m):
        """保守的權重初始化"""
        if isinstance(m, nn.Linear):
            # 使用更保守的初始化
            torch.nn.init.xavier_uniform_(m.weight, gain=0.01)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, (nn.Conv1d, nn.Conv2d)):
            torch.nn.init.xavier_uniform_(m.weight, gain=0.01)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0.0)
    
    def forward(self, market_data, sub_agent_qvalues):
        """
        前向傳播
        
        Args:
            market_data: [B, T, N] 或 [T, N] 市場數據
            sub_agent_qvalues: [B, n_agents, 3] 或 [n_agents, 3] 子代理Q值
        
        Returns:
            torch.Tensor: [B, 3] Q值輸出
        """
        try:
            # 🔧 修復：標準化輸入維度
            market_data, sub_agent_qvalues = self._standardize_inputs(market_data, sub_agent_qvalues)
            batch_size = market_data.size(0)
            
            # Multi-Scale特徵提取
            market_data_2d = market_data.unsqueeze(1)  # [B, 1, T, N]
            multi_scale_features = self.multi_scale_block(market_data_2d)  # [B, 8, T, N]
            
            # Backbone處理
            backbone_features = self.backbone_block(multi_scale_features)  # [B, out_channels]
            
            # 檢查backbone特徵
            if torch.isnan(backbone_features).any() or torch.isinf(backbone_features).any():
                print("Warning: backbone_features contains NaN or Inf")
                return torch.zeros(batch_size, 3, device=market_data.device)
            
            # 🔧 修復：固定的子代理Q值處理
            subagent_flat = self._process_subagent_qvalues(sub_agent_qvalues, batch_size)
            
            # 檢查維度匹配
            expected_backbone_size = self.action_block.backbone_features_size
            expected_subagent_size = self.action_block.subagent_qvalues_size
            actual_backbone_size = backbone_features.size(1)
            actual_subagent_size = subagent_flat.size(1)
            
            if actual_backbone_size != expected_backbone_size:
                raise ValueError(f"Backbone features size mismatch: expected {expected_backbone_size}, got {actual_backbone_size}")
            
            if actual_subagent_size != expected_subagent_size:
                print(f"Warning: Subagent Q-values size mismatch: expected {expected_subagent_size}, got {actual_subagent_size}")
                # 調整子代理Q值大小
                subagent_flat = self._adjust_subagent_size(subagent_flat, expected_subagent_size)
            
            # 🔧 修復：使用固定的 ActionBlock
            action_output = self.action_block(backbone_features, subagent_flat)
            
            # 🔧 新增：輸出標準化和限制
            action_output = self.output_norm(action_output)
            action_output = torch.tanh(action_output) * self.output_scale  # 限制輸出範圍
            
            # 最終檢查
            if torch.isnan(action_output).any() or torch.isinf(action_output).any():
                print("Warning: action_output contains NaN or Inf after normalization")
                return torch.zeros(batch_size, 3, device=market_data.device)
            
            return action_output
            
        except Exception as e:
            print(f"Error in MSCNN forward: {e}")
            batch_size = market_data.size(0) if market_data.dim() > 0 else 1
            device = market_data.device if hasattr(market_data, 'device') else torch.device('cpu')
            return torch.zeros(batch_size, 3, device=device)
    
    def _standardize_inputs(self, market_data, sub_agent_qvalues):
        """標準化輸入維度"""
        # 處理market_data
        if market_data.dim() == 2:  # [T, N] -> [1, T, N]
            market_data = market_data.unsqueeze(0)
        elif market_data.dim() == 1:  # [N] -> [1, 1, N]
            market_data = market_data.unsqueeze(0).unsqueeze(0)
        
        # 處理sub_agent_qvalues
        if sub_agent_qvalues.dim() == 2:  # [n_agents, 3] -> [1, n_agents, 3]
            sub_agent_qvalues = sub_agent_qvalues.unsqueeze(0)
        elif sub_agent_qvalues.dim() == 1:  # [features] -> [1, n_agents, 3]
            # 假設是展平的Q值，重新reshape
            total_qvalues = sub_agent_qvalues.size(0)
            if total_qvalues % 3 == 0:
                n_agents = total_qvalues // 3
                sub_agent_qvalues = sub_agent_qvalues.view(1, n_agents, 3)
            else:
                # 如果無法整除，使用默認形狀
                sub_agent_qvalues = sub_agent_qvalues.unsqueeze(0).unsqueeze(0)
        
        return market_data, sub_agent_qvalues
    
    def _process_subagent_qvalues(self, sub_agent_qvalues, batch_size):
        """處理子代理Q值"""
        # 展平子代理Q值: [B, n_agents, 3] -> [B, n_agents * 3]
        subagent_flat = sub_agent_qvalues.reshape(batch_size, -1)
        return subagent_flat
    
    def _adjust_subagent_size(self, subagent_flat, expected_size):
        """調整子代理Q值大小以匹配期望尺寸"""
        current_size = subagent_flat.size(1)
        
        if current_size < expected_size:
            # 如果當前尺寸小於期望，用零填充
            padding = torch.zeros(subagent_flat.size(0), expected_size - current_size, 
                                device=subagent_flat.device, dtype=subagent_flat.dtype)
            subagent_flat = torch.cat([subagent_flat, padding], dim=1)
        elif current_size > expected_size:
            # 如果當前尺寸大於期望，截取前面部分
            subagent_flat = subagent_flat[:, :expected_size]
        
        return subagent_flat

    def get_model_info(self):
        """獲取模型信息用於調試"""
        return {
            'backbone_output_channels': self.backbone_output_channels,
            'action_block_exists': self.action_block is not None,
            'action_block_backbone_size': self.action_block.backbone_features_size if self.action_block else None,
            'action_block_subagent_size': self.action_block.subagent_qvalues_size if self.action_block else None,
            'parameters': list(self.state_dict().keys())
        }