from utils.dataPreprocess import PreprocessData
import numpy as np
import torch
import torch.nn as nn
import os
import time
from models.Timesnet import TimesNet


from dataclasses import dataclass

@dataclass
class Experiment:
    agent: str
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray


class ModelTraining:
    def __init__(self, configs, unprocessed_data=None, policy_network=None, target_network=None):
        self.configs = configs
        self.device = torch.device("cuda" if torch.cuda.is_available() and configs['gpu']['use_gpu'] else "cpu")
        self.unprocessed_data = unprocessed_data
        self.data_processor = PreprocessData(data=self.unprocessed_data, window_size=configs['env']['window_size'])
        
        self.policy_network = policy_network
        self.target_network = target_network

    def modelTraining(self):
        
        train_data, test_data = self.data_processor.timeSeriesData()
        

        filePath = os.path.join(self.configs['model']['checkpoints'])
        if not os.path.exists(filePath):
            os.makedirs(filePath)
        
        trainLoader = torch.utils.data.DataLoader(
            dataset=torch.utils.data.TensorDataset(torch.tensor(train_data, dtype=torch.float32)),
            batch_size=self.configs['training']['batch_size'],
            shuffle=True,
            num_workers=0
        )
        testLoader = torch.utils.data.DataLoader(
            dataset=torch.utils.data.TensorDataset(torch.tensor(test_data, dtype=torch.float32)),
            batch_size=self.configs['training']['batch_size'],
            shuffle=False,
            num_workers=0
        )

        time_now = time.time()
        train_steps = len(trainLoader)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.configs['training']['learning_rate'])
        criterion = nn.CrossEntropyLoss()
        
        # DDQN agent training loop
        for epoch in range(self.configs['training']['epochs']):
            iter_count = 0
            train_loss = []

            self.policy_network.train()
            for batch_data in trainLoader:
                iter_count += 1
                epoch_start_time = time.time()

                if self.configs['gpu']['use_gpu']:
                    batch_data = batch_data[0].to(self.device)

                dec_inp = torch.zeros_like(batch_data[:, -self.configs['training']['pred_len']:]).float()
                dec_inp = torch.cat([batch_data[:, :self.configs['training']['label_len']], dec_inp], dim=1).float().to(self.device)

                outputs = self.policy_network(batch_data, dec_inp)
                
                predictions = outputs[:, -1, :]
                targets = batch_data[:, -self.configs['training']['pred_len']:, :].to(self.device)
                loss = criterion(predictions, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss.append(loss.item())
                self.soft_update_target()

            
            
            print(f"Epoch [{epoch+1}/{self.configs['training']['epochs']}], Train Loss: {np.mean(train_loss):.4f}, Time: {time.time() - epoch_start_time:.2f}s")
            train_loss = np.average(train_loss)
            
            lr_adjust = self.configs['training']['learning_rate'] * (0.5 ** (epoch // 1))
            self.configs['training']['learning_rate'] = lr_adjust

        best_model_path = os.path.join(filePath, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(best_model_path))
        print(f"Training completed. Best model loaded from {best_model_path}.")
        test_loss = self.validation(testLoader, criterion)
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Total training time: {time.time() - time_now:.2f}s")
        # return self.model

    def validation(self, validLoader, criterion=None):
            """
            Validates the model on the validation dataset.

            :param validLoader: DataLoader for the validation dataset
            :return: Average validation loss
            """
            self.model.eval()
            valid_loss = 0.0

            with torch.no_grad():
                for batch_x, batch_y in validLoader:
                    if self.configs['gpu']['use_gpu']:
                        batch_x = batch_x.to(self.device)
                        batch_y = batch_y.to(self.device)
                        

                    dec_inp = torch.zeros_like(batch_y[:, -self.configs['training']['pred_len']:]).float()
                    dec_inp = torch.cat([batch_y[:, :self.configs['training']['label_len']], dec_inp], dim=1).float().to(self.device)

                    outputs = self(batch_x, dec_inp)
                    # f_dim = 0
                    # outputs = outputs[:, -self.configs['training']['pred_len']:, f_dim:]
                    # batch_y = batch_y[:, -self.configs['training']['pred_len']:, f_dim:].to(self.device)
                    # loss = criterion(outputs, batch_y)

                    predictions = outputs[:, -1, :]  # [B, D]
                    targets = batch_y.to(self.device)
                    loss = criterion(predictions, targets)

                    valid_loss += loss.item()

            return valid_loss / len(validLoader)