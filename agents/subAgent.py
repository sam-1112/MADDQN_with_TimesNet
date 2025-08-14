# agents/risk_agent.py 或 return_agent.py
import math
from models.Timesnet import TimesNet
from agents.baseAgent import BaseAgent
import torch
import torch.nn.functional as F
import numpy as np
import random
# from utils.training import ModelTraining


class subAgent(BaseAgent):
    def __init__(self, agentType='risk', **kwargs):
        policy_network = TimesNet(kwargs['configs'])
        target_network = TimesNet(kwargs['configs'])
        # 確保設備參數正確傳遞，優先使用 GPU
        device = kwargs.get('device', torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        super().__init__(
            configs=kwargs['configs'],
            model=policy_network, 
            target_model=target_network, 
            action_dim=3,
            lr=kwargs['configs'].get('training', {}).get('learning_rate', 1e-3),
            device=device
        )
        self.agentType = agentType
        # define the agent model training
        # self.training = ModelTraining(configs=kwargs['configs'], 
        #                               unprocessed_data=kwargs['unprocessed_data'],
        #                               policy_network=self.policy_network,
        #                               target_network=self.target_network)
        

    def act(self, state, training=True):
        """ 
        更改狀態

        :param state: 當前狀態
        :return action: 採取的動作或Q值
        :return q_values: 3個動作的Q值
        """
        # 確保 state 是正確的 3D 形狀 (batch_size, seq_len, input_dim)
        # state = torch.tensor(state, dtype=torch.float32)
        # print(f"State shape: {state.shape}")
        # 如果是 2D (seq_len, input_dim)，添加 batch 維度
        if state.dim() == 2:
            state = state.unsqueeze(0)  # (1, seq_len, input_dim)
        # 如果是 1D，需要重新整形
        elif state.dim() == 1:
            # 假設需要重新整形為 (1, seq_len, 1) 或根據配置調整
            state = state.unsqueeze(0).unsqueeze(-1)  # (1, seq_len, 1)
        
        # 確保張量移動到正確的設備
        state = state.to(self.device)
        
        # 動作映射：模型輸出 [0, 1, 2] -> 環境期望 [-1, 0, 1]
        action_mapping = [-1, 0, 1]  # 索引0->-1(short), 索引1->0(hold), 索引2->1(long)
        # print(f"Epsilon value: {self.epsilon}")
        if training:
            if random.random() < self.epsilon:
                # 隨機選擇動作索引，然後映射為環境動作
                action_idx = random.randint(0, self.action_dim - 1)
                action = action_mapping[action_idx]
                return action, np.zeros(self.action_dim, dtype=np.float32)
            else:
                with torch.no_grad():
                    q_values = self.policy_network(state)
                    action_idx = q_values.argmax().item()
                    action = action_mapping[action_idx]
                    return action, q_values.squeeze().cpu().numpy()
        else:
            with torch.no_grad():
                q_values = self.policy_network(state)
                action_idx = q_values.argmax().item()
                action = action_mapping[action_idx]
                return action, q_values.squeeze().cpu().numpy()

    def learn(self, batch):
        """
        更新模型的學習方法

        :param batch: Tuple of (states, actions, rewards, next_states)
        """
        states, actions, rewards, next_states = batch
        
        # 確保狀態是正確的 3D 形狀並移動到正確的設備
        states = torch.tensor(states, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        
        # 處理狀態維度
        if states.dim() == 2:
            states = states.unsqueeze(0)
        if next_states.dim() == 2:
            next_states = next_states.unsqueeze(0)
            
        # 確保所有張量都在正確的設備上
        states = states.to(self.device)
        next_states = next_states.to(self.device)
        
        # 處理動作：將環境動作 [-1, 0, 1] 映射為模型索引 [0, 1, 2]
        action_mapping = {-1: 0, 0: 1, 1: 2}
        if isinstance(actions, (int, float)):
            mapped_action = action_mapping[actions]
            actions = torch.tensor([mapped_action], dtype=torch.long).to(self.device)
        else:
            mapped_actions = [action_mapping[a] for a in actions]
            actions = torch.tensor(mapped_actions, dtype=torch.long).to(self.device)
        
        # 處理獎勵
        if isinstance(rewards, (int, float)):
            rewards = torch.tensor([rewards], dtype=torch.float32).to(self.device)
        else:
            rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)

        # Calculate Q values
        current_q_values = self.policy_network(states)
        next_q_values_target = self.target_network(next_states)
        
        # 計算目標 Q 值
        with torch.no_grad():
            target_q_values = rewards + self.gamma * next_q_values_target.max(1)[0]
        
        # 選擇對應動作的 Q 值
        if current_q_values.dim() == 1:
            # 如果 Q 值是 1D，直接用索引選擇
            current_q_value_selected = current_q_values[actions[0]].unsqueeze(0)  # 確保維度一致
        else:
            # 如果 Q 值是 2D，使用 gather
            current_q_value_selected = current_q_values.gather(1, actions.unsqueeze(1)).squeeze()
        
        # 確保 target_q_values 和 current_q_value_selected 維度一致
        if target_q_values.dim() == 0:
            target_q_values = target_q_values.unsqueeze(0)
        if current_q_value_selected.dim() == 0:
            current_q_value_selected = current_q_value_selected.unsqueeze(0)

        loss = F.mse_loss(current_q_value_selected, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.decay_epsilon()

    def decay_epsilon(self, episode):
        """
        根據 epsilon_decay 參數，衰減 epsilon 的值，並確保其不低於 epsilon_min。
        """
        # 指數衰減公式
        # self.epsilon = self.epsilon_min + (self.epsilon_max - self.epsilon_min) * math.exp(-self.epsilon_decay * episode)
        # 線性衰減公式
        self.epsilon = max(self.epsilon_min, self.epsilon_max - (self.epsilon_max - self.epsilon_min) * (episode / self.episodes))