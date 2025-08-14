import math
from agents.baseAgent import BaseAgent
from models.msCNN import MSCNN
import torch
import numpy as np
import random
import torch.nn.functional as F

class finalAgent(BaseAgent):
    def __init__(self, agentType='final', **kwargs):
        policy_network = MSCNN(kwargs['configs'])
        target_network = MSCNN(kwargs['configs'])
        super().__init__(
            configs=kwargs['configs'],
            model=policy_network, 
            target_model=target_network, 
            action_dim=3,
            lr=kwargs['configs'].get('training', {}).get('learning_rate', 1e-3),
            device=kwargs.get('device', torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        )
        self.agentType = agentType
        

    def act(self, market_state, sub_agent_qvalues, training=True):
        """ 
        選擇動作，基於市場數據和子代理Q值

        :param market_state: 當前市場狀態 (環境狀態)
        :param sub_agent_qvalues: 子代理的Q值 (n_agents, 3)
        :return action: 採取的動作 (-1, 0, 1)
        :return q_values: 3個動作的Q值
        """
        # 確保輸入是 tensor 格式
        if not isinstance(market_state, torch.Tensor):
            market_state = torch.tensor(market_state, dtype=torch.float32).to(self.device)
        if not isinstance(sub_agent_qvalues, torch.Tensor):
            sub_agent_qvalues = torch.tensor(sub_agent_qvalues, dtype=torch.float32).to(self.device)
        
        # 確保有正確的維度
        if market_state.dim() == 2:  # (sequence_length, features)
            market_state = market_state.unsqueeze(0)  # (1, sequence_length, features)
        if sub_agent_qvalues.dim() == 2:  # (n_agents, 3)
            sub_agent_qvalues = sub_agent_qvalues.unsqueeze(0)  # (1, n_agents, 3)
        if training:
            # 使用 epsilon-greedy 策略
            if random.random() < self.epsilon:
                action = random.choice([-1, 0, 1])  # 環境動作空間
                # 確定 action_dim 的數量
                num_actions = len(self.action_dim) if isinstance(self.action_dim, list) else self.action_dim
                return action, np.zeros(num_actions, dtype=np.float32)
            else:
                with torch.no_grad():
                    # 設置為評估模式以避免 BatchNorm 問題
                    self.policy_network.eval()
                    q_values = self.policy_network(market_state, sub_agent_qvalues)
                    # 恢復訓練模式
                    self.policy_network.train()
                    
                    # 動作映射：網路索引 [0, 1, 2] -> 環境動作 [-1, 0, 1]
                    action_idx = q_values.argmax().item()
                    action_mapping = {0: -1, 1: 0, 2: 1}
                    action = action_mapping[action_idx]
                    
                    return action, q_values.squeeze().cpu().numpy()
        else:
            with torch.no_grad():
                # 設置為評估模式以避免 BatchNorm 問題
                self.policy_network.eval()
                q_values = self.policy_network(market_state, sub_agent_qvalues)
                # 恢復訓練模式
                self.policy_network.train()
                
                # 動作映射：網路索引 [0, 1, 2] -> 環境動作 [-1, 0, 1]
                action_idx = q_values.argmax().item()
                action_mapping = {0: -1, 1: 0, 2: 1}
                action = action_mapping[action_idx]
                
                return action, q_values.squeeze().cpu().numpy()

    def learn(self, experience):
        """
        更新模型的學習方法，支援 final agent 的特殊輸入格式

        :param experience: Tuple of (market_state, action, reward, next_market_state, subagent_qvalues, next_subagent_qvalues)
        """
        if isinstance(experience, tuple) and len(experience) == 6:
            market_state, action, reward, next_market_state, subagent_qvalues, next_subagent_qvalues = experience
            
            # 檢查獎勵是否有效
            if np.isnan(reward) or np.isinf(reward):
                print(f"Warning: Invalid reward in Final Agent learning: {reward}")
                return
            
            # 轉換為張量並確保正確的維度
            market_state = torch.tensor(market_state, dtype=torch.float32).to(self.device)
            if market_state.dim() == 2:  # (sequence_length, features)
                market_state = market_state.unsqueeze(0)  # (1, sequence_length, features)
            
            next_market_state = torch.tensor(next_market_state, dtype=torch.float32).to(self.device)
            if next_market_state.dim() == 2:  # (sequence_length, features)
                next_market_state = next_market_state.unsqueeze(0)  # (1, sequence_length, features)
            
            subagent_qvalues = torch.tensor(subagent_qvalues, dtype=torch.float32).to(self.device)
            if subagent_qvalues.dim() == 2:  # (n_agents, 3)
                subagent_qvalues = subagent_qvalues.unsqueeze(0)  # (1, n_agents, 3)
            
            next_subagent_qvalues = torch.tensor(next_subagent_qvalues, dtype=torch.float32).to(self.device)
            if next_subagent_qvalues.dim() == 2:  # (n_agents, 3)
                next_subagent_qvalues = next_subagent_qvalues.unsqueeze(0)  # (1, n_agents, 3)
            
            # 動作和獎勵張量
            action_tensor = torch.tensor([action], dtype=torch.int64).to(self.device)
            reward_tensor = torch.tensor([reward], dtype=torch.float32).to(self.device)
            
            # 檢查張量是否包含 NaN 或 Inf
            tensors_to_check = [market_state, next_market_state, subagent_qvalues, next_subagent_qvalues, reward_tensor]
            tensor_names = ['market_state', 'next_market_state', 'subagent_qvalues', 'next_subagent_qvalues', 'reward']
            
            for i, (tensor, name) in enumerate(zip(tensors_to_check, tensor_names)):
                if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                    print(f"Warning: {name} contains NaN or Inf in Final Agent learning")
                    print(f"  Shape: {tensor.shape}, Mean: {tensor.mean().item():.6f}")
                    return
            
            # 計算當前 Q 值
            current_q_values = self.policy_network(market_state, subagent_qvalues)  # (1, 3)
            
            # 檢查 Q 值是否有效
            if torch.isnan(current_q_values).any() or torch.isinf(current_q_values).any():
                print("Warning: Current Q values contain NaN or Inf")
                print(f"  Q values: {current_q_values}")
                return
            
            # 計算目標 Q 值
            with torch.no_grad():
                next_q_values = self.target_network(next_market_state, next_subagent_qvalues)  # (1, 3)
                
                # 檢查目標 Q 值是否有效
                if torch.isnan(next_q_values).any() or torch.isinf(next_q_values).any():
                    print("Warning: Next Q values contain NaN or Inf")
                    print(f"  Next Q values: {next_q_values}")
                    return
                
                # Double DQN: 使用 policy network 選擇動作，target network 評估
                next_actions = self.policy_network(next_market_state, next_subagent_qvalues).argmax(1)  # (1,)
                next_q_value = next_q_values.gather(1, next_actions.unsqueeze(1)).squeeze(1)  # (1,)
                
                target_q_value = reward_tensor + (self.gamma * next_q_value)  # (1,)
                
                
            
            # 動作映射：環境動作 [-1, 0, 1] -> 網路索引 [0, 1, 2]
            action_mapping = {-1: 0, 0: 1, 1: 2}
            mapped_action = action_mapping[action_tensor.item()]
            mapped_action_tensor = torch.tensor([mapped_action], dtype=torch.long).to(self.device)  # (1,)
            
            # 選擇對應動作的 Q 值
            selected_q_value = current_q_values.gather(1, mapped_action_tensor.unsqueeze(1)).squeeze(1)  # (1,)
            
            # 確保兩個張量的形狀完全一致
            # print(f"Debug: selected_q_value shape: {selected_q_value.shape}, target_q_value shape: {target_q_value.shape}")
            # print(f"Debug: selected_q_value: {selected_q_value.item():.6f}, target_q_value: {target_q_value.item():.6f}")
            
            # 檢查選擇的 Q 值和目標值是否有效
            if torch.isnan(selected_q_value).any() or torch.isinf(selected_q_value).any():
                print("Warning: Selected Q value contains NaN or Inf")
                print(f"  Selected Q value: {selected_q_value}")
                return
            
            if torch.isnan(target_q_value).any() or torch.isinf(target_q_value).any():
                print("Warning: Target Q value contains NaN or Inf")
                print(f"  Target Q value: {target_q_value}")
                return
            
            # 計算損失 - 確保兩個張量形狀一致
            loss = F.mse_loss(selected_q_value, target_q_value)
            
            # 檢查損失是否有效
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: Loss is NaN or Inf: {loss}")
                print(f"  Selected Q: {selected_q_value.item():.6f}, Target Q: {target_q_value.item():.6f}")
                return
            
            # 反向傳播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 檢查梯度並計算梯度範數
            total_norm = 0
            param_count = 0
            for name, p in self.policy_network.named_parameters():
                if p.grad is not None:
                    if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                        print(f"Warning: Gradient contains NaN or Inf in parameter: {name}")
                        return
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                    param_count += 1
            
            total_norm = total_norm ** (1. / 2)
            
            
            self.optimizer.step()
            
            print(f"Final Agent MSE Loss ||y - Q(s,a;θ)||²: {loss.item():.6f}, Gradient Norm: {total_norm:.6f}")
        
        elif isinstance(experience, tuple) and len(experience) == 5:
            # 舊格式：(states, actions, rewards, next_states, current_subagent_qvalues)
            states, actions, rewards, next_states, current_subagent_qvalues = experience
            
            # 將 subagent qvalues 當作 next_subagent_qvalues（向後兼容）
            next_subagent_qvalues = current_subagent_qvalues
            
            # 遞迴調用新格式
            new_experience = (states, actions, rewards, next_states, current_subagent_qvalues, next_subagent_qvalues)
            self.learn(new_experience)
        
        else:
            print(f"Warning: Invalid experience format for Final Agent: {type(experience)}, length: {len(experience) if hasattr(experience, '__len__') else 'N/A'}")

    def decay_epsilon(self, episode):
        """
        根據 epsilon_decay 參數，衰減 epsilon 的值，並確保其不低於 epsilon_min。
        """
        # 指數衰減公式
        # self.epsilon = self.epsilon_min + (self.epsilon_max - self.epsilon_min) * math.exp(-self.epsilon_decay * episode)
        # 線性衰減公式
        self.epsilon = max(self.epsilon_min, self.epsilon_max - (self.epsilon_max - self.epsilon_min) * (episode / self.episodes))