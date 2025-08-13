import os
import numpy as np
import glob
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from data.data import FetchData
from utils.dataPreprocess import PreprocessData
from environment.maddqnEnv import MADDQNENV
from agents.subAgent import subAgent
from agents.finalAgent import finalAgent
import json
from datetime import datetime

class MADDQN:
    def __init__(self, configs, args):
        self.configs = configs
        self.args = args
        # 正確設定設備
        self.device = torch.device("cuda" if torch.cuda.is_available() and configs['gpu']['use_gpu'] else "cpu")
        print(f"Using device: {self.device}")
        
        self.returnAgentModel = subAgent(agentType='return', configs=self.configs)
        self.riskAgentModel = subAgent(agentType='risk', configs=self.configs)
        self.finalAgentModel = finalAgent(configs=self.configs)

        self.reward_stats = {
            'risk_agent': {'rewards': []},
            'return_agent': {'rewards': []},
            'final_agent': {'rewards': []}
        }
        
        # 添加 episode 層級的回報追蹤
        self.episode_returns = {
            'risk_agent': [],
            'return_agent': [],
            'final_agent': []
        }
        
        self.subEnvList = []

        self.test_dates = {
            'train_start_date': self.configs['data']['train_start_date'],
            'train_end_date': self.configs['data']['train_end_date'],
            'test_start_date': self.configs['data']['test_start_date'],
            'test_end_date': self.configs['data']['test_end_date']
        }

        self.test_prices = {
            'train_prices': [],
            'test_prices': []
        }


    def initialize(self):
        """
        初始化MADDQN系統的所有組件
        """
        print("正在初始化MADDQN系統...")
        
        # 初始化資料處理器和環境
        self.fetcher = FetchData(ticker=self.configs['data']['ticker'], start_date=self.configs['data']['train_start_date'], end_date=self.configs['data']['test_end_date'])
        self.unprocessed_data = self.fetcher.get_data()
        dates = [self.configs['data']['train_start_date'], self.configs['data']['train_end_date'], self.configs['data']['test_start_date'], self.configs['data']['test_end_date']]
        self.data_processor = PreprocessData(data=self.unprocessed_data, window_size=self.configs['env']['window_size'], dates=dates)
        self.train_data, self.test_data = self.data_processor.timeSeriesData()
        
        window_size = self.configs['env']['window_size']
    
        # 計算實際可用的數據索引範圍
        total_sequences = len(self.train_data) + len(self.test_data)
        available_data_start_idx = window_size - 1  # 第一個可用序列對應的原始數據索引
        available_data_end_idx = available_data_start_idx + total_sequences - 1
        
        # 實際的訓練集日期範圍
        self.actual_train_start_date = self.unprocessed_data.index[available_data_start_idx]
        train_end_idx = available_data_start_idx + len(self.train_data) - 1
        self.actual_train_end_date = self.unprocessed_data.index[train_end_idx]
        
        # 實際的測試集日期範圍  
        test_start_idx = train_end_idx + 1
        self.actual_test_start_date = self.unprocessed_data.index[test_start_idx]
        self.actual_test_end_date = self.unprocessed_data.index[available_data_end_idx]
        
        # 打印實際日期範圍與配置比較
        print(f"\n📅 日期範圍比較:")
        print(f"配置的訓練期間: {self.configs['data']['train_start_date']} ~ {self.configs['data']['train_end_date']}")
        print(f"實際的訓練期間: {self.actual_train_start_date.date()} ~ {self.actual_train_end_date.date()}")
        print(f"配置的測試期間: {self.configs['data']['test_start_date']} ~ {self.configs['data']['test_end_date']}")
        print(f"實際的測試期間: {self.actual_test_start_date.date()} ~ {self.actual_test_end_date.date()}")
        print(f"窗口大小影響: 前 {window_size-1} 天的數據用於構建窗口，無法作為獨立樣本")
    
        
        # Initialize all subagents' environments - 修正參數
        self.subEnvList = []
        for agentType in ['risk', 'return']:
            subEnv = MADDQNENV(
                configs=self.configs,
                tradeData=self.unprocessed_data,  # 移除 configs 參數
                window_size=self.configs['env']['window_size'],
                n_agents=self.configs['env'][f'{agentType}_agent']
            )
            self.subEnvList.append(subEnv)

        # Initialize the main environment - 修正參數
        self.env = MADDQNENV(
            configs=self.configs,
            tradeData=self.unprocessed_data,  # 移除 configs 參數
            window_size=self.configs['env']['window_size'],
            n_agents=self.configs['env']['risk_agent'] + self.configs['env']['return_agent']
        )
        
    def training_process(self):
        """
        執行 MADDQN 訓練過程 - 根據論文演算法實現
        Algorithm: Multi-Agent Deep Double Q-Learning with TimesNet
        """
        print("開始 MADDQN 訓練過程...")
        self.initialize()
        
        # 獲取訓練數據總長度
        total_sequences = len(self.train_data)
        print(f"Total training sequences: {total_sequences}")

        # 主訓練循環 - for episode = 1 to M do
        for episode in range(1, self.configs['training']['episodes'] + 1):
            print(f"\n=== Episode {episode}/{self.configs['training']['episodes']} ===")
            
            # 重置環境 - Initialize sequence s1 = {x1} and preprocessed state φ1 = φ(s1)
            self.env.reset()
            for subEnv in self.subEnvList:
                subEnv.reset()
            
            # 初始化episode變量
            episode_risk_return = 0.0
            episode_return_return = 0.0
            episode_final_return = 0.0
            current_episode_losses = {
                'risk_agent': [],
                'return_agent': [],
                'final_agent': []
            }
            
            # 存儲所有子代理的Q值 - Q-values of each agent needs to be stored
            num_agents = self.configs['env']['risk_agent'] + self.configs['env']['return_agent']
            max_timesteps = min(total_sequences, self.configs.get('training', {}).get('max_episode_steps', total_sequences))
            QValues_List = np.zeros((num_agents, max_timesteps, 3), dtype=np.float32)
            
            # 時間步循環 - 使用嵌套進度條
            timestep_pbar = tqdm(
                range(max_timesteps),
                desc=f"Ep{episode:02d} Steps",
                unit="step",
                leave=False,  # 不保留時間步進度條
                ncols=140,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
            )

            # 時間步循環 - for i=1 to T do
            for timestep in timestep_pbar:
                if timestep >= total_sequences:
                    break
                    
                # print(f"  Timestep {timestep + 1}/{max_timesteps}")
                
                # 獲取當前序列 si = {xi} and preprocessed state φi = φ(si)
                current_sequence = self.train_data[timestep]
                preprocessed_state = torch.tensor(current_sequence, dtype=torch.float32).to(self.device)
                
                # 更新環境狀態
                self.env.state = current_sequence
                old_state = current_sequence.copy()
                agent_rewards = []  # 用於存儲每個代理的獎勵
                
                # 子代理處理循環 - for j=1 to J do
                for agentIndex in range(num_agents):
                    # 確定代理類型和模型
                    if agentIndex < self.configs['env']['risk_agent']:
                        model = self.riskAgentModel
                        agent = 'risk'
                        subAgentEnv = self.subEnvList[0]
                    else:
                        model = self.returnAgentModel
                        agent = 'return'
                        subAgentEnv = self.subEnvList[1]
                    
                    # Select aj,i with ε-greedy method
                    # Otherwise, calculate Qj = Q(φ(si), aj; θj)
                    # Select aj,i = argmaxaj Qj
                    action, QValues = model.act(preprocessed_state)
                    QValues_List[agentIndex, timestep, :] = QValues
                    
                    # Sub-agent executes action aj,i in the environment
                    # Sub-agent gets reward rj,i and observes next state si+1, φi+1 = φ(si+1)
                    next_state, reward, done, info = subAgentEnv.step(
                        action, agentType=agent, tradeAmount=self.configs['env']['trade_amount']
                    )

                    agent_rewards.append(reward)  # 收集每個代理的獎勵
                    
                    # 累積獎勵
                    if agent == 'risk':
                        episode_risk_return += reward
                        # 記錄單步獎勵用於分佈分析
                        self.reward_stats['risk_agent']['rewards'].append(reward)
                    elif agent == 'return':
                        episode_return_return += reward
                        # 記錄單步獎勵用於分佈分析
                        self.reward_stats['return_agent']['rewards'].append(reward)
                    
                    # 檢查終止條件
                    is_last_timestep = (timestep == max_timesteps - 1)
                    episode_done = done or is_last_timestep
                    
                    # Store the transition (φi, aj,i, rj,i, φi+1) in Bj
                    model.replay_buffer.add({
                        'agent': agent,
                        'state': old_state,
                        'action': action,
                        'reward': reward,
                        'next_state': next_state,
                        'done': episode_done
                    })
                    
                    # 子代理學習 - for j=1 to J for each sub-agent j do
                    self._train_subagent_step(model, agent, current_episode_losses, timestep)
                
                # Final Agent處理
                # Select aF,i with ε-greedy method
                # Otherwise, select aF,i = argmaxaF Q(φ(si), Q1, Q2, ⋯, QJ, aF; θF)
                current_subagent_qvalues = QValues_List[:, timestep, :]
                final_action, final_q_values = self.finalAgentModel.act(
                    current_sequence, current_subagent_qvalues
                )
                
                # Final agent executes action aF,i in the environment
                # Final agent gets reward ri and observes next state si+1, φi+1 = φ(si+1)
                final_next_state, final_reward, final_done, info = self.env.step(
                    final_action, agentType='final', tradeAmount=self.configs['env']['trade_amount']
                )
                
                episode_final_return += final_reward
                # 記錄final agent的單步獎勵
                self.reward_stats['final_agent']['rewards'].append(final_reward)
                final_episode_done = final_done or is_last_timestep
                
                # 準備下一個timestep的子代理Q值
                next_timestep = min(timestep + 1, max_timesteps - 1)
                next_subagent_qvalues = QValues_List[:, next_timestep, :] if not final_episode_done else current_subagent_qvalues
                
                # Store the transition (φi, aF,i, rF,i, Q1, Q2, ⋯, QJ, φi+1) in BF
                self.finalAgentModel.replay_buffer.add({
                    'agent': 'final',
                    'state': old_state,
                    'action': final_action,
                    'reward': final_reward,
                    'subagent_qvalues': current_subagent_qvalues.copy(),
                    'next_state': final_next_state,
                    'next_subagent_qvalues': next_subagent_qvalues.copy(),
                    'done': final_episode_done
                })
                
                # Final Agent學習
                self._train_final_agent_step(current_episode_losses, timestep)
                
                # 目標網絡更新 - Update the parameters of the target network every C steps
                if (timestep + 1) % self.configs['training']['target_update_frequency'] == 0:
                    self._update_target_networks(timestep + 1)

                # 更新時間步進度條的後綴信息
                avg_reward = np.mean(agent_rewards) if agent_rewards else 0
                timestep_pbar.set_postfix({
                    'Risk_R': f'{episode_risk_return:.1f}',
                    'Return_R': f'{episode_return_return:.1f}',
                    'Final_R': f'{episode_final_return:.1f}'
                })

            # 關閉時間步進度條
            timestep_pbar.close()

            # Episode結束後的處理
            # Epsilon衰減
            self.returnAgentModel.decay_epsilon(episode=episode)
            self.riskAgentModel.decay_epsilon(episode=episode)
            self.finalAgentModel.decay_epsilon(episode=episode)

            # 記錄episode結果
            self._record_episode_results(episode, episode_risk_return, episode_return_return, 
                                        episode_final_return, current_episode_losses)

            
            # 每10個episode打印進度
            if episode % 10 == 0:
                self._print_training_progress(episode)
                # 分析獎勵分佈
                self.analyze_reward_distribution()
            
            # save models for all agents
            self.save_models(episode)

        # 訓練完成
        print("\nMADDQN 訓練過程完成！")
        
        # 最終獎勵分佈分析
        self.analyze_reward_distribution()
        self.plot_reward_distribution()
        self.plot_episode_returns()
        self.plot_loss_trends()
        self.plot_stock_history()
    
    def _train_subagent_step(self, model, agent, current_episode_losses, timestep):
        """
        訓練子代理的單步 - 直接實現DQN學習邏輯
        
        Args:
            model: 子代理模型 (subAgent)
            agent: 代理類型 ('risk' 或 'return')
            current_episode_losses: 當前episode的損失記錄字典
            timestep: 當前時間步
        """
        try:
            # 檢查replay buffer是否有足夠的樣本進行訓練
            if model.replay_buffer.__len__() < self.configs['training']['batch_size']:
                # print(f"    {agent.capitalize()} Agent: Insufficient samples in replay buffer (need {self.configs['training']['batch_size']}, have {model.replay_buffer.__len__()})")
                return
            
            # 從replay buffer中採樣一個batch
            minibatch = model.replay_buffer.sample(self.configs['training']['batch_size'])
            
            # 提取batch數據
            states = []
            actions = []
            rewards = []
            next_states = []
            dones = []
            
            for exp in minibatch:
                states.append(exp['state'])
                actions.append(exp['action'])
                rewards.append(exp['reward'])
                next_states.append(exp['next_state'])
                dones.append(exp['done'])
            
            # 轉換為張量
            states = torch.tensor(np.array(states), dtype=torch.float32).to(model.device)
            next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(model.device)
            rewards = torch.tensor(rewards, dtype=torch.float32).to(model.device)
            dones = torch.tensor(dones, dtype=torch.bool).to(model.device)
            
            # 處理動作：環境動作 [-1, 0, 1] -> 模型索引 [0, 1, 2]
            action_mapping = {-1: 0, 0: 1, 1: 2}
            mapped_actions = [action_mapping.get(action, 1) for action in actions]  # 默認為1(hold)
            actions = torch.tensor(mapped_actions, dtype=torch.long).to(model.device)
            
            # 確保狀態維度正確 (batch_size, seq_len, features)
            if states.dim() == 2:
                states = states.unsqueeze(1)  # 添加序列維度
            if next_states.dim() == 2:
                next_states = next_states.unsqueeze(1)
            
            # 當前Q值
            current_q_values = model.policy_network(states)  # [batch_size, 3]
            current_q_values_selected = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
            
            # 計算目標Q值
            with torch.no_grad():
                # Double DQN: 使用policy network選擇動作，target network評估Q值
                next_q_values_policy = model.policy_network(next_states)
                next_actions = next_q_values_policy.argmax(1)
                
                next_q_values_target = model.target_network(next_states)
                next_q_values_selected = next_q_values_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
                
                # 計算目標值：r + γ * max_a Q(s', a) * (1 - done)
                target_q_values = rewards + (model.gamma * next_q_values_selected * ~dones)
            
            # 計算損失
            loss = torch.nn.functional.mse_loss(current_q_values_selected, target_q_values)
            
            # 檢查損失是否有效
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"    {agent.capitalize()} Agent: Invalid loss detected (NaN/Inf), skipping update")
                return
            
            # 反向傳播
            model.optimizer.zero_grad()
            loss.backward()
            model.optimizer.step()
            
            # 記錄損失
            loss_value = loss.item()
            agent_key = f"{agent}_agent"
            current_episode_losses[agent_key].append(loss_value)
            # print(f"    {agent.capitalize()} Agent Loss at Timestep {timestep + 1}: {loss_value:.6f}")
            
        except Exception as e:
            print(f"    Error in {agent} agent training at timestep {timestep + 1}: {e}")
            import traceback
            traceback.print_exc()
    
    def _train_final_agent_step(self, current_episode_losses, timestep):
        """
        訓練最終代理的單步 - 直接實現DQN學習邏輯
        
        Args:
            current_episode_losses: 當前episode的損失記錄字典
            timestep: 當前時間步
        """
        try:
            # 檢查replay buffer是否有足夠的樣本進行訓練
            if self.finalAgentModel.replay_buffer.__len__() < self.configs['training']['batch_size']:
                # print(f"    Final Agent: Insufficient samples in replay buffer (need {self.configs['training']['batch_size']}, have {self.finalAgentModel.replay_buffer.__len__()})")
                return
            
            # 從replay buffer中採樣一個batch
            minibatch = self.finalAgentModel.replay_buffer.sample(self.configs['training']['batch_size'])
            
            # 提取batch數據
            market_states = []
            subagent_qvalues = []
            actions = []
            rewards = []
            next_market_states = []
            next_subagent_qvalues = []
            dones = []
            
            for exp in minibatch:
                market_states.append(exp['state'])
                subagent_qvalues.append(exp['subagent_qvalues'])
                actions.append(exp['action'])
                rewards.append(exp['reward'])
                next_market_states.append(exp['next_state'])
                next_subagent_qvalues.append(exp['next_subagent_qvalues'])
                dones.append(exp['done'])
            
            # 轉換為張量
            market_states = torch.tensor(np.array(market_states), dtype=torch.float32).to(self.finalAgentModel.device)
            subagent_qvalues = torch.tensor(np.array(subagent_qvalues), dtype=torch.float32).to(self.finalAgentModel.device)
            next_market_states = torch.tensor(np.array(next_market_states), dtype=torch.float32).to(self.finalAgentModel.device)
            next_subagent_qvalues = torch.tensor(np.array(next_subagent_qvalues), dtype=torch.float32).to(self.finalAgentModel.device)
            rewards = torch.tensor(rewards, dtype=torch.float32).to(self.finalAgentModel.device)
            dones = torch.tensor(dones, dtype=torch.bool).to(self.finalAgentModel.device)
            
            # 處理動作：環境動作 [-1, 0, 1] -> 模型索引 [0, 1, 2]
            action_mapping = {-1: 0, 0: 1, 1: 2}
            mapped_actions = [action_mapping.get(action, 1) for action in actions]  # 默認為1(hold)
            actions = torch.tensor(mapped_actions, dtype=torch.long).to(self.finalAgentModel.device)
            
            # 確保狀態維度正確
            if market_states.dim() == 2:
                market_states = market_states.unsqueeze(1)  # 添加序列維度
            if next_market_states.dim() == 2:
                next_market_states = next_market_states.unsqueeze(1)
            
            # 確保子代理Q值維度正確
            if subagent_qvalues.dim() == 2:
                # (batch_size, n_agents*3) -> (batch_size, n_agents, 3)
                n_agents = self.configs['env']['risk_agent'] + self.configs['env']['return_agent']
                subagent_qvalues = subagent_qvalues.reshape(-1, n_agents, 3)
            if next_subagent_qvalues.dim() == 2:
                n_agents = self.configs['env']['risk_agent'] + self.configs['env']['return_agent']
                next_subagent_qvalues = next_subagent_qvalues.reshape(-1, n_agents, 3)
            
            # 將子代理Q值展平為輸入特徵
            batch_size = subagent_qvalues.size(0)
            subagent_qvalues_flat = subagent_qvalues.view(batch_size, -1)  # [batch_size, n_agents*3]
            next_subagent_qvalues_flat = next_subagent_qvalues.view(batch_size, -1)
            
            # 當前Q值 - MSCNN需要市場數據和子代理Q值作為輸入
            current_q_values = self.finalAgentModel.policy_network(market_states, subagent_qvalues_flat)  # [batch_size, 3]
            current_q_values_selected = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
            
            # 計算目標Q值
            with torch.no_grad():
                # Double DQN: 使用policy network選擇動作，target network評估Q值
                next_q_values_policy = self.finalAgentModel.policy_network(next_market_states, next_subagent_qvalues_flat)
                next_actions = next_q_values_policy.argmax(1)
                
                next_q_values_target = self.finalAgentModel.target_network(next_market_states, next_subagent_qvalues_flat)
                next_q_values_selected = next_q_values_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
                
                # 計算目標值：r + γ * max_a Q(s', a) * (1 - done)
                target_q_values = rewards + (self.finalAgentModel.gamma * next_q_values_selected * ~dones)
            
            # 計算損失
            loss = torch.nn.functional.mse_loss(current_q_values_selected, target_q_values)
            
            # 檢查損失是否有效
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"    Final Agent: Invalid loss detected (NaN/Inf), skipping update")
                return
            
            # 反向傳播
            self.finalAgentModel.optimizer.zero_grad()
            loss.backward()
            self.finalAgentModel.optimizer.step()
            
            # 記錄損失
            loss_value = loss.item()
            current_episode_losses['final_agent'].append(loss_value)
            # print(f"    Final Agent Loss at Timestep {timestep + 1}: {loss_value:.6f}")
            
        except Exception as e:
            print(f"    Error in final agent training at timestep {timestep + 1}: {e}")
            import traceback
            traceback.print_exc()
    
    def _update_target_networks(self, timestep):
        """
        更新所有子代理和最終代理的目標網絡
        """
        self.returnAgentModel.target_network.load_state_dict(self.returnAgentModel.policy_network.state_dict())
        self.riskAgentModel.target_network.load_state_dict(self.riskAgentModel.policy_network.state_dict())
        self.finalAgentModel.target_network.load_state_dict(self.finalAgentModel.policy_network.state_dict())
        # print(f"  Target networks updated at Timestep {timestep + 1}")
    

    def _record_episode_results(self, episode, risk_return, return_return, final_return, losses):
        """
        記錄每個episode的每個代理的回報率和損失值
        """
        # 記錄 episode 層級的回報 - 這是修正的重點
        self.episode_returns['risk_agent'].append(risk_return)
        self.episode_returns['return_agent'].append(return_return)
        self.episode_returns['final_agent'].append(final_return)
        print(f"Episode {episode} Results:")
        print(f"  Risk Agent Return: {risk_return:.2f}, Losses: {np.mean(losses['risk_agent']):.4f}")
        print(f"  Return Agent Return: {return_return:.2f}, Losses: {np.mean(losses['return_agent']):.4f}")
        print(f"  Final Agent Return: {final_return:.2f}, Losses: {np.mean(losses['final_agent']):.4f}")
    
    def _print_training_progress(self, episode):
        """
        打印訓練進度
        """
        print(f"\n=== Training Progress after Episode {episode} ===")
        print(f"  Risk Agent Epsilon: {self.riskAgentModel.epsilon:.3f}")
        print(f"  Return Agent Epsilon: {self.returnAgentModel.epsilon:.3f}")
        print(f"  Final Agent Epsilon: {self.finalAgentModel.epsilon:.3f}")
    
    def analyze_reward_distribution(self):
        """
        分析每個代理的獎勵分佈
        """
        print("\n=== Reward Distribution Analysis ===")
        for agent, stats in self.reward_stats.items():
            rewards = stats['rewards']
            if rewards:
                mean_reward = np.mean(rewards)
                std_reward = np.std(rewards)
                print(f"  {agent.capitalize()} Agent - Mean Reward: {mean_reward:.2f}, Std Dev: {std_reward:.2f}")
            else:
                print(f"  {agent.capitalize()} Agent - No rewards recorded.")
        print("=== End of Reward Distribution Analysis ===\n")
    
    def plot_reward_distribution(self):
        """
        繪製每個代理的獎勵分佈圖
        """
        
        plt.figure(figsize=(12, 6))
        for agent, stats in self.reward_stats.items():
            rewards = stats['rewards']
            if rewards:
                plt.hist(rewards, bins=50, alpha=0.5, label=f"{agent.capitalize()} Agent")
        
        plt.title("Reward Distribution of Agents")
        plt.xlabel("Reward")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid()
        plt.savefig("reward_distribution.png")

    def plot_episode_returns(self):
        """
        繪製每個episode的回報率趨勢圖
        """
        episodes = list(range(1, self.configs['training']['episodes'] + 1))
        risk_returns = self.episode_returns['risk_agent']
        return_returns = self.episode_returns['return_agent'] 
        final_returns = self.episode_returns['final_agent']
        
        # 如果episode數量不匹配，進行調整
        expected_episodes = self.configs['training']['episodes']
        if len(risk_returns) < expected_episodes:
            # 如果數據不足，用0填充
            risk_returns.extend([0] * (expected_episodes - len(risk_returns)))
            return_returns.extend([0] * (expected_episodes - len(return_returns)))
            final_returns.extend([0] * (expected_episodes - len(final_returns)))
        elif len(risk_returns) > expected_episodes:
            # 如果數據過多，截取
            risk_returns = risk_returns[:expected_episodes]
            return_returns = return_returns[:expected_episodes]
            final_returns = final_returns[:expected_episodes]

        plt.figure(figsize=(12, 6))
        plt.plot(episodes, risk_returns, label='Risk Agent Returns', marker='o', alpha=0.7)
        plt.plot(episodes, return_returns, label='Return Agent Returns', marker='o', alpha=0.7)
        plt.plot(episodes, final_returns, label='Final Agent Returns', marker='o', alpha=0.7)
        
        plt.title("Episode Returns of Agents")
        plt.xlabel("Episode")
        plt.ylabel("Cumulative Return per Episode")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig("episode_returns.png", dpi=300, bbox_inches='tight')
        plt.show()
        print("📊 Episode回報趨勢圖已保存為 'episode_returns.png'")
    
    def plot_loss_trends(self):
        """
        繪製每個代理的損失趨勢圖
        """
        episodes = list(range(1, self.configs['training']['episodes'] + 1))
        
        risk_losses = [np.mean(self.reward_stats['risk_agent']['rewards'][i::self.configs['training']['episodes']]) for i in range(self.configs['training']['episodes'])]
        return_losses = [np.mean(self.reward_stats['return_agent']['rewards'][i::self.configs['training']['episodes']]) for i in range(self.configs['training']['episodes'])]
        final_losses = [np.mean(self.reward_stats['final_agent']['rewards'][i::self.configs['training']['episodes']]) for i in range(self.configs['training']['episodes'])]

        plt.figure(figsize=(12, 6))
        plt.plot(episodes, risk_losses, label='Risk Agent Losses', marker='o')
        plt.plot(episodes, return_losses, label='Return Agent Losses', marker='o')
        plt.plot(episodes, final_losses, label='Final Agent Losses', marker='o')
        
        plt.title("Loss Trends of Agents")
        plt.xlabel("Episode")
        plt.ylabel("Average Loss")
        plt.legend()
        plt.grid()
        plt.savefig("loss_trends.png")

    def plot_stock_history(self):
        """
        繪製股票歷史價格圖
        """
        
        try:
            # 檢查數據結構並正確獲取日期和價格
            dates = self.unprocessed_data.index  # 已確認index是DatetimeIndex
            
            # 處理MultiIndex列結構：('Close', 'DJI')
            if ('Close', 'DJI') in self.unprocessed_data.columns:
                prices = self.unprocessed_data[('Close', 'DJI')]
            elif 'Close' in [col[0] if isinstance(col, tuple) else col for col in self.unprocessed_data.columns]:
                # 找到Close相關的列
                close_col = None
                for col in self.unprocessed_data.columns:
                    if isinstance(col, tuple) and col[0] == 'Close':
                        close_col = col
                        break
                    elif col == 'Close':
                        close_col = col
                        break
                if close_col:
                    prices = self.unprocessed_data[close_col]
                    print(f"Using column '{close_col}' as close price")
                else:
                    print("Error: No close price column found")
                    return
            else:
                print("Error: No close price column found")
                print(f"Available columns: {list(self.unprocessed_data.columns)}")
                return
            
            # 確保價格是數值類型並轉換為純數值
            prices = prices.values  # 轉換為numpy數組
            prices = np.array(prices, dtype=float)  # 確保是float類型
            
            # 獲取純數值而不是數組
            start_price = float(prices[0]) if len(prices) > 0 else 0.0
            end_price = float(prices[-1]) if len(prices) > 0 else 0.0
            total_return = ((end_price - start_price) / start_price * 100) if start_price != 0 else 0.0
            
            # 創建圖表
            plt.figure(figsize=(12, 6))
            plt.plot(dates, prices, label='Close Price', color='blue', linewidth=1.5)
            
            # 設置圖表屬性
            plt.title("Stock Price History (Close Price Only)", fontsize=14, fontweight='bold')
            plt.xlabel("Date")
            plt.ylabel("Price ($)")
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 如果日期太多，自動調整x軸標籤
            if len(dates) > 50:
                plt.xticks(rotation=45)
                # 只顯示部分日期標籤
                step = max(1, len(dates) // 10)
                selected_dates = dates[::step]
                plt.xticks(selected_dates, rotation=45)
            
            plt.tight_layout()
            
            # 在圖上添加統計信息 - 現在使用純數值
            stats_text = f'Start: ${start_price:.2f}\nEnd: ${end_price:.2f}\nReturn: {total_return:.2f}%'
            plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            plt.show()
            print("Stock price history plotted successfully.")
            
            # 打印數據信息用於調試
            print(f"Data shape: {self.unprocessed_data.shape}")
            print(f"Columns: {list(self.unprocessed_data.columns)}")
            print(f"Index name: {self.unprocessed_data.index.name}")
            print(f"Price range: ${start_price:.2f} - ${end_price:.2f}")
            print(f"Total return: {total_return:.2f}%")
            
        except Exception as e:
            print(f"Error plotting stock history: {e}")
            print(f"Data columns: {list(self.unprocessed_data.columns)}")
            print(f"Data shape: {self.unprocessed_data.shape}")
            print(f"Index: {self.unprocessed_data.index}")
            import traceback
            traceback.print_exc()
    

    def testing_process(self):
        """
        執行 MADDQN 測試過程 - 修正累積回報計算
        """
        print("開始 MADDQN 測試過程...")
        
        
        
        # 重置環境
        self.initialize()
        self.env.reset()
        for subEnv in self.subEnvList:
            subEnv.reset()
        # 獲取測試數據總長度
        total_sequences = len(self.test_data)
        print(f"Total testing sequences: {total_sequences}")

        # choose the most recently model all agents
        model_paths_pattern = glob.glob(os.path.join(self.configs['training']['model_save_dir'], "*agent_*.pth"))
        risk_agent_model_path =  [path for path in model_paths_pattern if path.startswith("./checkpoints/risk_agent")]
        return_agent_model_path = [path for path in model_paths_pattern if path.startswith("./checkpoints/return_agent")]
        final_agent_model_path = [path for path in model_paths_pattern if path.startswith("./checkpoints/final_agent")]
        sorted(risk_agent_model_path, key=os.path.getmtime)
        sorted(return_agent_model_path, key=os.path.getmtime)
        sorted(final_agent_model_path, key=os.path.getmtime)
        model_path = [
            risk_agent_model_path[0] if risk_agent_model_path else None,
            return_agent_model_path[0] if return_agent_model_path else None,
            final_agent_model_path[0] if final_agent_model_path else None
        ]
        try:
            self.load_models(model_paths=model_path)
            print("模型加載成功，開始測試過程...")
        except Exception as e:
            print(f"模型載入失敗: {e}")
            return

        # 初始化變量
        num_agents = self.configs['env']['risk_agent'] + self.configs['env']['return_agent']
        QValues_List = np.zeros((num_agents, total_sequences, 3), dtype=np.float32)
        
        # 修正：初始化投資組合跟蹤變量
        initial_balance = float(self.configs['env']['initial_balance'])
        
        # 累積回報追蹤 - 以百分比形式累積
        cumulative_returns = {
            'risk_agent': 0.0,
            'return_agent': 0.0,
            'final_agent': 0.0
        }
        
        # 投資組合價值追蹤 - 以實際金額追蹤
        portfolio_values = {
            'risk_agent': initial_balance,
            'return_agent': initial_balance,
            'final_agent': initial_balance
        }
        
        # 單步獎勵累積
        episode_rewards = {
            'risk_agent': 0.0,
            'return_agent': 0.0,
            'final_agent': 0.0
        }
        
        # 主測試循環
        timestep_pbar = tqdm(
            range(total_sequences), 
            desc="Testing", 
            unit="step",
            ncols=200,
            leave=True,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}'
        )
        
        # 記錄第一個價格作為基準
        first_price = None
        
        for timestep in timestep_pbar:
            if timestep >= total_sequences:
                break
            
            # 獲取當前數據
            current_sequence = self.test_data[timestep]
            # test_start_idx = len(self.train_data)
            # current_data_index = test_start_idx + timestep
            actual_test_start_idx = self.configs['env']['window_size'] - 1 + len(self.train_data)
            current_data_index = actual_test_start_idx + timestep
            
            if current_data_index >= len(self.unprocessed_data):
                current_data_index = len(self.unprocessed_data) - 1
            
            # 獲取當前價格
            current_price = self.unprocessed_data.iloc[current_data_index]['Close']
            if hasattr(current_price, 'item'):
                current_price = current_price.item()
            else:
                current_price = float(current_price)
            
            # 記錄第一個價格
            if first_price is None:
                first_price = current_price
            
            current_date = self.unprocessed_data.index[current_data_index]
            preprocessed_state = torch.tensor(current_sequence, dtype=torch.float32).to(self.device)
            
            # 更新環境狀態
            self.env.state = current_sequence
            
            # 當前時間步的獎勵
            timestep_rewards = {
                'risk_agent': 0.0,
                'return_agent': 0.0,
                'final_agent': 0.0
            }
            
            # 子代理處理循環
            for agentIndex in range(num_agents):
                # 確定代理類型和模型
                if agentIndex < self.configs['env']['risk_agent']:
                    model = self.riskAgentModel
                    agent = 'risk'
                    subAgentEnv = self.subEnvList[0]
                    agent_key = 'risk_agent'
                else:
                    model = self.returnAgentModel
                    agent = 'return'
                    subAgentEnv = self.subEnvList[1]
                    agent_key = 'return_agent'
                
                # 使用訓練好的模型進行預測
                with torch.no_grad():
                    action, QValues = model.act(preprocessed_state, training=False)
                    QValues_List[agentIndex, timestep, :] = QValues
                    
                    # 確保動作在正確範圍內
                    if action not in [-1, 0, 1]:
                        if action in [0, 1, 2]:
                            action_mapping = {0: -1, 1: 0, 2: 1}
                            action = action_mapping[action]
                        else:
                            action = 0

                # 執行動作並獲取獎勵
                next_state, reward, done, info = subAgentEnv.step(
                    action, agentType=agent, tradeAmount=self.configs['env']['trade_amount']
                )
                
                # 累積獎勵
                timestep_rewards[agent_key] += reward
                episode_rewards[agent_key] += reward
            
            # Final Agent處理
            current_subagent_qvalues = QValues_List[:, timestep, :]
            
            with torch.no_grad():
                finalAgent_action, final_q_values = self.finalAgentModel.act(
                    current_sequence, current_subagent_qvalues, training=False
                )
                
                # 處理 Final Agent 動作
                if isinstance(finalAgent_action, torch.Tensor):
                    if finalAgent_action.dim() == 0:
                        final_action = finalAgent_action.item()
                    else:
                        final_action = finalAgent_action.argmax().item()
                else:
                    final_action = finalAgent_action
                
                # 確保動作有效
                if final_action in [0, 1, 2]:
                    action_mapping = {0: -1, 1: 0, 2: 1}
                    final_action = action_mapping[final_action]
                elif final_action not in [-1, 0, 1]:
                    final_action = 0
            
            # Final agent執行動作
            final_next_state, final_reward, final_done, info = self.env.step(
                final_action, agentType='final', tradeAmount=self.configs['env']['trade_amount']
            )
            
            timestep_rewards['final_agent'] = final_reward
            episode_rewards['final_agent'] += final_reward
            
            # 修正：計算累積回報率
            # 方法1: 基於獎勵的累積回報率
            for agent_key in ['risk_agent', 'return_agent', 'final_agent']:
                # 將獎勵轉換為回報率（假設獎勵已經是百分比形式）
                step_return_pct = timestep_rewards[agent_key]
                
                # 累積回報率計算：(1 + r1) * (1 + r2) * ... - 1
                cumulative_returns[agent_key] = ((1 + cumulative_returns[agent_key]/100) * 
                                            (1 + step_return_pct/100) - 1) * 100
                
                # 更新投資組合價值
                portfolio_values[agent_key] = initial_balance * (1 + cumulative_returns[agent_key]/100)
            
            # 方法2: 基於價格變動的累積回報率（作為參考）
            market_return = ((current_price - first_price) / first_price) * 100
            
            # 更新進度條信息
            timestep_pbar.set_postfix({
                'Risk': f'{cumulative_returns["risk_agent"]:.3f}%',
                'Return': f'{cumulative_returns["return_agent"]:.3f}%',
                'Final': f'{cumulative_returns["final_agent"]:.3f}%',
                'Market': f'{market_return:.3f}%',
                'Price': f'{current_price:.0f}'
            })
        
        print("\nMADDQN 測試過程完成！")
        
        # 打印詳細結果
        print("\n" + "="*80)
        print("📊 測試結果總結")
        print("="*80)
        
        print(f"\n📈 期間價格變動:")
        print(f"   起始價格: ${first_price:.2f}")
        print(f"   結束價格: ${current_price:.2f}")
        print(f"   市場回報: {market_return:.4f}%")
        
        print(f"\n💰 各Agent表現:")
        for agent_key in ['risk_agent', 'return_agent', 'final_agent']:
            agent_name = agent_key.replace('_', ' ').title()
            print(f"   {agent_name}:")
            print(f"     總獎勵: {episode_rewards[agent_key]:.6f}")
            print(f"     累積回報率: {cumulative_returns[agent_key]:.4f}%")
            print(f"     最終價值: ${portfolio_values[agent_key]:,.2f}")
            print(f"     相對市場: {cumulative_returns[agent_key] - market_return:.4f}%")
            print()
        
        # 找出最佳表現
        best_agent = max(cumulative_returns.keys(), key=lambda k: cumulative_returns[k])
        worst_agent = min(cumulative_returns.keys(), key=lambda k: cumulative_returns[k])
        
        print(f"🏆 最佳表現: {best_agent.replace('_', ' ').title()} ({cumulative_returns[best_agent]:.4f}%)")
        print(f"📉 最差表現: {worst_agent.replace('_', ' ').title()} ({cumulative_returns[worst_agent]:.4f}%)")
        
        # 計算Alpha (相對市場的超額回報)
        print(f"\n📊 Alpha分析 (相對市場的超額回報):")
        for agent_key in ['risk_agent', 'return_agent', 'final_agent']:
            alpha = cumulative_returns[agent_key] - market_return
            agent_name = agent_key.replace('_', ' ').title()
            status = "🔥 超越市場" if alpha > 0 else "📉 跑輸市場" if alpha < 0 else "⚖️ 持平市場"
            print(f"   {agent_name}: {alpha:.4f}% {status}")
        
        print("="*80)
        
        # 保存結果到屬性中，供後續繪圖使用
        self.test_cumulative_returns = cumulative_returns
        self.test_portfolio_values_final = portfolio_values
        self.test_episode_rewards = episode_rewards
        self.market_return = market_return
        
        # 繪製測試結果
        self.plot_test_results()
        self.save_test_statistics()
        
        return {
            'cumulative_returns': cumulative_returns,
            'portfolio_values': portfolio_values,
            'episode_rewards': episode_rewards,
            'market_return': market_return
        }

    def plot_test_results(self):
        """
        繪製測試結果圖表
        """
        try:
            # 檢查必需的屬性是否存在
            if not hasattr(self, 'test_cumulative_returns'):
                print("⚠️ 測試結果數據不存在，無法繪圖")
                return
            
            # 創建圖表
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('MADDQN Testing Results Analysis', fontsize=16, fontweight='bold')
            
            # 1. 累積回報率比較
            agents = list(self.test_cumulative_returns.keys())
            returns = list(self.test_cumulative_returns.values())
            returns.append(self.market_return)  # 添加市場回報
            agents.append('Market')
            
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
            bars = ax1.bar(range(len(agents)), returns, color=colors)
            ax1.set_title('Cumulative Returns Comparison', fontweight='bold')
            ax1.set_ylabel('Return (%)')
            ax1.set_xticks(range(len(agents)))
            ax1.set_xticklabels([agent.replace('_', ' ').title() for agent in agents], rotation=45)
            ax1.grid(True, alpha=0.3)
            
            # 添加數值標籤
            for bar, value in zip(bars, returns):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{value:.3f}%', ha='center', va='bottom' if height >= 0 else 'top')
            
            # 2. 投資組合價值對比
            portfolio_values = list(self.test_portfolio_values_final.values())
            ax2.bar(range(len(agents)-1), portfolio_values, color=colors[:-1])
            ax2.set_title('Final Portfolio Values', fontweight='bold')
            ax2.set_ylabel('Value ($)')
            ax2.set_xticks(range(len(agents)-1))
            ax2.set_xticklabels([agent.replace('_', ' ').title() for agent in agents[:-1]], rotation=45)
            ax2.grid(True, alpha=0.3)
            
            # 添加初始值線
            initial_balance = float(self.configs['env']['initial_balance'])
            ax2.axhline(y=initial_balance, color='red', linestyle='--', alpha=0.7, label=f'Initial: ${initial_balance:,.0f}')
            ax2.legend()
            
            # 3. Alpha分析（相對市場超額回報）
            alphas = [self.test_cumulative_returns[agent] - self.market_return 
                    for agent in ['risk_agent', 'return_agent', 'final_agent']]
            agent_names = [agent.replace('_', ' ').title() for agent in ['risk_agent', 'return_agent', 'final_agent']]
            
            colors_alpha = ['green' if alpha > 0 else 'red' for alpha in alphas]
            bars_alpha = ax3.bar(range(len(agent_names)), alphas, color=colors_alpha, alpha=0.7)
            ax3.set_title('Alpha (Excess Return vs Market)', fontweight='bold')
            ax3.set_ylabel('Alpha (%)')
            ax3.set_xticks(range(len(agent_names)))
            ax3.set_xticklabels(agent_names, rotation=45)
            ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            ax3.grid(True, alpha=0.3)
            
            # 添加數值標籤
            for bar, value in zip(bars_alpha, alphas):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'{value:.3f}%', ha='center', va='bottom' if height >= 0 else 'top')
            
            # 4. 總獎勵對比
            total_rewards = list(self.test_episode_rewards.values())
            ax4.bar(range(len(agent_names)), total_rewards, color=colors[:-1])
            ax4.set_title('Total Episode Rewards', fontweight='bold')
            ax4.set_ylabel('Total Reward')
            ax4.set_xticks(range(len(agent_names)))
            ax4.set_xticklabels(agent_names, rotation=45)
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # 保存圖表
            plt.savefig('maddqn_test_results.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            print("📊 測試結果圖表已保存為 'maddqn_test_results.png'")
            
        except Exception as e:
            print(f"繪圖過程中發生錯誤: {e}")
            import traceback
            traceback.print_exc()

    def save_test_statistics(self):
        """
        保存測試統計結果到文件
        """
        try:
            # 獲取測試期間的實際日期和價格
            test_start_idx = len(self.train_data)
            test_end_idx = test_start_idx + len(self.test_data) - 1
            
            # 確保索引不超出範圍
            test_end_idx = min(test_end_idx, len(self.unprocessed_data) - 1)
            
            # 獲取測試期間的日期
            test_start_date = self.unprocessed_data.index[test_start_idx]
            test_end_date = self.unprocessed_data.index[test_end_idx]
            
            # 獲取測試期間的價格
            # 處理MultiIndex列結構
            if ('Close', 'DJI') in self.unprocessed_data.columns:
                close_col = ('Close', 'DJI')
            else:
                # 找到Close相關的列
                close_col = None
                for col in self.unprocessed_data.columns:
                    if isinstance(col, tuple) and col[0] == 'Close':
                        close_col = col
                        break
                    elif col == 'Close':
                        close_col = col
                        break
            
            if close_col:
                test_start_price = float(self.unprocessed_data.iloc[test_start_idx][close_col])
                test_end_price = float(self.unprocessed_data.iloc[test_end_idx][close_col])
            else:
                test_start_price = 0.0
                test_end_price = 0.0
            
            # 準備統計數據
            stats = {
                'test_date': datetime.now().isoformat(),
                'test_period': {
                    'start_date': str(test_start_date),
                    'end_date': str(test_end_date),
                    'total_days': len(self.test_data)
                },
                'market_data': {
                    'start_price': test_start_price,
                    'end_price': test_end_price,
                    'market_return': float(self.market_return)
                },
                'agent_performance': {
                    agent: {
                        'cumulative_return_pct': float(self.test_cumulative_returns[agent]),
                        'total_rewards': float(self.test_episode_rewards[agent]),
                        'final_portfolio_value': float(self.test_portfolio_values_final[agent]),
                        'alpha_vs_market': float(self.test_cumulative_returns[agent] - self.market_return)
                    }
                    for agent in ['risk_agent', 'return_agent', 'final_agent']
                },
                'performance_metrics': {
                    'best_agent': max(self.test_cumulative_returns.keys(), 
                                    key=lambda k: self.test_cumulative_returns[k]),
                    'worst_agent': min(self.test_cumulative_returns.keys(), 
                                    key=lambda k: self.test_cumulative_returns[k]),
                    'agents_beating_market': [agent for agent in ['risk_agent', 'return_agent', 'final_agent'] 
                                            if self.test_cumulative_returns[agent] > self.market_return]
                },
                'config_used': {
                    'data': self.configs['data'],
                    'env': self.configs['env'],
                    'training': {k: v for k, v in self.configs['training'].items() 
                            if k not in ['risk_agent_model_path', 'return_agent_model_path', 'final_agent_model_path']}
                }
            }
            
            # 保存到JSON文件
            filename = f"maddqn_test_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2, ensure_ascii=False, default=str)
            
            print(f"📄 測試統計結果已保存為 '{filename}'")
            
            # 打印關鍵統計信息
            print(f"\n📋 關鍵統計信息:")
            print(f"   測試期間: {test_start_date.date()} 至 {test_end_date.date()}")
            print(f"   測試天數: {len(self.test_data)} 天")
            print(f"   市場回報: {self.market_return:.4f}%")
            print(f"   最佳Agent: {stats['performance_metrics']['best_agent'].replace('_', ' ').title()}")
            print(f"   超越市場的Agent數量: {len(stats['performance_metrics']['agents_beating_market'])}")
            
        except Exception as e:
            print(f"保存統計結果時發生錯誤: {e}")
            import traceback
            traceback.print_exc()
    
    def load_models(self, model_paths):
        """
        載入訓練好的模型
        
        Args:
            model_paths: 包含三個模型路徑的列表 [risk_agent_path, return_agent_path, final_agent_path]
        """
        try:
            risk_agent_path, return_agent_path, final_agent_path = model_paths
            
            # 載入 Risk Agent 模型
            if risk_agent_path and os.path.exists(risk_agent_path):
                risk_state_dict = torch.load(risk_agent_path, map_location=self.device, weights_only=True)
                self.riskAgentModel.policy_network.load_state_dict(risk_state_dict)
                self.riskAgentModel.target_network.load_state_dict(risk_state_dict)
                print(f"✅ Risk Agent 模型已載入: {risk_agent_path}")
            else:
                print(f"⚠️ Risk Agent 模型路徑無效或檔案不存在: {risk_agent_path}")
                
            # 載入 Return Agent 模型
            if return_agent_path and os.path.exists(return_agent_path):
                return_state_dict = torch.load(return_agent_path, map_location=self.device, weights_only=True)
                self.returnAgentModel.policy_network.load_state_dict(return_state_dict)
                self.returnAgentModel.target_network.load_state_dict(return_state_dict)
                print(f"✅ Return Agent 模型已載入: {return_agent_path}")
            else:
                print(f"⚠️ Return Agent 模型路徑無效或檔案不存在: {return_agent_path}")
                
            # 載入 Final Agent 模型
            if final_agent_path and os.path.exists(final_agent_path):
                final_state_dict = torch.load(final_agent_path, map_location=self.device, weights_only=True)
                self.finalAgentModel.policy_network.load_state_dict(final_state_dict)
                self.finalAgentModel.target_network.load_state_dict(final_state_dict)
                print(f"✅ Final Agent 模型已載入: {final_agent_path}")
            else:
                print(f"⚠️ Final Agent 模型路徑無效或檔案不存在: {final_agent_path}")
            
            # 設置為評估模式
            self.riskAgentModel.policy_network.eval()
            self.riskAgentModel.target_network.eval()
            self.returnAgentModel.policy_network.eval()
            self.returnAgentModel.target_network.eval()
            self.finalAgentModel.policy_network.eval()
            self.finalAgentModel.target_network.eval()
            
            print("📋 所有模型已設置為評估模式")
            
        except Exception as e:
            print(f"❌ 模型載入過程中發生錯誤: {e}")
            raise e

    def save_models(self, episode=None):
        """
        保存訓練好的模型
        
        Args:
            episode: 當前episode數 (可選，用於檔名)
        """
        try:
            
            # 創建模型保存目錄
            save_dir = self.configs['training']['model_save_dir']
            
            # 生成檔名
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            episode_suffix = f"_ep{episode}" if episode else ""
            
            # 保存 Risk Agent
            risk_path = os.path.join(save_dir, f'risk_agent{episode_suffix}_{timestamp}.pth')
            torch.save(self.riskAgentModel.policy_network.state_dict(), risk_path)
            print(f"✅ Risk Agent 模型已保存: {risk_path}")
            
            # 保存 Return Agent  
            return_path = os.path.join(save_dir, f'return_agent{episode_suffix}_{timestamp}.pth')
            torch.save(self.returnAgentModel.policy_network.state_dict(), return_path)
            print(f"✅ Return Agent 模型已保存: {return_path}")
            
            # 保存 Final Agent
            final_path = os.path.join(save_dir, f'final_agent{episode_suffix}_{timestamp}.pth')
            torch.save(self.finalAgentModel.policy_network.state_dict(), final_path)
            print(f"✅ Final Agent 模型已保存: {final_path}")
            
            
            return {
                'risk_agent_path': risk_path,
                'return_agent_path': return_path,
                'final_agent_path': final_path
            }
            
        except Exception as e:
            print(f"❌ 模型保存過程中發生錯誤: {e}")
            raise e