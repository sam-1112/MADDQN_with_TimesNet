import numpy as np
from dataclasses import dataclass

@dataclass
class Observation:
    price: float
    volume: int
    initial_balance: float
    balance: float
    position: int
    position_state: int = 0  # 新增：持倉狀態 {-1, 0, 1}



class MADDQNENV:
    def __init__(self, tradeData, window_size=10, n_agents=3):
        self.tradeData = tradeData
        self.window_size = window_size
        self.n_agents = n_agents
        self.agent = 'final'
        self.current_step = 0
        self.state = None
        self.done = False
        self.action_space = np.array([-1, 0, 1])  # Example action space: 0 for hold, 1 for sell, -1 for buy
        self.history = []  # To store the history of actions and rewards

        self.observation = Observation(
            price=0.0,  # Placeholder, will be updated in reset
            volume=0,   # Placeholder, will be updated in reset
            initial_balance=100000000.0,  # Initial balance
            balance=100000000.0,  # Initial balance
            position=0,  # Initial position
            position_state=0  # 初始化持倉狀態為0（無持倉）
        )
        # Initialize the environment
        self.reset()

    def reset(self):
        self.current_step = 0
        self.done = False
        self.state = self.tradeData.iloc[self.current_step:self.current_step + self.window_size].values
        # self.state = self.tradeData[self.current_step:self.current_step + self.window_size]
        self.history = []
        self.agent = 'final'  # Reset agent type
        # Reset observation
        # Get the last row of the state window
        last_row = self.state[-1]
        self.observation = Observation(
            price=last_row[0],  # First column (assuming it's price)
            volume=last_row[-1],  # Second column (assuming it's volume)
            balance=100000000.0,  # Reset balance
            initial_balance=100000000.0,  # Initial balance
            position=0,  # Reset position
            position_state=0  # 重置持倉狀態
        )
        return self.state
    
    def get_observation(self):
        if self.current_step + self.window_size < len(self.tradeData):
            return self.observation
            # return self.tradeData[self.current_step]
        else:
            raise Exception("Current step exceeds trade data length. Please reset the environment.")
    
    def step(self, action, agentType='final', tradeAmount=1000):
        if self.done:
            raise Exception("Environment is done. Please reset the environment.")
        info = {}
        # Update the current step
        print(f"Action taken: {action}, Current step: {self.current_step}")

        # 更新持倉狀態（這是獎勵函數中使用的POS_t）
        self.observation.position_state = action  # 直接將動作作為持倉狀態

        self.current_step += 1
        if action == -1:
            # short action
            self.observation.position -= tradeAmount
            self.observation.balance += (self.observation.price * tradeAmount)
            # self.observation.volume = tradeAmount
        elif action == 0:
            # hold action
            pass
        elif action == 1:
            # long action
            self.observation.position += tradeAmount
            self.observation.balance -= (self.observation.price * tradeAmount)
            # self.observation.volume = tradeAmount

        else:
            raise ValueError("Invalid action. Choose from -1 (short), 0 (hold), or 1 (long).")
        
        reward = self.calculate_reward(agentType, self.tradeData.iloc[:, 0].values, self.current_step)
        # reward = self.calculate_reward(agentType, self.tradeData[:, 2], self.current_step)
        self.history.append((self.current_step, action, reward))
        # Update observation
        if self.current_step + self.window_size < len(self.tradeData):
            self.state = self.tradeData.iloc[self.current_step:self.current_step + self.window_size].values
            # self.state = self.tradeData[self.current_step:self.current_step + self.window_size]
            last_row = self.state[-1]
            self.observation.price = last_row[0]  # First column (assuming it's price)
            self.observation.volume = last_row[-1]  # Second column (assuming it's volume)
        else:
            self.done = True
            info = {
                "portfolio_value": self.observation.balance + (self.observation.position * self.observation.price),
                "average_cost": self.observation.balance / (self.observation.position if self.observation.position != 0 else 1),
                "holdings": self.observation.position,
                "position_state": self.observation.position_state
            }

        return self.state, reward, self.done, info


    def calculate_reward(self, agentType='final', prices=[], step=0):
        """
        修正的獎勵函數：使用position_state（{-1,0,1}）而不是累積position
        
        Risk Agent Reward_t = POS_t × mean(R_t^n) / std(R_t^n)
        Return Agent Reward_t = POS_t × (p_{t+n} - p_t) / p_t × 100
        Final Agent Reward_t = POS_t × (p_{t+1} - p_t) / p_t × 100
        
        其中 POS_t ∈ {-1, 0, 1}
        """
        if step >= len(prices):
            return 0.0
        
        current_price = prices[step]
        # 使用position_state而不是累積的position
        position_state = self.observation.position_state  # {-1, 0, 1}
        reward = 0.0
        # print(f"DEBUG Reward - Agent: {agentType}, Step: {step}")
        # print(f"  Current price: {current_price:.6f}")
        # print(f"  Position state (POS_t): {position_state}")
        
        if agentType == 'final':
            # Final Agent: POS_t × (p_{t+1} - p_t) / p_t × 100
            if step + 1 < len(prices):
                next_price = prices[step + 1]
                price_return = (next_price - current_price) / current_price
                reward = position_state * price_return * 100
                
                # print(f"  Next price: {next_price:.6f}")
                # print(f"  Price return: {price_return:.6f}")
                # print(f"  Raw reward: {reward:.6f}")
                
                # 由於position_state只有{-1,0,1}，獎勵範圍自然受限
                # 但仍然可以加上安全限制
                # reward = np.clip(reward, -10.0, 10.0)
                # print(f"  Final reward: {reward:.6f}")
            else:
                print("Warning: Not enough future prices to calculate final reward.")
                reward = 0.0
                
        elif agentType == 'risk':
            # Risk Agent: POS_t × mean(R_t^n) / std(R_t^n)
            n = self.window_size  # 未來10天的收益率
            
            return_rates = []
            for i in range(1, n + 1):
                if step + i < len(prices):
                    future_price = prices[step + i - 1]
                    return_rate = (future_price - current_price) / current_price
                    return_rates.append(return_rate)
            
            if len(return_rates) > 1:
                return_rates = np.array(return_rates)
                mean_return = np.mean(return_rates)
                std_return = np.std(return_rates)
                
                if std_return > 1e-8:
                    sharpe_ratio = mean_return / std_return
                    reward = position_state * sharpe_ratio
                else:
                    reward = position_state * mean_return
                    
                # 限制獎勵範圍
                # reward = np.clip(reward, -5.0, 5.0)
            else:
                print("Warning: Not enough return rates to calculate risk reward.")
                reward = 0.0
                
        elif agentType == 'return':
            # Return Agent: POS_t × (p_{t+n} - p_t) / p_t × 100
            n = self.window_size
            
            if step + n < len(prices):
                future_price = prices[step + n - 1]
                price_return = (future_price - current_price) / current_price
                reward = position_state * price_return * 100
                
                # 限制獎勵範圍
                # reward = np.clip(reward, -15.0, 15.0)
            else:
                print("Warning: Not enough future prices to calculate return reward.")
                reward = 0.0
                
        else:
            raise ValueError("Invalid agentType. Choose from 'final', 'risk', or 'return'.")
        
        return reward