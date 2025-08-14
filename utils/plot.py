import matplotlib.pyplot as plt
import numpy as np

class PlotUtils:
    
    def __init__(self, configs):
        self.configs = configs
    
    def plot_reward_distribution(self, reward_stats):
        """
        繪製每個代理的獎勵分佈圖
        """
        
        plt.figure(figsize=(12, 6))
        for agent, stats in reward_stats.items():
            rewards = stats['rewards']
            if rewards:
                plt.hist(rewards, bins=50, alpha=0.5, label=f"{agent.capitalize()} Agent")
        
        plt.title("Reward Distribution of Agents")
        plt.xlabel("Reward")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid()
        plt.savefig("reward_distribution.png")

    def plot_episode_returns(self, episode_returns):
        """
        繪製每個episode的回報率趨勢圖
        """
        episodes = list(range(1, self.configs['training']['episodes'] + 1))
        risk_returns = episode_returns['risk_agent']
        return_returns = episode_returns['return_agent'] 
        final_returns = episode_returns['final_agent']

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

    def plot_loss_trends(self, reward_stats):
        """
        繪製每個代理的損失趨勢圖
        """
        episodes = list(range(1, self.configs['training']['episodes'] + 1))

        risk_losses = [np.mean(reward_stats['risk_agent']['rewards'][i::self.configs['training']['episodes']]) for i in range(self.configs['training']['episodes'])]
        return_losses = [np.mean(reward_stats['return_agent']['rewards'][i::self.configs['training']['episodes']]) for i in range(self.configs['training']['episodes'])]
        final_losses = [np.mean(reward_stats['final_agent']['rewards'][i::self.configs['training']['episodes']]) for i in range(self.configs['training']['episodes'])]

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
    
    def plot_stock_history(self, unprocessed_data):
        """
        繪製股票歷史價格圖
        """
        
        try:
            # 檢查數據結構並正確獲取日期和價格
            dates = unprocessed_data.index  # 已確認index是DatetimeIndex
            
            # 處理MultiIndex列結構：('Close', 'DJI')
            if ('Close', 'DJI') in unprocessed_data.columns:
                prices = unprocessed_data[('Close', 'DJI')]
            elif 'Close' in [col[0] if isinstance(col, tuple) else col for col in unprocessed_data.columns]:
                # 找到Close相關的列
                close_col = None
                for col in unprocessed_data.columns:
                    if isinstance(col, tuple) and col[0] == 'Close':
                        close_col = col
                        break
                    elif col == 'Close':
                        close_col = col
                        break
                if close_col:
                    prices = unprocessed_data[close_col]
                    print(f"Using column '{close_col}' as close price")
                else:
                    print("Error: No close price column found")
                    return
            else:
                print("Error: No close price column found")
                print(f"Available columns: {list(unprocessed_data.columns)}")
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
            print(f"Data shape: {unprocessed_data.shape}")
            print(f"Columns: {list(unprocessed_data.columns)}")
            print(f"Index name: {unprocessed_data.index.name}")
            print(f"Price range: ${start_price:.2f} - ${end_price:.2f}")
            print(f"Total return: {total_return:.2f}%")
            
        except Exception as e:
            print(f"Error plotting stock history: {e}")
            print(f"Data columns: {list(unprocessed_data.columns)}")
            print(f"Data shape: {unprocessed_data.shape}")
            print(f"Index: {unprocessed_data.index}")
            import traceback
            traceback.print_exc()

    def plot_test_results(self, test_cumulative_returns, market_return, test_portfolio_values_final, test_episode_rewards):
        """
        繪製測試結果圖表
        """
        try:
            # 檢查必需的屬性是否存在
            if not test_cumulative_returns:
                print("⚠️ 測試結果數據不存在，無法繪圖")
                return
            
            # 創建圖表
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('MADDQN Testing Results Analysis', fontsize=16, fontweight='bold')
            
            # 1. 累積回報率比較
            agents = list(test_cumulative_returns.keys())
            returns = list(test_cumulative_returns.values())
            returns.append(market_return)  # 添加市場回報
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
            portfolio_values = list(test_portfolio_values_final.values())
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
            alphas = [test_cumulative_returns[agent] - market_return
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
            total_rewards = list(test_episode_rewards.values())
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