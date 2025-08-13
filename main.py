from MADDQN import MADDQN
import config
import config.loadConfig
import argparse


def parse_arguement():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="MADDQN Training and Testing")
    
    parser.add_argument('--mode', type=str, default='single', choices=['single', 'multi'], help="Mode of operation: single or multi tickers")
    parser.add_argument('--attention', type=bool, default=False, help="Use attention mechanism in the final agent")
    parser.add_argument('--reward_shaping', type=bool, default=False, help="Use reward shaping in each agent")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguement()
    # args = parse_args()
    configs = config.loadConfig.loding_yaml()
    maddqn = MADDQN(configs=configs, args=args)

    # 使用重新實現的訓練過程，基於論文演算法
    maddqn.training_process()
    maddqn.testing_process()
    print("Model training completed successfully.")