from MADDQN import MADDQN
import config
import config.loadConfig

if __name__ == "__main__":
    # args = parse_args()
    args = config.loadConfig.loding_yaml()
    maddqn = MADDQN(configs=args)
    
    # 使用重新實現的訓練過程，基於論文演算法
    maddqn.training_process()
    maddqn.testing_process()
    print("Model training completed successfully.")