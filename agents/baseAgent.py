# agents/base_agent.py
import torch
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
from utils.replayBuffer import replayBuffer

class BaseAgent:
    def __init__(self, configs, model, target_model, action_dim=3, lr=1e-3, device='cpu', done=False ):
        self.policy_network = model.to(device)
        self.target_network = target_model.to(device)
        self.action_dim = action_dim
        self.gamma = configs['agent']['gamma']
        self.optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=lr)
        self.epsilon = configs['agent']['epsilon_start']
        self.epsilon_max = configs['agent']['epsilon_start']
        self.epsilon_min = configs['agent']['epsilon_end']
        self.epsilon_decay = configs['agent']['epsilon_decay']
        self.device = configs['gpu']['use_gpu'] and torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.replay_buffer = replayBuffer(max_size=configs['training']['replay_memory_size'])
        self.done = done
        self.episodes = configs['training']['episodes']

    def act(self, state):
        pass
        

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def soft_update_target(self, tau=0.01):
        for target_param, param in zip(self.target_network.parameters(), self.policy_network.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def learn(self, batch):
        raise NotImplementedError  # 子類別覆寫
