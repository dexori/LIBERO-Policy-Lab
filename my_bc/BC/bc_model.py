import torch.nn as nn

class BCPolicy(nn.Module):
    def __init__(self,obs_dim=15,action_dim=7):
        super().__init__()

        self.net=nn.Sequential(
            nn.Linear(obs_dim,128),
            nn.ReLU(),

            nn.Linear(128,128),
            nn.ReLU(),

            nn.Linear(128,action_dim),
            nn.Tanh()
        )

    def forward(self,obs):
        return self.net(obs)


