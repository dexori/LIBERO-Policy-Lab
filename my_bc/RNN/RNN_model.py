import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialSoftmax(nn.Module):
    """Convert each feature map into its expected (x, y) coordinate."""

    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, feature_map):
        batch_size, channels, height, width = feature_map.shape

        # Coordinate grids are created on the same device / dtype as the input.
        pos_x = torch.linspace(
            -1.0, 1.0, width,
            device=feature_map.device,
            dtype=feature_map.dtype,
        )
        pos_y = torch.linspace(
            -1.0, 1.0, height,
            device=feature_map.device,
            dtype=feature_map.dtype,
        )
        grid_y, grid_x = torch.meshgrid(pos_y, pos_x, indexing="ij")

        attention = F.softmax(
            feature_map.reshape(batch_size, channels, height * width)
            / self.temperature,
            dim=-1,
        )

        expected_x = torch.sum(
            attention * grid_x.reshape(1, 1, -1),
            dim=-1,
        )
        expected_y = torch.sum(
            attention * grid_y.reshape(1, 1, -1),
            dim=-1,
        )

        # 128 feature maps -> 256 spatial coordinates.
        return torch.cat([expected_x, expected_y], dim=-1)

class BCRNNPolicy(nn.Module):
    def __init__(self,state_dim=15,action_dim=7,hidden_dim=128):
        super().__init__()

        self.vision_agent_encoder=nn.Sequential(
            nn.Conv2d(
                3,32,
                kernel_size=5,
                stride=2,
                padding=2
            ),
            nn.ReLU(),

            nn.Conv2d(
                32,64,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.ReLU(),

            nn.Conv2d(
                64,128,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.ReLU(),

            # Keep the feature locations instead of averaging them away.
            SpatialSoftmax(),
        )

        self.vision_wrist_encoder=nn.Sequential(
                nn.Conv2d(
                    3,32,
                    kernel_size=5,
                    stride=2,
                    padding=2
                ),
                nn.ReLU(),
    
                nn.Conv2d(
                    32,64,
                    kernel_size=3,
                    stride=2,
                    padding=1
                ),
                nn.ReLU(),
    
                nn.Conv2d(
                    64,128,
                    kernel_size=3,
                    stride=2,
                    padding=1
                ),
                nn.ReLU(),
    
                # Keep the feature locations instead of averaging them away.
                SpatialSoftmax(),
            )

        self.state_encoder=nn.Sequential(
            # Same proprioception backbone as the state-only BC model.
            nn.Linear(state_dim,128),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.ReLU()
        )

        self.rnn = nn.LSTM(
            input_size=128*2+128*2+128,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        self.policy=nn.Sequential(
            nn.Linear(hidden_dim,128),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128,action_dim),
            nn.Tanh()
        )

    def forward(self,agent_image,wrist_image,state,hidden=None):

        B, T, C, H, W = (agent_image.shape)
         # 把 B 和 T 合并
        agent_image = agent_image.reshape(B * T,C,H,W)
        B, T, C, H, W = (wrist_image.shape)
        # 把 B 和 T 合并
        wrist_image = wrist_image.reshape(B * T,C,H,W)
        state = state.reshape(B * T,-1)

        visual_feature1=self.vision_agent_encoder(agent_image)
        # visual_feature1 = visual_feature1.reshape(B * T,-1)

        visual_feature2=self.vision_wrist_encoder(wrist_image)
        # visual_feature2 = visual_feature2.reshape(B * T,-1)

        state_feature=self.state_encoder(state)

        feature=torch.cat(
            [
               visual_feature1,
               visual_feature2,
               state_feature,
            ],
            dim=1
        )

         # 恢复 sequence
        feature = feature.reshape(B,T,-1)

        out, hidden = self.rnn(feature,hidden)
        

        action=self.policy(out)

        return action,hidden

        
