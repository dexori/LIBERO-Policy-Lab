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

class VisionEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder=nn.Sequential(
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
            SpatialSoftmax()
        )

    def forward(self,x):
        return self.encoder(x)
    
class ACT(nn.Module):
    def __init__(self,
                state_dim=15,
                action_dim=7,
                chunk_size=10,

                d_model=256,
                nhead=4,

                latent_dim=32,

                style_layers=2,
                memory_layers=2,
                decoder_layers=2,
        ):
        super().__init__()

        self.state_dim=state_dim
        self.action_dim=action_dim
        self.chunk_size=chunk_size
        self.d_model=d_model
        self.latent_dim=latent_dim

        self.agent_encoder=VisionEncoder()

        self.wrist_encoder=VisionEncoder()

        self.agent_projection = (
            nn.Linear(
                256,
                d_model,
            )
        )

        self.wrist_projection = (
            nn.Linear(
                256,
                d_model,
            )
        )

        self.cls_token = (
            nn.Parameter(
                torch.randn(
                    1,
                    1,
                    d_model,
                ) * 0.02
            )
        )
        self.style_state_projection = (
            nn.Linear(
                state_dim,
                d_model,
            )
        )

        self.style_action_projection = (
            nn.Linear(
                action_dim,
                d_model,
            )
        )
        self.style_position = (
            nn.Parameter(
                torch.randn(
                    1,
                    chunk_size + 2,
                    d_model,
                ) * 0.02
            )
        )

  
        style_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=512,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )

        self.style_encoder = nn.TransformerEncoder(
            style_layer,
            num_layers=style_layers,
        )

        self.chunk_size = chunk_size
        self.action_dim = action_dim

        self.latent_head = (
            nn.Linear(
                d_model,
                latent_dim * 2,
            )
        )

        self.policy_state_projection = (
            nn.Linear(
                state_dim,
                d_model,
            )
        )

        self.latent_projection = (
            nn.Linear(
                latent_dim,
                d_model,
            )
        )
        self.obs_position = (
            nn.Parameter(
                torch.randn(
                    1,
                    4,
                    d_model,
                ) * 0.02
            )
        )
        memory_layer = (
            nn.TransformerEncoderLayer(

                d_model=d_model,

                nhead=nhead,

                dim_feedforward=512,

                dropout=0.1,

                activation="gelu",

                batch_first=True,
            )
        )


        self.memory_encoder = (
            nn.TransformerEncoder(
                memory_layer,
                num_layers=memory_layers,
            )
        )
        self.action_queries = (
            nn.Embedding(
                chunk_size,
                d_model,
            )
        )

        decoder_layer = (
            nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=512,
                dropout=0.1,
                activation="gelu",
                batch_first=True,
            )
        )


        self.decoder = (
            nn.TransformerDecoder(
                decoder_layer,
                num_layers=decoder_layers,
            )
        )

        self.action_head = (
            nn.Linear(
                d_model,
                action_dim,
            )
        )

    def encode_style(
        self,
        state,
        action_chunk,
        is_pad,
        sample=True,
    ):
        if is_pad is None:
            raise ValueError(
                "is_pad is required when action_chunk is provided"
            )

        B = state.shape[0]
        cls_token = (
            self.cls_token.expand(
                B,
                -1,
                -1,
            )
        )
        state_token = (
            self
            .style_state_projection(
                state
            )
            .unsqueeze(1)
        )
        action_tokens = (
            self
            .style_action_projection(
                action_chunk
            )
        )
        # [B,K+2,D]

        tokens = torch.cat(
            [
                cls_token,
                state_token,
                action_tokens,
            ],
            dim=1,
        )

        tokens = (
            tokens
            + self.style_position[
                :, :tokens.shape[1]
            ]
        )

        prefix_mask = torch.zeros(

            B,
            2,

            dtype=torch.bool,

            device=state.device,
        )

        full_mask = torch.cat(
            [
                prefix_mask,
                is_pad,
            ],
            dim=1,
        )

        encoded = (
            self.style_encoder(

                tokens,

                src_key_padding_mask=
                    full_mask,
            )
        )


        style_feature = (
            encoded[:, 0]
        )

        latent_stats = (
            self.latent_head(
                style_feature
            )
        )
        mu = latent_stats[
            :, :self.latent_dim
        ]

        logvar = latent_stats[
            :, self.latent_dim:
        ]

        std = torch.exp(
            0.5 * logvar
        )


        if sample:
            epsilon = torch.randn_like(std)
            z = mu+ std * epsilon
        else:
            z = mu
        return (z,mu,logvar,)
        

    def forward(self,agent_image,wrist_image,state,action_chunk=None,is_pad=None,sample_latent=True,):

        B = state.shape[0]

        if action_chunk is not None:

            z, mu, logvar = (
                self.encode_style(
                    state,
                    action_chunk,
                    is_pad,
                    sample=sample_latent,
                )
            )
        else:
            z = torch.zeros(
                B,
                self.latent_dim,
                device=state.device,
                dtype=state.dtype,
            )
            mu = None
            logvar = None

        agent_feature=self.agent_encoder(agent_image)
        wrist_feature=self.wrist_encoder(wrist_image)

        agent_token = self.agent_projection(agent_feature)
        wrist_token = self.wrist_projection(wrist_feature)

        state_token = self.policy_state_projection(state)
        latent_token = self.latent_projection(z)

        memory_tokens = (
            torch.stack(
                [
                    agent_token,
                    wrist_token,
                    state_token,
                    latent_token,
                ],
                dim=1,
            )
        )

        memory_tokens = (
            memory_tokens
            + self.obs_position
        )
        memory=self.memory_encoder(memory_tokens)

        queries = self.action_queries.weight.unsqueeze(0).expand(B,-1,-1)

        decoder_output = (
            self.decoder(
                tgt=queries,
                memory=memory,
            )
        )

        predicted_actions = (
            self.action_head(
                decoder_output
            )
        )

        return predicted_actions,mu,logvar
        
