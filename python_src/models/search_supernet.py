import torch
import torch.nn as nn
from models.supernet_operators import (
    TemporalMixedOp,
    InteractionMixedOp,
    MapMixedOp,
    PolylineEncoder
)

class ArchParams(nn.Module):
    """
    Architectural parameters (alpha) controlling continuous operator weighting in DARTS.
    Maintains separate parameter vectors for Temporal, Interaction, and Map branches.
    """
    def __init__(self, num_temporal: int = 5, num_interaction: int = 5, num_map: int = 4):
        super().__init__()
        self.alpha_temporal = nn.Parameter(1e-3 * torch.randn(num_temporal))
        self.alpha_interaction = nn.Parameter(1e-3 * torch.randn(num_interaction))
        self.alpha_map = nn.Parameter(1e-3 * torch.randn(num_map))

    def softmax_all(self, temperature: float = 1.0) -> tuple:
        """Applies temperature-scaled softmax to produce probability distribution over operators."""
        t = max(1e-4, temperature)
        return (
            torch.softmax(self.alpha_temporal / t, dim=-1),
            torch.softmax(self.alpha_interaction / t, dim=-1),
            torch.softmax(self.alpha_map / t, dim=-1),
        )

class SearchSuperNet(nn.Module):
    """
    TrajectoryLNAS Bi-Level Differentiable SuperNet.
    
    Contains continuous operator mixtures across 3 distinct domain branches:
      1. Temporal Branch: Single-agent sequential kinematics (Identity, GRU, LSTM, TCN, Self-Attention)
      2. Interaction Branch: Multi-agent social interactions (Identity, GCN, GAT, SocialPooling, PairwiseAttention)
      3. HD Map Branch: Vectorized road geometry (Identity, LaneAttention, LaneGraphEncoder, MapFeatureFusion)
      
    Multi-Modal Trajectory Prediction:
      Predicts K candidate future trajectories (default K=6) along with confidence probabilities
      to benchmark directly against state-of-the-art models on nuScenes / Argoverse.
    """
    def __init__(
        self,
        d_model: int = 128,
        past_len: int = 20,
        future_len: int = 30,
        num_modes: int = 6,
        max_agents: int = 16,
        max_lanes: int = 20,
        lane_pts: int = 10
    ):
        super().__init__()
        self.d_model = d_model
        self.past_len = past_len
        self.future_len = future_len
        self.num_modes = num_modes
        self.max_agents = max_agents
        self.max_lanes = max_lanes
        self.lane_pts = lane_pts

        # Coordinate embedding layers
        self.input_proj = nn.Linear(2, d_model)
        self.neighbor_proj = nn.Linear(2, d_model)
        self.map_encoder = PolylineEncoder(in_channels=2, d_model=d_model)

        # 3 Dedicated Domain-Aware Search Branches
        self.temporal_branch = TemporalMixedOp(d_model)
        self.interaction_branch = InteractionMixedOp(d_model)
        self.map_branch = MapMixedOp(d_model)

        # Tri-Branch fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU()
        )

        # Multi-Modal Future Trajectory Decoder
        self.decoder = nn.GRU(d_model, d_model, batch_first=True)
        # Outputs K displacements for each future timestep: [B, future_len, K * 2]
        self.out_proj = nn.Linear(d_model, num_modes * 2)
        # Confidence classification head over the K modes
        self.prob_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, num_modes)
        )

    def forward(
        self,
        past: torch.Tensor,
        alphas_tuple: tuple,
        agents_past: torch.Tensor = None,
        map_lanes: torch.Tensor = None
    ):
        """
        Continuous forward pass weighting each operator by its architectural alpha.
        past: [B, T_past, 2] or [B, T_past, Agents, 2]
        alphas_tuple: (alpha_temp, alpha_inter, alpha_map)
        agents_past: [B, T_past, N_agents, 2] (optional surrounding agents history)
        map_lanes: [B, N_lanes, lane_pts, 2] (optional vectorized lane polylines)
        
        Returns:
            pred_trajectories: [B, K, future_len, 2]
            mode_probs: [B, K] (logits)
        """
        # If past is 4D [B, T, N, 2], split target ego and agents
        if past.dim() == 4:
            if agents_past is None:
                agents_past = past
            past = past[:, :, 0, :]

        B, T_past, _ = past.shape

        # 1. Target ego coordinate embedding
        target_feat = self.input_proj(past)  # [B, T_past, d_model]

        # 2. Surrounding agents processing
        neighbor_feats = None
        agent_mask = None
        agent_pos = None

        if agents_past is not None and agents_past.dim() == 4:
            # agents_past: [B, T_past, N, 2]
            N = agents_past.shape[2]
            # Detect valid agents (non-zero padded)
            agent_mask = (agents_past.abs().sum(dim=(1, 3)) > 1e-4) # [B, N]
            agent_mask[:, 0] = True # Target ego is always valid

            # Embed neighbor coordinates
            emb_neighbors = self.neighbor_proj(agents_past) # [B, T, N, d_model]
            # Observation timestamp features + temporal pooling
            neighbor_feats = emb_neighbors.mean(dim=1) # [B, N, d_model]
            agent_pos = agents_past[:, -1, :, :] # [B, N, 2] positions at observation t=19
        else:
            # Fallback for standalone / unit tests
            neighbor_feats = target_feat.mean(dim=1, keepdim=True) # [B, 1, d_model]
            agent_mask = torch.ones(B, 1, dtype=torch.bool, device=past.device)

        # 3. Vectorized HD Map processing
        lane_feats = None
        lane_mask = None

        if map_lanes is not None and map_lanes.dim() == 4:
            # map_lanes: [B, N_lanes, lane_pts, 2]
            lane_mask = (map_lanes.abs().sum(dim=(2, 3)) > 1e-4) # [B, N_lanes]
            lane_feats = self.map_encoder(map_lanes) # [B, N_lanes, d_model]
        else:
            # Fallback for standalone / unit tests
            lane_feats = target_feat.mean(dim=1, keepdim=True) # [B, 1, d_model]
            lane_mask = torch.ones(B, 1, dtype=torch.bool, device=past.device)

        alpha_temp, alpha_inter, alpha_map = alphas_tuple

        # 4. Evaluate three branches concurrently
        feat_temp = self.temporal_branch(target_feat, alpha_temp)
        feat_inter = self.interaction_branch(
            target_feat, neighbor_feats, alpha_inter, agent_mask=agent_mask, agent_pos=agent_pos
        )
        feat_map = self.map_branch(
            target_feat, lane_feats, alpha_map, lane_mask=lane_mask
        )

        # 5. Tri-Branch Fusion
        fused = torch.cat([feat_temp, feat_inter, feat_map], dim=-1)  # [B, T_past, 3 * d_model]
        x_fused = self.fusion(fused)  # [B, T_past, d_model]

        # 6. Multi-Modal Autoregressive Decoding
        # Replicate final temporal state over future prediction horizon
        dec_in = x_fused[:, -1:, :].repeat(1, self.future_len, 1)  # [B, future_len, d_model]
        dec_out, _ = self.decoder(dec_in)  # [B, future_len, d_model]

        # Multi-step displacement prediction for K modes
        deltas = self.out_proj(dec_out)  # [B, future_len, K * 2]
        deltas = deltas.view(B, self.future_len, self.num_modes, 2).permute(0, 2, 1, 3)  # [B, K, future_len, 2]
        pred_trajs = torch.cumsum(deltas, dim=2)  # [B, K, future_len, 2] cumulative displacement

        # Mode classification logits from ego trajectory context
        mode_logits = self.prob_head(x_fused[:, -1, :])  # [B, K]

        return pred_trajs, mode_logits

    def get_ops_flops(self):
        """Calculates theoretical FLOPs for all candidate operators across branches."""
        return {
            "temporal": self.temporal_branch.get_ops_flops(self.past_len),
            "interaction": self.interaction_branch.get_ops_flops(self.past_len, num_agents=self.max_agents),
            "map": self.map_branch.get_ops_flops(self.past_len, num_lanes=self.max_lanes),
            "base": (
                2 * self.past_len * 2 * self.d_model  # input_proj
                + 2 * self.past_len * self.max_agents * 2 * self.d_model # neighbor_proj
                + self.map_encoder.get_flops(self.max_lanes, self.lane_pts, self.d_model) # map_encoder
                + 2 * self.past_len * 3 * self.d_model * self.d_model  # fusion
                + self.future_len * 3 * (2 * self.d_model * self.d_model)  # decoder
                + 2 * self.future_len * self.d_model * (self.num_modes * 2)  # out_proj
                + 2 * self.d_model * self.d_model + 2 * self.d_model * self.num_modes # prob_head
            )
        }
