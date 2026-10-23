import torch
import torch.nn as nn
from models.supernet_operators import (
    OpIdentity,
    GRUEncoder,
    LSTMEncoder,
    TemporalConvEncoder,
    TemporalSelfAttention,
    GCNLayer,
    GATLayer,
    SocialPooling,
    PairwiseAttention,
    PolylineEncoder,
    LaneAttention,
    LaneGraphEncoder,
    MapFeatureFusion,
)

class FinalTrajectoryModel(nn.Module):
    """
    Extracted standalone multi-modal trajectory prediction model instantiated from the
    best discrete operators discovered by TrajectoryLNAS.
    
    Predicts K candidate future trajectory modes (default K=6) along with confidence probabilities.
    Fully consumes target ego trajectory, neighbor agent interactions, and vectorized HD map lanes.
    """
    def __init__(
        self,
        best_temporal_idx: int,
        best_interaction_idx: int,
        best_map_idx: int,
        d_model: int = 128,
        past_len: int = 20,
        future_len: int = 30,
        num_modes: int = 6,
        max_agents: int = 16,
        max_lanes: int = 20,
        lane_pts: int = 10
    ):
        super().__init__()
        self.past_len = past_len
        self.future_len = future_len
        self.d_model = d_model
        self.num_modes = num_modes
        self.max_agents = max_agents
        self.max_lanes = max_lanes
        self.lane_pts = lane_pts

        self.best_temporal_idx = int(best_temporal_idx)
        self.best_interaction_idx = int(best_interaction_idx)
        self.best_map_idx = int(best_map_idx)

        # 1. Coordinate Embedding Layers
        self.input_proj = nn.Linear(2, d_model)
        self.neighbor_proj = nn.Linear(2, d_model)
        self.map_encoder = PolylineEncoder(in_channels=2, d_model=d_model)

        # 2. Selected Temporal Operator
        temporal_candidates = [
            lambda: OpIdentity(d_model),
            lambda: GRUEncoder(d_model),
            lambda: LSTMEncoder(d_model),
            lambda: TemporalConvEncoder(d_model),
            lambda: TemporalSelfAttention(d_model),
        ]
        self.temporal_op = temporal_candidates[self.best_temporal_idx]()

        # 3. Selected Interaction Operator
        interaction_candidates = [
            lambda: OpIdentity(d_model),
            lambda: GCNLayer(d_model),
            lambda: GATLayer(d_model),
            lambda: SocialPooling(d_model),
            lambda: PairwiseAttention(d_model),
        ]
        self.interaction_op = interaction_candidates[self.best_interaction_idx]()

        # 4. Selected Map Operator
        map_candidates = [
            lambda: OpIdentity(d_model),
            lambda: LaneAttention(d_model),
            lambda: LaneGraphEncoder(d_model),
            lambda: MapFeatureFusion(d_model),
        ]
        self.map_op = map_candidates[self.best_map_idx]()

        # 5. Tri-Branch Fusion Layer
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU()
        )

        # 6. Multi-Modal Recurrent Decoder & Mode Classification Head
        self.decoder = nn.GRU(d_model, d_model, batch_first=True)
        self.out_proj = nn.Linear(d_model, num_modes * 2)
        self.prob_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, num_modes)
        )

    def forward(
        self,
        past: torch.Tensor,
        agents_past: torch.Tensor = None,
        map_lanes: torch.Tensor = None
    ):
        """
        Forward multi-modal trajectory inference pass.
        past: [B, 20, 2] or [B, 20, N, 2]
        agents_past: [B, 20, N_agents, 2] (optional surrounding agents history)
        map_lanes: [B, N_lanes, lane_pts, 2] (optional vectorized lane polylines)
        
        Returns:
            pred_trajectories: [B, K, 30, 2]
            mode_probs: [B, K] (logits)
        """
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
            N = agents_past.shape[2]
            agent_mask = (agents_past.abs().sum(dim=(1, 3)) > 1e-4) # [B, N]
            agent_mask[:, 0] = True
            emb_neighbors = self.neighbor_proj(agents_past)
            neighbor_feats = emb_neighbors.mean(dim=1) # [B, N, d_model]
            agent_pos = agents_past[:, -1, :, :]
        else:
            neighbor_feats = target_feat.mean(dim=1, keepdim=True)
            agent_mask = torch.ones(B, 1, dtype=torch.bool, device=past.device)

        # 3. Vectorized HD Map processing
        lane_feats = None
        lane_mask = None

        if map_lanes is not None and map_lanes.dim() == 4:
            lane_mask = (map_lanes.abs().sum(dim=(2, 3)) > 1e-4)
            lane_feats = self.map_encoder(map_lanes)
        else:
            lane_feats = target_feat.mean(dim=1, keepdim=True)
            lane_mask = torch.ones(B, 1, dtype=torch.bool, device=past.device)

        # 4. Extract features through discovered optimal operators
        feat_temp = self.temporal_op(target_feat)
        feat_inter = self.interaction_op(
            target_feat, neighbor_feats=neighbor_feats, agent_mask=agent_mask, agent_pos=agent_pos
        )
        feat_map = self.map_op(
            target_feat, lane_feats=lane_feats, lane_mask=lane_mask
        )

        # 5. Multi-modal fusion
        combined = torch.cat([feat_temp, feat_inter, feat_map], dim=-1)
        x_fused = self.fusion(combined)

        # 6. Decode future positions via cumulative displacement (kinematically consistent)
        dec_in = x_fused[:, -1:, :].repeat(1, self.future_len, 1)
        dec_out, _ = self.decoder(dec_in)

        # Predict K candidate displacements
        deltas = self.out_proj(dec_out)  # [B, future_len, K * 2]
        deltas = deltas.view(B, self.future_len, self.num_modes, 2).permute(0, 2, 1, 3)
        pred = torch.cumsum(deltas, dim=2)  # [B, K, future_len, 2]

        # Mode classification confidence logits
        mode_logits = self.prob_head(x_fused[:, -1, :])  # [B, K]

        return pred, mode_logits

    def get_summary(self):
        temporal_names = ["Identity", "GRU", "LSTM", "TemporalConv", "TemporalSelfAttention"]
        interaction_names = ["Identity", "GCN", "GAT", "SocialPooling", "PairwiseAttention"]
        map_names = ["Identity", "LaneAttention", "LaneGraphEncoder", "MapFeatureFusion"]

        t_name = temporal_names[self.best_temporal_idx] if 0 <= self.best_temporal_idx < len(temporal_names) else "Custom"
        i_name = interaction_names[self.best_interaction_idx] if 0 <= self.best_interaction_idx < len(interaction_names) else "Custom"
        m_name = map_names[self.best_map_idx] if 0 <= self.best_map_idx < len(map_names) else "Custom"

        # Theoretical FLOPs calculation for standalone discrete model
        base_flops = (
            2 * self.past_len * 2 * self.d_model  # input_proj
            + 2 * self.past_len * self.max_agents * 2 * self.d_model  # neighbor_proj
            + self.map_encoder.get_flops(self.max_lanes, self.lane_pts, self.d_model)  # map_encoder
            + 2 * self.past_len * 3 * self.d_model * self.d_model  # fusion
            + self.future_len * 3 * (2 * self.d_model * self.d_model)  # decoder
            + 2 * self.future_len * self.d_model * (self.num_modes * 2)  # out_proj
            + 2 * self.d_model * self.d_model + 2 * self.d_model * self.num_modes  # prob_head
        )

        op_flops = (
            self.temporal_op.get_flops(seq_len=self.past_len, d_model=self.d_model)
            + self.interaction_op.get_flops(seq_len=self.past_len, d_model=self.d_model, num_agents=self.max_agents)
            + self.map_op.get_flops(seq_len=self.past_len, d_model=self.d_model, num_lanes=self.max_lanes)
        )
        total_flops_g = round((base_flops + op_flops) / 1e9, 4)

        return {
            "temporal_op": t_name,
            "interaction_op": i_name,
            "map_op": m_name,
            "best_temporal_idx": self.best_temporal_idx,
            "best_interaction_idx": self.best_interaction_idx,
            "best_map_idx": self.best_map_idx,
            "d_model": self.d_model,
            "num_modes": self.num_modes,
            "max_agents": self.max_agents,
            "max_lanes": self.max_lanes,
            "past_len": self.past_len,
            "future_len": self.future_len,
            "theoretical_flops_g": total_flops_g,
            "parameters": sum(p.numel() for p in self.parameters())
        }
