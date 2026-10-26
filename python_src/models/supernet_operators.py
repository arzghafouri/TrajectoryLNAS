import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================================================================
# Utility: Numerical Safety Helpers
# =========================================================================

def safe_cross_attention(mha_layer: nn.MultiheadAttention, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, key_padding_mask: torch.Tensor = None) -> torch.Tensor:
    """
    Executes nn.MultiheadAttention safely, preventing NaN outputs when key_padding_mask
    contains rows where ALL keys are masked (e.g., padded samples with no neighbors or lanes).
    
    When an entire row in the batch is masked, standard PyTorch MultiheadAttention computes
    softmax(-inf) = NaN, polluting backpropagation gradients. This helper guarantees numerical
    stability by unmasking a sentinel key and zeroing out the attention output for fully-masked rows.
    """
    if key_padding_mask is None:
        out, _ = mha_layer(query, key, value)
        return out

    # key_padding_mask: [B, S_key] where True indicates padded/masked
    all_masked = key_padding_mask.all(dim=-1)  # [B]
    if not all_masked.any():
        out, _ = mha_layer(query, key, value, key_padding_mask=key_padding_mask)
        return out

    # If any batch row is fully masked, unmask index 0 temporarily to prevent softmax(-inf) = NaN
    sanitized_mask = key_padding_mask.clone()
    sanitized_mask[all_masked, 0] = False
    out, _ = mha_layer(query, key, value, key_padding_mask=sanitized_mask)

    # Zero out the output for rows that were completely masked
    valid_row_mask = (~all_masked).unsqueeze(-1).unsqueeze(-1).float()
    return out * valid_row_mask


# =========================================================================
# 1. Temporal Operators (Single-Agent Trajectory Dynamics)
# =========================================================================

class OpIdentity(nn.Module):
    """Identity operator that passes input features through directly."""
    def __init__(self, d_model=None):
        super().__init__()
        self.d_model = d_model
        
    def forward(self, x, *args, **kwargs):
        return x
        
    def get_flops(self, seq_len=20, d_model=128, **kwargs):
        return 0

class GRUEncoder(nn.Module):
    """Recurrent GRU encoder for capturing sequential motion history."""
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.gru = nn.GRU(d_model, d_model, batch_first=True)

    def forward(self, x, *args, **kwargs):
        # x: [B, T, D]
        out, _ = self.gru(x)
        return out

    def get_flops(self, seq_len=20, d_model=128, **kwargs):
        # Standard GRU per step: 3 gates (reset, update, new candidate).
        # Linear transformations: 3 * (W_ih + W_hh + biases)
        # MACs = 3 * (d_model * d_model + d_model * d_model + 2 * d_model)
        # FLOPs = 2 * MACs
        return seq_len * 6 * (2 * d_model * d_model + 2 * d_model)

class LSTMEncoder(nn.Module):
    """LSTM encoder with cell states for long-term temporal dependencies."""
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.lstm = nn.LSTM(d_model, d_model, batch_first=True)

    def forward(self, x, *args, **kwargs):
        # x: [B, T, D]
        out, _ = self.lstm(x)
        return out

    def get_flops(self, seq_len=20, d_model=128, **kwargs):
        # Standard LSTM per step: 4 gates (input, forget, cell, output).
        # FLOPs = 2 * 4 * (2 * d_model * d_model + 2 * d_model)
        return seq_len * 8 * (2 * d_model * d_model + 2 * d_model)

class TemporalConvEncoder(nn.Module):
    """Temporal Convolutional Network (TCN) 1D with dilated receptive field."""
    def __init__(self, d_model, kernel=3, dilation=1):
        super().__init__()
        self.d_model = d_model
        padding = (kernel - 1) * dilation // 2
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=kernel, padding=padding, dilation=dilation)
        self.norm = nn.BatchNorm1d(d_model)
        self.act = nn.ReLU()

    def forward(self, x, *args, **kwargs):
        # x: [B, T, D] -> [B, D, T]
        residual = x
        x_t = x.transpose(1, 2)
        out = self.act(self.norm(self.conv(x_t)))
        out = out.transpose(1, 2)
        return out + residual

    def get_flops(self, seq_len=20, d_model=128, **kwargs):
        # Conv1d FLOPs: 2 * seq_len * kernel * in_channels * out_channels
        return 2 * seq_len * 3 * d_model * d_model

class TemporalSelfAttention(nn.Module):
    """Multi-Head Self-Attention over the trajectory temporal dimension."""
    def __init__(self, d_model, heads=4):
        super().__init__()
        self.d_model = d_model
        self.heads = heads
        self.attn = nn.MultiheadAttention(d_model, heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, *args, **kwargs):
        # x: [B, T, D]
        attn_out, _ = self.attn(x, x, x)
        return self.norm(x + attn_out)

    def get_flops(self, seq_len=20, d_model=128, **kwargs):
        # QKV Projections: 3 * (2 * seq_len * d_model^2)
        # QK^T + Softmax * V: 4 * seq_len^2 * d_model
        # Out Projection: 2 * seq_len * d_model^2
        return 8 * seq_len * d_model * d_model + 4 * seq_len * seq_len * d_model


# =========================================================================
# 2. Multi-Agent Social Interaction Operators (Agent-to-Agent Domain)
# =========================================================================

class GCNLayer(nn.Module):
    """
    Graph Convolutional Network modeling spatial adjacency between ego vehicle
    and surrounding agents based on relative distance at current observation step.
    Includes numerical degree normalization and robust logical masking.
    """
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.fc = nn.Linear(d_model, d_model)
        self.act = nn.ReLU()
        self.norm = nn.LayerNorm(d_model)

    def forward(self, target_feat, neighbor_feats=None, agent_mask=None, agent_pos=None, *args, **kwargs):
        # target_feat: [B, T, D]
        # neighbor_feats: [B, N, D] where index 0 is ego, 1..N-1 are neighbors
        if neighbor_feats is None:
            return target_feat

        B, T, D = target_feat.shape
        N = neighbor_feats.shape[1]

        # Construct spatial adjacency from agent coordinates or learned similarity
        if agent_pos is not None:
            # agent_pos: [B, N, 2] coordinates at t=T-1
            dist = torch.cdist(agent_pos, agent_pos)  # [B, N, N]
            adj = torch.exp(-dist / 10.0)             # Gaussian spatial decay (10m bandwidth)
        else:
            # Distance-agnostic normalized adjacency with self-loops
            adj = torch.eye(N, device=neighbor_feats.device).unsqueeze(0).expand(B, -1, -1).clone()
            if N > 1:
                norm_feats = F.normalize(neighbor_feats, p=2, dim=-1)
                sim = torch.bmm(norm_feats, norm_feats.transpose(1, 2)).clamp(min=0.0)
                adj = 0.5 * adj + 0.5 * sim

        if agent_mask is not None:
            # agent_mask: [B, N] (True for valid, False for padded)
            # Use logical AND (&) for boolean tensors instead of arithmetic multiplication
            mask_2d = agent_mask.unsqueeze(1) & agent_mask.unsqueeze(2)  # [B, N, N]
            adj = adj * mask_2d.float()

        # Degree normalization (D^-1 A) with numerical protection against zero-degrees
        deg = adj.sum(dim=-1, keepdim=True)
        d_inv = torch.where(deg > 1e-5, 1.0 / deg, torch.zeros_like(deg))
        norm_adj = adj * d_inv

        # GCN propagation: [B, N, N] x [B, N, D] -> [B, N, D]
        gcn_out = self.act(self.fc(torch.bmm(norm_adj, neighbor_feats)))
        ego_updated = gcn_out[:, 0:1, :]  # Ego updated representation [B, 1, D]

        # Broadcast over temporal dimension and apply residual connection
        return self.norm(target_feat + ego_updated.expand(-1, T, -1))

    def get_flops(self, seq_len=20, d_model=128, num_agents=16, **kwargs):
        # Adjacency matmul: 2 * num_agents * num_agents * d_model
        # FC layer: 2 * num_agents * d_model * d_model
        return 2 * num_agents * num_agents * d_model + 2 * num_agents * d_model * d_model

class GATLayer(nn.Module):
    """
    Graph Attention Network modeling dynamic directed attention from ego vehicle to neighbors.
    Ego trajectory query attends over surrounding agent key/value vectors with NaN protection.
    """
    def __init__(self, d_model, heads=4):
        super().__init__()
        self.d_model = d_model
        self.heads = heads
        self.attn = nn.MultiheadAttention(d_model, heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, target_feat, neighbor_feats=None, agent_mask=None, *args, **kwargs):
        # target_feat: [B, T, D]
        # neighbor_feats: [B, N, D]
        if neighbor_feats is None:
            return target_feat

        key_padding_mask = (~agent_mask) if agent_mask is not None else None
        attn_out = safe_cross_attention(self.attn, target_feat, neighbor_feats, neighbor_feats, key_padding_mask)
        return self.norm(target_feat + attn_out)

    def get_flops(self, seq_len=20, d_model=128, num_agents=16, **kwargs):
        # QKV projections + Cross Attention between seq_len and num_agents + Out projection
        return 4 * (seq_len + num_agents) * d_model * d_model + 4 * seq_len * num_agents * d_model

class SocialPooling(nn.Module):
    """
    Social Pooling aggregating neighbor agents' spatial context across surrounding vehicles.
    Pools valid neighbor representations and injects pooled social context into ego history.
    """
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, target_feat, neighbor_feats=None, agent_mask=None, *args, **kwargs):
        # target_feat: [B, T, D]
        # neighbor_feats: [B, N, D]
        if neighbor_feats is None:
            return target_feat

        B, T, D = target_feat.shape
        if agent_mask is not None:
            mask = agent_mask.unsqueeze(-1).float()  # [B, N, 1]
            denom = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
            pooled = (neighbor_feats * mask).sum(dim=1, keepdim=True) / denom
        else:
            pooled = neighbor_feats.mean(dim=1, keepdim=True)  # [B, 1, D]

        social_context = self.fc(pooled)  # [B, 1, D]
        return self.norm(target_feat + social_context.expand(-1, T, -1))

    def get_flops(self, seq_len=20, d_model=128, num_agents=16, **kwargs):
        return num_agents * d_model + 4 * d_model * d_model

class PairwiseAttention(nn.Module):
    """
    Pairwise Spatial-Gated Attention estimating reciprocal influence between ego vehicle and each neighbor.
    Unlike standard MultiheadAttention, this operator explicitly incorporates relative Euclidean
    distance gating between ego and surrounding agents, providing unique inductive bias.
    """
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dist_scale = nn.Parameter(torch.tensor(0.1))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, target_feat, neighbor_feats=None, agent_mask=None, agent_pos=None, *args, **kwargs):
        if neighbor_feats is None:
            return target_feat

        B, T, D = target_feat.shape
        N = neighbor_feats.shape[1]

        Q = self.q_proj(target_feat)  # [B, T, D]
        K = self.k_proj(neighbor_feats)  # [B, N, D]
        V = self.v_proj(neighbor_feats)  # [B, N, D]

        scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(D)  # [B, T, N]

        # Modulate scores with physical distance decay if agent coordinates are provided
        if agent_pos is not None:
            # ego is index 0
            ego_pos = agent_pos[:, 0:1, :]  # [B, 1, 2]
            dist_to_ego = torch.norm(agent_pos - ego_pos, p=2, dim=-1, keepdim=True)  # [B, N, 1]
            spatial_bias = -torch.abs(self.dist_scale) * dist_to_ego.transpose(1, 2)  # [B, 1, N]
            scores = scores + spatial_bias

        if agent_mask is not None:
            all_masked = (~agent_mask).all(dim=-1, keepdim=True)  # [B, 1]
            mask_expanded = agent_mask.unsqueeze(1)  # [B, 1, N]
            if all_masked.any():
                mask_expanded = mask_expanded.clone()
                mask_expanded[all_masked.squeeze(-1), 0, 0] = True

            scores = scores.masked_fill(~mask_expanded, -1e9)
            attn_weights = F.softmax(scores, dim=-1)

            if all_masked.any():
                valid_mask = (~all_masked).unsqueeze(-1).float()
                attn_weights = attn_weights * valid_mask
        else:
            attn_weights = F.softmax(scores, dim=-1)

        context = torch.bmm(attn_weights, V)  # [B, T, D]
        out = self.out_proj(context)
        return self.norm(target_feat + out)

    def get_flops(self, seq_len=20, d_model=128, num_agents=16, **kwargs):
        # QKV Projections + Attention Matrix + Out Projection
        q_proj = 2 * seq_len * d_model * d_model
        kv_proj = 4 * num_agents * d_model * d_model
        attn_scores = 2 * seq_len * num_agents * d_model
        attn_values = 2 * seq_len * num_agents * d_model
        out_proj = 2 * seq_len * d_model * d_model
        return q_proj + kv_proj + attn_scores + attn_values + out_proj


# =========================================================================
# 3. Vectorized HD Map Operators (Map & Polyline Topology Domain)
# =========================================================================

class PolylineEncoder(nn.Module):
    """
    PointNet / VectorNet-style Polyline Subgraph Encoder for HD Map Centerlines.
    Encodes [B, N_lanes, lane_pts, 2 or 4] or [B, N_lanes, 2 or 4] into high-dimensional lane embeddings [B, N_lanes, d_model].
    Includes segment tangent vectors (direction) and optional lane type embedding.
    """
    def __init__(self, in_channels=4, d_model=128):
        super().__init__()
        self.d_model = d_model
        self.in_channels = in_channels
        self.point_mlp = nn.Sequential(
            nn.Linear(in_channels, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU()
        )
        self.lane_type_emb = nn.Embedding(4, d_model)
        self.lane_proj = nn.Linear(d_model, d_model)

    def forward(self, map_lanes, lane_types=None, mask=None, **kwargs):
        """
        map_lanes / polylines: [B, N_lanes, lane_pts, C] or [B, N_lanes, C]
        lane_types: [B, N_lanes] (optional)
        mask: [B, N_lanes] or [B, N_lanes, lane_pts] (optional)
        Returns: [B, N_lanes, d_model]
        """
        # Accept 'polylines' alias if passed via kwargs or first argument
        polylines = map_lanes
        if polylines.dim() == 4:
            B, N, P, C = polylines.shape
            if C == 2 and self.in_channels >= 4:
                tangents = torch.zeros_like(polylines)
                tangents[:, :, 1:, :] = polylines[:, :, 1:, :] - polylines[:, :, :-1, :]
                tangents[:, :, 0, :] = tangents[:, :, 1, :]
                polylines = torch.cat([polylines, tangents], dim=-1)
                C = polylines.shape[-1]

            x = polylines.reshape(B * N * P, C)
            pt_feat = self.point_mlp(x)  # [B*N*P, D]
            pt_feat = pt_feat.view(B, N, P, self.d_model)

            if mask is not None and mask.dim() == 3:
                m = mask.unsqueeze(-1).float()
                denom = m.sum(dim=2, keepdim=True).clamp(min=1.0)
                lane_feat = (pt_feat * m).sum(dim=2) / denom.squeeze(-1)
            else:
                lane_feat, _ = pt_feat.max(dim=2)  # [B, N, D]

            if lane_types is not None:
                lane_feat = lane_feat + self.lane_type_emb(lane_types.clamp(0, 3))
            return self.lane_proj(lane_feat)

        elif polylines.dim() == 3:
            B, N, C = polylines.shape
            x = polylines.reshape(B * N, C)
            lane_feat = self.point_mlp(x).view(B, N, self.d_model)
            if lane_types is not None:
                lane_feat = lane_feat + self.lane_type_emb(lane_types.clamp(0, 3))
            return self.lane_proj(lane_feat)
        else:
            return self.lane_proj(self.point_mlp(polylines))

    def get_flops(self, num_lanes=20, lane_pts=10, d_model=128, **kwargs):
        # point_mlp (4 channels) + maxpool + lane_proj
        return num_lanes * lane_pts * (2 * self.in_channels * (d_model // 2) + 2 * (d_model // 2) * d_model) + num_lanes * 2 * d_model * d_model

class LaneAttention(nn.Module):
    """
    Cross-attention mechanism between vehicle trajectory state and HD Map lane vectors.
    Query = Ego Trajectory, Key/Value = Vectorized HD Map Lanes.
    Protected against NaN when all lanes in a sample are masked/padded.
    """
    def __init__(self, d_model, heads=4):
        super().__init__()
        self.d_model = d_model
        self.heads = heads
        self.attn = nn.MultiheadAttention(d_model, heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, target_feat, lane_feats=None, lane_mask=None, *args, **kwargs):
        # target_feat: [B, T, D]
        # lane_feats: [B, N_lanes, D]
        if lane_feats is None:
            return target_feat

        key_padding_mask = (~lane_mask) if lane_mask is not None else None
        attn_out = safe_cross_attention(self.attn, target_feat, lane_feats, lane_feats, key_padding_mask)
        return self.norm(target_feat + attn_out)

    def get_flops(self, seq_len=20, d_model=128, num_lanes=20, **kwargs):
        q_proj = 2 * seq_len * d_model * d_model
        kv_proj = 4 * num_lanes * d_model * d_model
        attn_matrix = 2 * seq_len * num_lanes * d_model
        attn_value = 2 * seq_len * num_lanes * d_model
        out_proj = 2 * seq_len * d_model * d_model
        return q_proj + kv_proj + attn_matrix + attn_value + out_proj

class LaneGraphEncoder(nn.Module):
    """
    Encodes road graph lane-to-lane topological connectivity using Graph Convolutions,
    then projects fused road graph context onto the ego agent trajectory.
    """
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.gcn = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads=2, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, target_feat, lane_feats=None, lane_adj=None, lane_mask=None, *args, **kwargs):
        if lane_feats is None:
            return target_feat

        B, N, D = lane_feats.shape
        # Compute lane adjacency if not provided (based on distance between lane endpoints)
        if lane_adj is None:
            norm_lanes = F.normalize(lane_feats, p=2, dim=-1)
            lane_adj = torch.bmm(norm_lanes, norm_lanes.transpose(1, 2)).clamp(min=0.0)
            lane_adj = lane_adj + torch.eye(N, device=lane_feats.device).unsqueeze(0)

        if lane_mask is not None:
            mask_2d = lane_mask.unsqueeze(1) & lane_mask.unsqueeze(2)
            lane_adj = lane_adj * mask_2d.float()

        # Degree normalization (D^-1 A) with zero-division safety
        deg = lane_adj.sum(dim=-1, keepdim=True)
        d_inv = torch.where(deg > 1e-5, 1.0 / deg, torch.zeros_like(deg))
        norm_adj = lane_adj * d_inv
        graph_lanes = self.gcn(torch.bmm(norm_adj, lane_feats))  # [B, N_lanes, D]

        # Cross attend from ego trajectory to topological road network safely
        key_padding_mask = (~lane_mask) if lane_mask is not None else None
        attn_out = safe_cross_attention(self.cross_attn, target_feat, graph_lanes, graph_lanes, key_padding_mask)
        return self.norm(target_feat + attn_out)

    def get_flops(self, seq_len=20, d_model=128, num_lanes=20, **kwargs):
        gcn_flops = 2 * num_lanes * num_lanes * d_model + 4 * num_lanes * d_model * d_model
        attn_flops = 4 * (seq_len + num_lanes) * d_model * d_model + 4 * seq_len * num_lanes * d_model
        return gcn_flops + attn_flops


class MapFeatureFusion(nn.Module):
    """
    Direct spatial pooling and non-linear MLP projection of lane geometry with agent state.
    """
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.fc = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, target_feat, lane_feats=None, lane_mask=None, *args, **kwargs):
        if lane_feats is None:
            return target_feat

        B, T, D = target_feat.shape
        if lane_mask is not None:
            mask = lane_mask.unsqueeze(-1).float()
            denom = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
            pooled_lane = (lane_feats * mask).sum(dim=1, keepdim=True) / denom
        else:
            pooled_lane = lane_feats.mean(dim=1, keepdim=True)  # [B, 1, D]

        expanded_lane = pooled_lane.expand(-1, T, -1)  # [B, T, D]
        fused = self.fc(torch.cat([target_feat, expanded_lane], dim=-1))
        return self.norm(target_feat + fused)

    def get_flops(self, seq_len=20, d_model=128, num_lanes=20, **kwargs):
        return num_lanes * d_model + 2 * seq_len * (2 * d_model * d_model + d_model * d_model)


# =========================================================================
# 4. Continuous Mixed Operations for DARTS
# =========================================================================

class TemporalMixedOp(nn.Module):
    """Continuous relaxation of Temporal branch operators."""
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.ops = nn.ModuleList([
            OpIdentity(d_model),
            GRUEncoder(d_model),
            LSTMEncoder(d_model),
            TemporalConvEncoder(d_model),
            TemporalSelfAttention(d_model),
        ])
        self.op_names = [
            "Identity",
            "GRUEncoder",
            "LSTMEncoder",
            "TemporalConvEncoder",
            "TemporalSelfAttention"
        ]

    def forward(self, x, alpha_temp):
        return sum(w * op(x) for w, op in zip(alpha_temp, self.ops))

    def get_ops_flops(self, seq_len=20):
        return [op.get_flops(seq_len=seq_len, d_model=self.d_model) for op in self.ops]

class InteractionMixedOp(nn.Module):
    """Continuous relaxation of Multi-agent Interaction branch operators."""
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.ops = nn.ModuleList([
            OpIdentity(d_model),
            GCNLayer(d_model),
            GATLayer(d_model),
            SocialPooling(d_model),
            PairwiseAttention(d_model),
        ])
        self.op_names = [
            "Identity",
            "GCNLayer",
            "GATLayer",
            "SocialPooling",
            "PairwiseAttention"
        ]

    def forward(self, target_feat, neighbor_feats, alpha_inter, agent_mask=None, agent_pos=None):
        return sum(
            w * op(target_feat, neighbor_feats=neighbor_feats, agent_mask=agent_mask, agent_pos=agent_pos)
            for w, op in zip(alpha_inter, self.ops)
        )

    def get_ops_flops(self, seq_len=20, num_agents=16):
        return [op.get_flops(seq_len=seq_len, d_model=self.d_model, num_agents=num_agents) for op in self.ops]

class MapMixedOp(nn.Module):
    """Continuous relaxation of HD Map / Road topology branch operators."""
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.ops = nn.ModuleList([
            OpIdentity(d_model),
            LaneAttention(d_model),
            LaneGraphEncoder(d_model),
            MapFeatureFusion(d_model),
        ])
        self.op_names = [
            "Identity",
            "LaneAttention",
            "LaneGraphEncoder",
            "MapFeatureFusion"
        ]

    def forward(self, target_feat, lane_feats, alpha_map, lane_adj=None, lane_mask=None):
        return sum(
            w * op(target_feat, lane_feats=lane_feats, lane_adj=lane_adj, lane_mask=lane_mask)
            for w, op in zip(alpha_map, self.ops)
        )

    def get_ops_flops(self, seq_len=20, num_lanes=20):
        return [op.get_flops(seq_len=seq_len, d_model=self.d_model, num_lanes=num_lanes) for op in self.ops]
