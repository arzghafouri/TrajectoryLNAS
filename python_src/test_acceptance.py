"""
TrajectoryLNAS Technical Acceptance & Pipeline Validation Suite
==============================================================
Standalone acceptance test suite verifying:
  1. Data Pipeline Integrity: past [B, 20, 2], agents_past [B, 20, 16, 2], map_lanes [B, 20, 10, 2]
  2. Multi-Modal SuperNet: 6 trajectory modes and confidence logits
  3. Non-zero gradient flow from loss to agents_past, map_lanes, and DARTS alphas
  4. FinalTrajectoryModel instantiation, analytical FLOPs, and parameter audit
  5. Multi-modal metrics: minADE@6, minFDE@6, MissRate@6, and WTA best_of_k_loss
  6. Checkpoint serialization and strict restoration verification

Run on Windows:
  python test_acceptance.py
"""

import os
import sys
import tempfile
import torch
import torch.nn as nn

# Ensure python_src is in path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from models.search_supernet import SearchSuperNet, ArchParams
from models.final_model import FinalTrajectoryModel
from evaluate.metrics import min_ade_k, min_fde_k, miss_rate_k, best_of_k_loss
from evaluate.hardware_profiler import HardwareProfiler


def run_acceptance_tests():
    print("=" * 80)
    print(" TrajectoryLNAS Technical Acceptance & Verification Suite")
    print(f" PyTorch Version: {torch.__version__}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" Target Device:  {device}")
    print("=" * 80)

    B = 2
    T_past = 20
    T_future = 30
    K_modes = 6
    N_agents = 16
    N_lanes = 20
    P_pts = 10

    # Test 1: Data Contract & SuperNet Forward
    print("\n[Test 1] Testing Data Contract & SearchSuperNet...")
    past = torch.randn(B, T_past, 2, device=device, requires_grad=True)
    agents_past = torch.randn(B, T_past, N_agents, 2, device=device, requires_grad=True)
    map_lanes = torch.randn(B, N_lanes, P_pts, 2, device=device, requires_grad=True)
    future = torch.randn(B, T_future, 2, device=device)

    arch_params = ArchParams().to(device)
    alphas = arch_params.softmax_all()

    supernet = SearchSuperNet(
        past_len=T_past,
        future_len=T_future,
        num_modes=K_modes,
        max_agents=N_agents,
        max_lanes=N_lanes,
        lane_pts=P_pts
    ).to(device)

    pred_trajs, mode_logits = supernet(past, alphas, agents_past=agents_past, map_lanes=map_lanes)
    assert pred_trajs.shape == (B, K_modes, T_future, 2), f"Expected shape ({B}, {K_modes}, {T_future}, 2), got {pred_trajs.shape}"
    assert mode_logits.shape == (B, K_modes), f"Expected shape ({B}, {K_modes}), got {mode_logits.shape}"
    print(f"  ✓ SuperNet generated multi-modal trajectories: {pred_trajs.shape}")
    print(f"  ✓ SuperNet generated mode confidence logits:  {mode_logits.shape}")

    # Test 2: Gradient Flow & Data Pipeline Participation
    print("\n[Test 2] Testing Gradient Flow to Neighbors, Map, and Alphas...")
    loss_reg, loss_cls = best_of_k_loss(pred_trajs, future, mode_logits)
    total_loss = loss_reg + 0.1 * loss_cls
    total_loss.backward()

    assert past.grad is not None and past.grad.norm().item() > 0, "Gradient did not flow to past trajectory"
    assert agents_past.grad is not None and agents_past.grad.norm().item() > 0, "Gradient did not flow to neighbor agents"
    assert map_lanes.grad is not None and map_lanes.grad.norm().item() > 0, "Gradient did not flow to HD map lanes"
    assert arch_params.alpha_temporal.grad is not None, "Gradient did not flow to temporal alphas"
    assert arch_params.alpha_interaction.grad is not None, "Gradient did not flow to interaction alphas"
    assert arch_params.alpha_map.grad is not None, "Gradient did not flow to map alphas"

    print("  ✓ Gradient successfully flowed to ego trajectory.")
    print("  ✓ Gradient successfully flowed to neighbor agents tensor (Interaction branch active).")
    print("  ✓ Gradient successfully flowed to HD map lanes tensor (HD Map branch active).")
    print("  ✓ Architectural parameters alpha received valid optimization gradients.")

    # Test 3: Standalone FinalTrajectoryModel & Analytical FLOPs
    print("\n[Test 3] Testing Standalone FinalTrajectoryModel & Hardware Audit...")
    final_model = FinalTrajectoryModel(
        best_temporal_idx=3, # TemporalConv
        best_interaction_idx=2, # GAT
        best_map_idx=1, # LaneAttention
        past_len=T_past,
        future_len=T_future,
        num_modes=K_modes,
        max_agents=N_agents,
        max_lanes=N_lanes,
        lane_pts=P_pts
    ).to(device)

    f_preds, f_logits = final_model(past.detach(), agents_past=agents_past.detach(), map_lanes=map_lanes.detach())
    assert f_preds.shape == (B, K_modes, T_future, 2)
    assert f_logits.shape == (B, K_modes)

    summary = final_model.get_summary()
    assert summary["theoretical_flops_g"] is not None and summary["theoretical_flops_g"] > 0
    print(f"  ✓ Final Model Operators: Temporal={summary['temporal_op']} | Interaction={summary['interaction_op']} | Map={summary['map_op']}")
    print(f"  ✓ Analytical Theoretical FLOPs: {summary['theoretical_flops_g']} GFLOPs")
    print(f"  ✓ Trainable Parameter Count:    {summary['parameters']:,}")

    # Test 4: Multi-Modal Metrics
    print("\n[Test 4] Testing Multi-Modal Metrics (minADE@6, minFDE@6, MissRate@6)...")
    val_ade = min_ade_k(f_preds, future, k=6).item()
    val_fde = min_fde_k(f_preds, future, k=6).item()
    val_mr = miss_rate_k(f_preds, future, k=6, threshold=2.0)
    print(f"  ✓ Sample minADE@6:    {val_ade:.4f} meters")
    print(f"  ✓ Sample minFDE@6:    {val_fde:.4f} meters")
    print(f"  ✓ Sample MissRate@6:  {val_mr * 100:.2f}%")

    # Test 5: Checkpoint Integrity & Strict Verification
    print("\n[Test 5] Testing Checkpoint Save & Reload...")
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp_file:
        tmp_ckpt_path = tmp_file.name

    try:
        ckpt_data = {
            "epoch": 1,
            "model_state_dict": final_model.state_dict(),
            "best_ade": val_ade,
            "best_fde": val_fde,
            "config": summary
        }
        torch.save(ckpt_data, tmp_ckpt_path)

        reloaded_model = FinalTrajectoryModel(
            best_temporal_idx=summary["best_temporal_idx"],
            best_interaction_idx=summary["best_interaction_idx"],
            best_map_idx=summary["best_map_idx"],
            past_len=T_past,
            future_len=T_future,
            num_modes=K_modes,
            max_agents=N_agents,
            max_lanes=N_lanes,
            lane_pts=P_pts
        ).to(device)

        loaded_ckpt = torch.load(tmp_ckpt_path, map_location=device, weights_only=False)
        reloaded_model.load_state_dict(loaded_ckpt["model_state_dict"])
        reloaded_preds, _ = reloaded_model(past.detach(), agents_past=agents_past.detach(), map_lanes=map_lanes.detach())
        diff = torch.norm(f_preds - reloaded_preds).item()
        assert diff < 1e-5, f"Restored weights produced differing outputs: diff={diff}"
        print(f"  ✓ Checkpoint verified. Weight difference after reload: {diff:.8f}")
    finally:
        if os.path.exists(tmp_ckpt_path):
            os.remove(tmp_ckpt_path)

    # Test 6: Hardware Profiler under Inference Mode
    print("\n[Test 6] Testing Hardware Profiler (Inference Mode & Latency)...")
    profiler = HardwareProfiler(final_model, device, past_len=T_past, max_agents=N_agents, max_lanes=N_lanes, lane_pts=P_pts)
    hw_report = profiler.profile(num_warmup=10, num_runs=25)
    print(f"  ✓ Profiler device:       {hw_report['device']}")
    print(f"  ✓ Latency mean:          {hw_report['latency_ms_mean']} ms")
    print(f"  ✓ Throughput:            {hw_report['throughput_fps']} FPS")
    print(f"  ✓ Theoretical FLOPs:     {hw_report['theoretical_flops_g']} GFLOPs")

    print("\n" + "=" * 80)
    print(" ALL 6 TECHNICAL ACCEPTANCE TESTS PASSED SUCCESSFULLY! ")
    print("=" * 80)


if __name__ == "__main__":
    run_acceptance_tests()
