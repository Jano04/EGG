"""Exp5 Age-based stagger adults v6: everyone starts at age 100 (full adults)
Adults: temp=0.5, lr=0.001 | Children: temp=5.0, lr=0.025
Both extremes widened from v4:
  - Adults more stable: temp 0.6->0.5, lr 0.002->0.001
  - Children more chaotic: temp 3.5->5.0, lr 0.015->0.025
All agents start as fully mature adults with lowest plasticity.
Goal: Maximize gap with all-children condition by having stable anchors
and newborns that can rapidly learn from them.
"""

def grid():
    base = [
        "--n_agents=10",
        "--total_agents=50",
        "--warmup_threshold=0.0",
        "--fresh_start",
        "--kill_epoch=10",
        "--random_seed=0",
        "--enable_plasticity",
        "--lr_min=0.001",
        "--lr_max=0.025",
        "--temp_min=0.5",
        "--temp_max=5.0",
        "--plasticity_max_age=100",
        "--plasticity_function=sigmoid",
        "--plasticity_steepness=10.0",
        "--plasticity_critical_point=0.35",
        "--stagger_ages", "100", "100", "100", "100", "100", "100", "100", "100", "100", "100",
        "--load_from_checkpoint=experiments/exp5/warmup/warmup.tar",
        "--checkpoint_dir=experiments/exp5/age_based_stagger_adults_v6",
        "--n_epochs=400",
        "--compute_cross_gen_online",
        "--deterministic",
        "--tensorboard",
        "--snapshot_freq=10",
    ]
    return [base]
