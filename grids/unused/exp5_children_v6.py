"""Exp5 Children v6: all agents have uniform EXTREME high plasticity
Children: temp=5.0, lr=0.025
Both temp and LR increased from v4 (temp: 3.5->5.0, lr: 0.015->0.025)
Extremity product: 0.125 (vs v4's 0.0525) - more than doubled.
Goal: Push children into more chaos to widen gap with age-based.
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
        "--lr_min=0.025",
        "--lr_max=0.025",
        "--temp_min=5.0",
        "--temp_max=5.0",
        "--plasticity_max_age=100",
        "--load_from_checkpoint=experiments/exp5/warmup/warmup.tar",
        "--checkpoint_dir=experiments/exp5/children_v6",
        "--n_epochs=400",
        "--compute_cross_gen_online",
        "--deterministic",
        "--tensorboard",
        "--snapshot_freq=10",
    ]
    return [base]
