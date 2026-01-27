"""Exp5 Children v4: all agents have uniform high plasticity
Children: temp=3.5, lr=0.015
Only temp increased from v3 (2.5 -> 3.5), LR unchanged.
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
        "--lr_min=0.015",
        "--lr_max=0.015",
        "--temp_min=3.5",
        "--temp_max=3.5",
        "--plasticity_max_age=100",
        "--load_from_checkpoint=experiments/exp5/warmup/warmup.tar",
        "--checkpoint_dir=experiments/exp5/children_v4",
        "--n_epochs=400",
        "--compute_cross_gen_online",
        "--deterministic",
        "--tensorboard",
        "--snapshot_freq=10",
    ]
    return [base]
