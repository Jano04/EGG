"""Exp6 Children v1: homogeneous high plasticity (temp=1.6, lr=0.006)"""

def grid():
    base = [
        "--n_agents=10",
        "--total_agents=50",
        "--warmup_threshold=0.0",
        "--fresh_start",
        "--kill_epoch=10",
        "--random_seed=0",
        "--enable_plasticity",
        "--lr_min=0.006",
        "--lr_max=0.006",
        "--temp_min=1.6",
        "--temp_max=1.6",
        "--plasticity_max_age=100",
        "--load_from_checkpoint=experiments/exp5/warmup/warmup.tar",
        "--checkpoint_dir=experiments/exp6/children_v1",
        "--n_epochs=400",
        "--compute_cross_gen_online",
        "--deterministic",
        "--tensorboard",
        "--snapshot_freq=10",
    ]
    return [base]
