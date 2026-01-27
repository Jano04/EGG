"""Exp6 Age-based v2: no stagger, everyone starts at age 0
Adults: temp=0.4, lr=0.001 | Children: temp=2.0, lr=0.012
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
        "--lr_max=0.012",
        "--temp_min=0.4",
        "--temp_max=2.0",
        "--plasticity_max_age=100",
        "--plasticity_function=sigmoid",
        "--plasticity_steepness=10.0",
        "--plasticity_critical_point=0.35",
        "--load_from_checkpoint=experiments/exp5/warmup/warmup.tar",
        "--checkpoint_dir=experiments/exp6/age_based_v2",
        "--n_epochs=400",
        "--compute_cross_gen_online",
        "--deterministic",
        "--tensorboard",
        "--snapshot_freq=10",
    ]
    return [base]
