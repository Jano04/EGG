"""Exp6 Age-based stagger v3: auto stagger [0, 10, 20, ..., 90]
Adults: temp=0.6, lr=0.002 | Children: temp=2.5, lr=0.015
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
        "--lr_min=0.002",
        "--lr_max=0.015",
        "--temp_min=0.6",
        "--temp_max=2.5",
        "--plasticity_max_age=100",
        "--plasticity_function=sigmoid",
        "--plasticity_steepness=10.0",
        "--plasticity_critical_point=0.35",
        "--stagger_ages",  # Auto mode: [0, k, 2k, ...] = [0, 10, 20, ..., 90]
        "--load_from_checkpoint=experiments/exp5/warmup/warmup.tar",
        "--checkpoint_dir=experiments/exp6/age_based_stagger_v3",
        "--n_epochs=400",
        "--compute_cross_gen_online",
        "--deterministic",
        "--tensorboard",
        "--snapshot_freq=10",
    ]
    return [base]
