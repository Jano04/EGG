"""Exp5 Age-based v4: no stagger, everyone starts at age 0
Adults: temp=0.6, lr=0.002 | Children: temp=3.5, lr=0.015
Only temp_max increased from v3 (2.5 -> 3.5), LR unchanged.
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
        "--temp_max=3.5",
        "--plasticity_max_age=100",
        "--plasticity_function=sigmoid",
        "--plasticity_steepness=10.0",
        "--plasticity_critical_point=0.35",
        "--load_from_checkpoint=experiments/exp5/warmup/warmup.tar",
        "--checkpoint_dir=experiments/exp5/age_based_v4",
        "--n_epochs=400",
        "--compute_cross_gen_online",
        "--deterministic",
        "--tensorboard",
        "--snapshot_freq=10",
    ]
    return [base]
