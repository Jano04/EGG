"""Experiment 3: Age-based plasticity condition.

All agents start at age 100 (full adults). Plasticity varies with age:
- Young: temp=5.0, lr=0.025 (high plasticity)
- Old: temp=0.5, lr=0.001 (low plasticity)

Must be Python because stagger_ages requires a list that JSON/nest cannot express.
"""


def grid():
    configs = []
    for seed in [0, 1, 2]:
        configs.append([
            "--n_agents=10",
            "--total_agents=50",
            "--warmup_threshold=0.0",
            "--fresh_start",
            "--kill_epoch=10",
            f"--random_seed={seed}",
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
            "--load_from_checkpoint=experiments/exp3_plasticity/warmup/final.tar",
            f"--checkpoint_dir=experiments/exp3_plasticity/age_based/s{seed}",
            "--n_epochs=400",
            "--compute_cross_gen_online",
            "--deterministic",
            "--snapshot_freq=10",
        ])
    return configs
