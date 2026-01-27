"""Exp5 Adults v6 with multiple seeds.
Adults: temp=0.5, lr=0.001
"""

def grid():
    configs = []
    for seed in [1, 2]:
        configs.append([
            "--n_agents=10",
            "--total_agents=50",
            "--warmup_threshold=0.0",
            "--fresh_start",
            "--kill_epoch=10",
            f"--random_seed={seed}",
            "--enable_plasticity",
            "--lr_min=0.001",
            "--lr_max=0.001",
            "--temp_min=0.5",
            "--temp_max=0.5",
            "--plasticity_max_age=100",
            "--load_from_checkpoint=experiments/exp5/warmup/warmup.tar",
            f"--checkpoint_dir=experiments/exp5/adults_v6/s{seed}",
            "--n_epochs=400",
            "--compute_cross_gen_online",
            "--deterministic",
            "--tensorboard",
            "--snapshot_freq=10",
        ])
    return configs
