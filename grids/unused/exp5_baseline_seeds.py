"""Exp5 Baseline with multiple seeds."""

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
            "--lr=0.003",
            "--load_from_checkpoint=experiments/exp5/warmup/warmup.tar",
            f"--checkpoint_dir=experiments/exp5/baseline/s{seed}",
            "--n_epochs=400",
            "--compute_cross_gen_online",
            "--deterministic",
            "--tensorboard",
            "--snapshot_freq=10",
        ])
    return configs
