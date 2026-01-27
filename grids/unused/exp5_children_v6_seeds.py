"""Exp5 Children v6 with multiple seeds.
Children: temp=5.0, lr=0.025
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
            "--lr_min=0.025",
            "--lr_max=0.025",
            "--temp_min=5.0",
            "--temp_max=5.0",
            "--plasticity_max_age=100",
            "--load_from_checkpoint=experiments/exp5/warmup/warmup.tar",
            f"--checkpoint_dir=experiments/exp5/children_v6/s{seed}",
            "--n_epochs=400",
            "--compute_cross_gen_online",
            "--deterministic",
            "--tensorboard",
            "--snapshot_freq=10",
        ])
    return configs
