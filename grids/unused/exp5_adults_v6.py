"""Exp5 Adults v6: all agents have uniform adult (low) plasticity
Adults: temp=0.5, lr=0.001
Parameters match the adult values in age_based_stagger_adults_v6 for fair comparison.
This is the control condition where everyone behaves like a mature adult.
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
        "--lr_max=0.001",
        "--temp_min=0.5",
        "--temp_max=0.5",
        "--plasticity_max_age=100",
        "--load_from_checkpoint=experiments/exp5/warmup/warmup.tar",
        "--checkpoint_dir=experiments/exp5/adults_v6",
        "--n_epochs=400",
        "--compute_cross_gen_online",
        "--deterministic",
        "--tensorboard",
        "--snapshot_freq=10",
    ]
    return [base]
