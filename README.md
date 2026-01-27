# Age-Based Plasticity in Emergent Communication

A referential game with population dynamics and age-dependent plasticity.
Agents learn to communicate while the population undergoes generational turnover.

## Quick Start

```bash
# Single run
python -m egg.zoo.aging.train --n_agents=4 --n_epochs=100

# Grid search
python -m egg.nest.nest_local --game egg.zoo.aging.train \
    --sweep egg/zoo/aging/grids/exp1_baseline.json --n_workers=2
```

## Installation

### 1. Clone EGG
```bash
git clone https://github.com/facebookresearch/EGG.git
cd EGG
```

### 2. Create environment
```bash
conda create -n egg-aging python=3.9 -y
conda activate egg-aging
```

### 3. Install PyTorch
Choose the command for your CUDA version (see https://pytorch.org):
```bash
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CPU only
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 4. Download aging module
```bash
git clone https://github.com/Jano04/EGG.git aging
cp -r aging/* EGG/egg/zoo/aging/
```

### 5. Install dependencies
```bash
cd EGG
pip install -r egg/zoo/aging/requirements.txt
pip install -e .
```

### 6. Verify
```bash
python -c "import egg.zoo.aging.train; print('OK')"
```

## Running Experiments

All experiments use EGG's grid search tool. See [nest documentation](../../docs/nest.md).

### Experiment 1: Baseline Population Sizes
Static populations (N=2-20) without turnover.
```bash
python -m egg.nest.nest_local --game egg.zoo.aging.train \
    --sweep egg/zoo/aging/grids/exp1_baseline.json --n_workers=2
```

### Experiment 2: Agent Turnover

**2a. Rate study** (N=10, varying k):
```bash
# Create warmup checkpoint
python -m egg.nest.nest_local --game egg.zoo.aging.train \
    --sweep egg/zoo/aging/grids/exp2_rate/warmup.json --n_workers=1

# Run turnover conditions
python -m egg.nest.nest_local --game egg.zoo.aging.train \
    --sweep egg/zoo/aging/grids/exp2_rate/k1_k20.json --n_workers=2
python -m egg.nest.nest_local --game egg.zoo.aging.train \
    --sweep egg/zoo/aging/grids/exp2_rate/k2_k5_k10.json --n_workers=2
```

**2b. Population study** (k=10, varying N):
```bash
python -m egg.nest.nest_local --game egg.zoo.aging.train \
    --sweep egg/zoo/aging/grids/exp2_population/n2.json \
    --sweep egg/zoo/aging/grids/exp2_population/n4.json \
    --sweep egg/zoo/aging/grids/exp2_population/n8.json \
    --sweep egg/zoo/aging/grids/exp2_population/n16.json \
    --n_workers=2
```

### Experiment 3: Plasticity Conditions

```bash
# Create warmup checkpoint
python -m egg.nest.nest_local --game egg.zoo.aging.train \
    --sweep egg/zoo/aging/grids/exp3_plasticity/warmup.json --n_workers=1

# Run conditions
python -m egg.nest.nest_local --game egg.zoo.aging.train \
    --sweep egg/zoo/aging/grids/exp3_plasticity/baseline.json --n_workers=2
python -m egg.nest.nest_local --game egg.zoo.aging.train \
    --sweep egg/zoo/aging/grids/exp3_plasticity/adults.json --n_workers=2
python -m egg.nest.nest_local --game egg.zoo.aging.train \
    --sweep egg/zoo/aging/grids/exp3_plasticity/children.json --n_workers=2
python -m egg.nest.nest_local --game egg.zoo.aging.train \
    --py_sweep=egg.zoo.aging.grids.exp3_plasticity.age_based --n_workers=2
```

## Analysis

### Generate figures
```bash
python -m egg.zoo.aging.visualizations_exp1
python -m egg.zoo.aging.visualizations_exp2
python -m egg.zoo.aging.visualizations_exp3
```

### Post-hoc cross-generational analysis
```bash
python -m egg.zoo.aging.analysis experiments/exp2_turnover --recursive
```

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--n_agents` | 2 | Population size |
| `--kill_epoch` | 0 | Epochs between deaths (0 = no turnover) |
| `--warmup_threshold` | 0.0 | Accuracy threshold before turnover starts |
| `--enable_plasticity` | False | Enable age-based plasticity |
| `--temp_min/max` | 0.1/1.0 | Gumbel-Softmax temperature range |
| `--lr_min/max` | 1e-4/1e-3 | Learning rate range |
| `--plasticity_max_age` | 100 | Age when plasticity reaches minimum |

Full parameter list: `python -m egg.zoo.aging.train --help`
