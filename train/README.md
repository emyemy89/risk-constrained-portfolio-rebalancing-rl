# Training

This directory contains the code used to train, evaluate, and experiment with the Reinforcement Learning(RL) agent.

The main entry point is `run_training.py`. It ties together the portfolio environment, the prepared market data, the RL algorithm, and the evaluation pipeline. 
Training runs can be repeated with different seeds and configurations, while the resulting portfolio behaviour is saved for later inspection.

### What is where

* `run_training.py` — main training script where training runs are configured and executed;
* `make_env.py` — creates and configures the portfolio environment used by the agent;
* `evaluate.py` — evaluates a trained model and produces portfolio-level results and plots;
* `utils/` — small utilities used by the training pipeline, such as algorithm selection, run information, and inspecting the data before training;
* `experiments/` — separate experiments that are useful for testing assumptions about the strategy rather than being part of the main training pipeline;
* `tests/` — experimental and diagnostic tests, including the momentum baseline and predictability checks;
* `results/` — outputs from completed runs. Results are organised by training fold and random seed and include portfolio-value, allocation, and cash-weight plots together with debug information;
* `../logs/tensorboard/` — TensorBoard logs generated during RL training. These can be used to inspect learning behaviour across runs;

### How the training process fits together

The general flow is:

```text
historical data
      ↓
data pipeline + features
      ↓
portfolio environment
      ↓
RL algorithm (e.g. PPO / RecurrentPPO)
      ↓
training
      ↓
trained model
      ↓
evaluation
      ↓
portfolio performance + allocation plots
```

The training setup uses multiple random seeds so that results are not judged from a single lucky or unlucky run. The `results/` directory also reflects the walk-forward validation structure, with separate `fold_*` directories and a final training run.

`models/` at the project root stores the resulting trained model, while this directory is mainly concerned with **how the model is trained and how its behaviour is evaluated**.

### Running training

Run the main training script from the project root:

```bash
python -m train.run_training
```

The exact experiment configuration should be checked in `run_training.py`, since that is the source of truth for the current training setup.

The generated TensorBoard logs can be inspected with:

```bash
tensorboard --logdir logs/tensorboard
```

The large number of files under `logs/` and `results/` is expected: they represent individual training runs, validation folds, and random seeds rather than separate pieces of source code.


use ´tensorboard --logdir logs/tensorboard` for logs
use `rm -rf logs models` to delete old artifacts
