# RL AI: ParallelSAC Training Pipeline

This project implements a multi-CPU reinforcement learning training pipeline using the SAC (Soft Actor-Critic) algorithm from Stable Baselines3.

## Setup

1.  Create and activate a virtual environment.
2.  Install dependencies using `pip-tools`:
    ```bash
    pip install pip-tools
    pip-compile --extra-index-url https://download.pytorch.org/whl/cpu --output-file requirements.txt pyproject.toml
    pip install -r requirements.txt
    ```

## Local Training

To train the agent locally:
```bash
python src/train.py --env-id Pendulum-v1 --total-timesteps 40000
```

## AWS Training

To train the agent on AWS SageMaker:
```bash
python aws-train.py
```

## Features

-   **ParallelSAC**: Dynamically scales to available CPUs using `SubprocVecEnv`.
-   **Logging**: Replicates `object-detection` style logging for TensorBoard.
-   **Evaluation**: Records and saves videos of evaluation episodes to `output/videos`.

## Sync dependencies
1. Compile dev dependencies `pip-compile --extra dev -o dev-requirements.txt pyproject.toml `
2. Compile general dependencies `pip-compile --constrain dev-requirements.txt  -o src/requirements.txt pyproject.toml`
3. Sync dependencies `pip-sync src/requirements.txt dev-requirements.txt`