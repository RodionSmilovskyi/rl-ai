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
