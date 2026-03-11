import os
import argparse
import boto3
import sagemaker
from sagemaker.estimator import Estimator
from sagemaker.debugger import TensorBoardOutputConfig

# Replicating configuration logic from object-detection/aws-train.py
WORKDIR = os.path.dirname(os.path.abspath(__file__))
ROLE = "arn:aws:iam::905418352696:role/SageMakerFullAccess"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", type=str, default="ppo-training-run")
    parser.add_argument("--job-name", type=str, default=None)
    parser.add_argument("--total-timesteps", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--env-id", type=str, default="Pendulum-v1")
    args = parser.parse_args()

    # AWS Session initialization (mimicking object-detection)
    boto_session = boto3.session.Session(
        profile_name="905418352696_AdministratorAccess", region_name="us-east-1"
    )
    sagemaker_session = sagemaker.Session(boto_session=boto_session)
    output_path = f"s3://{sagemaker_session.default_bucket()}/{args.prefix}"
    checkpoint_path = f"s3://{sagemaker_session.default_bucket()}/{args.prefix}/checkpoint"

    estimator = Estimator(
        sagemaker_session=sagemaker_session,
        base_job_name=args.job_name,
        # Using a PyTorch training image compatible with SB3 and Gymnasium
        image_uri=f"763104351884.dkr.ecr.{boto_session.region_name}.amazonaws.com/pytorch-training:2.6.0-cpu-py312-ubuntu22.04-sagemaker",
        role=ROLE,
        max_run=24 * 60 * 60,
        instance_count=1,
        instance_type="ml.c5.4xlarge", # Optimized for multi-CPU Parallel training
        source_dir="src",
        entry_point="train_ppo.py",
        output_path=output_path,
        checkpoint_s3_uri=checkpoint_path,
        tensorboard_output_config=TensorBoardOutputConfig(
            s3_output_path=f"s3://{sagemaker_session.default_bucket()}/{args.prefix}/tensorboard"
        ),
        hyperparameters={
            "total-timesteps": args.total_timesteps,
            "seed": args.seed,
            "lr": args.lr,
            "n-steps": args.n_steps,
            "batch-size": args.batch_size,
            "env-id": args.env_id,
            "prefix": args.prefix
        },
    )

    # Note: No explicit training data upload required for Pendulum, as it's built into Gymnasium.
    estimator.fit(wait=False)

    print(f"SageMaker PPO Training Job submitted: {args.job_name or args.prefix}")
