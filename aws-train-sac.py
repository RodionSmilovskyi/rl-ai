import os
import argparse
import boto3
import sagemaker
from sagemaker.train import ModelTrainer
from sagemaker.train.configs import SourceCode, Compute

# Replicating configuration logic from object-detection/aws-train.py
WORKDIR = os.path.dirname(os.path.abspath(__file__))
ROLE = "arn:aws:iam::905418352696:role/SageMakerFullAccess"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", type=str, default="rl-training-run")
    parser.add_argument("--job-name", type=str, default=None)
    parser.add_argument("--total-timesteps", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--env-id", type=str, default="Pendulum-v1")
    args = parser.parse_args()

    # AWS Session initialization (mimicking object-detection)
    boto_session = boto3.session.Session(
        profile_name="905418352696_AdministratorAccess", region_name="us-east-1"
    )
    from sagemaker.core.helper.session_helper import Session
    sagemaker_session = Session(boto_session=boto_session)
    output_path = f"s3://{sagemaker_session.default_bucket()}/{args.prefix}"

    # V3 uses ModelTrainer instead of Estimator
    trainer = ModelTrainer(
        sagemaker_session=sagemaker_session,
        training_image=f"763104351884.dkr.ecr.{boto_session.region_name}.amazonaws.com/pytorch-training:2.6.0-cpu-py312-ubuntu22.04-sagemaker",
        role=ROLE,
        source_code=SourceCode(
            source_dir="src",
            entry_script="train_sac.py"
        ),
        compute=Compute(
            instance_count=1,
            instance_type="ml.m5.large", # Cheapest available instance for training in us-east-1
        ),
        hyperparameters={
            "total-timesteps": args.total_timesteps,
            "seed": args.seed,
            "lr": args.lr,
            "batch-size": args.batch_size,
            "env-id": args.env_id,
            "prefix": args.prefix
        },
    )

    # Note: No explicit training data upload required for Pendulum, as it's built into Gymnasium.
    job_name_kwargs = {"job_name": args.job_name} if args.job_name else {}
    trainer.train(wait=False, **job_name_kwargs)

    print(f"SageMaker Training Job submitted: {args.job_name or args.prefix}")
