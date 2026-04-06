import os
import argparse
import boto3
import sagemaker
from sagemaker.train.model_trainer import ModelTrainer
from sagemaker.core.training.configs import SourceCode, Compute

# Replicating configuration logic from object-detection/aws-train.py
WORKDIR = os.path.dirname(os.path.abspath(__file__))
ROLE = "arn:aws:iam::905418352696:role/SageMakerFullAccess"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", type=str, default="ppo-altitude-run")
    parser.add_argument("--job-name", type=str, default=None)
    parser.add_argument("--total-timesteps", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-phase", type=int, default=4)
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
        training_image=f"763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.6.0-cpu-py312-ubuntu22.04-sagemaker",
        role=ROLE,
        base_job_name=args.job_name or args.prefix,
        source_code=SourceCode(
            source_dir="src",
            entry_script="train_altitude_curriculum_ppo.py",
            requirements="requirements.txt"
        ),
        compute=Compute(
            instance_count=1,
            instance_type="ml.c5.4xlarge", # Optimized for multi-CPU Parallel training
        ),
        hyperparameters={
            "total-timesteps": args.total_timesteps,
            "seed": args.seed,
            "lr": args.lr,
            "batch-size": args.batch_size,
            "prefix": args.prefix,
            "max-phase": args.max_phase
        },
    )

    trainer.train(wait=False)

    # Manual cleanup of internal SageMaker temp dirs to avoid messy __del__ exceptions on exit
    for attr in ["_temp_recipe_train_dir", "_temp_code_dir"]:
        temp_dir = getattr(trainer, attr, None)
        if temp_dir is not None:
            temp_dir.cleanup()
            setattr(trainer, attr, None)

    print(f"SageMaker Altitude PPO Training Job submitted: {args.job_name or args.prefix}")

if __name__ == "__main__":
    main()
