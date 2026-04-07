import os
import argparse
import boto3
import sagemaker
from sagemaker.train.model_trainer import ModelTrainer, StoppingCondition
from sagemaker.core.training.configs import SourceCode, Compute

# Replicating configuration logic from object-detection/aws-train.py
WORKDIR = os.path.dirname(os.path.abspath(__file__))
ROLE = "arn:aws:iam::905418352696:role/SageMakerFullAccess"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-name", type=str, default="sac-altitude")
    parser.add_argument("--total-timesteps", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-cpus", type=int, default=None)
    parser.add_argument("--max-phase", type=int, default=4)
    args = parser.parse_args()

    # AWS Session initialization (mimicking object-detection)
    boto_session = boto3.session.Session(
        profile_name="905418352696_AdministratorAccess", region_name="us-east-1"
    )
    from sagemaker.core.helper.session_helper import Session
    sagemaker_session = Session(boto_session=boto_session)

    hyperparameters = {
        "total-timesteps": args.total_timesteps,
        "seed": args.seed,
        "lr": args.lr,
        "batch-size": args.batch_size,
        "prefix": args.job_name,
        "max-phase": args.max_phase
    }

    if args.num_cpus is not None:
        hyperparameters["num-cpus"] = args.num_cpus

    # V3 uses ModelTrainer instead of Estimator
    trainer = ModelTrainer(
        sagemaker_session=sagemaker_session,
        training_image="763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.6.0-cpu-py312-ubuntu22.04-sagemaker",
        role=ROLE,
        base_job_name=args.job_name,
        source_code=SourceCode(
            source_dir="src",
            entry_script="train_altitude_curriculum.py",
            requirements="requirements.txt"
        ),
        compute=Compute(
            instance_count=1,
            instance_type="ml.c5.9xlarge", # Optimized for multi-CPU ParallelSAC
        ),
        hyperparameters=hyperparameters,
        stopping_condition=StoppingCondition(
            max_runtime_in_seconds=6 * 60 * 60 # 6 hours
        ),
        environment={
            "TF_ENABLE_ONEDNN_OPTS": "0"
        }
    )

    # Note: No explicit training data upload required as assets are in source_dir/assets (handled by SageMaker)
    trainer.train(wait=False)

    # Manual cleanup of internal SageMaker temp dirs to avoid messy __del__ exceptions on exit
    for attr in ["_temp_recipe_train_dir", "_temp_code_dir"]:
        temp_dir = getattr(trainer, attr, None)
        if temp_dir is not None:
            temp_dir.cleanup()
            setattr(trainer, attr, None)

    print(f"SageMaker Altitude Training Job submitted: {args.job_name}")

if __name__ == "__main__":
    main()
