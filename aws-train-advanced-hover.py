#!/usr/bin/env python
import os
import argparse
import boto3
import sagemaker
from sagemaker.train.model_trainer import ModelTrainer, StoppingCondition
from sagemaker.core.training.configs import SourceCode, Compute, InputData, TensorBoardOutputConfig

# Replicating configuration logic from object-detection/aws-train.py
WORKDIR = os.path.dirname(os.path.abspath(__file__))
ROLE = "arn:aws:iam::905418352696:role/SageMakerFullAccess"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-name", type=str, default="sac-advanced-hover")
    parser.add_argument("--total-timesteps", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-cpus", type=int, default=None)
    
    # New arguments matching train_advanced_hover.py
    parser.add_argument("--ent-coef", type=float, default=0.001)
    parser.add_argument("--train-freq", type=int, default=64)
    parser.add_argument("--gradient-steps", type=int, default=64)
    parser.add_argument("--critic-warmup-steps", type=int, default=20000)
    parser.add_argument("--bc-data-dir", type=str, default=None, help="Local path to expert dataset directory")
    parser.add_argument("--bc-epochs", type=int, default=100)
    parser.add_argument("--bc-batch-size", type=int, default=64)
    parser.add_argument("--success-threshold", type=float, default=25.0)
    parser.add_argument("--n-eval-episodes", type=int, default=20)

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
        "ent-coef": args.ent_coef,
        "train-freq": args.train_freq,
        "gradient-steps": args.gradient_steps,
        "critic-warmup-steps": args.critic_warmup_steps,
        "bc-epochs": args.bc_epochs,
        "bc-batch-size": args.bc_batch_size,
        "success-threshold": args.success_threshold,
        "n-eval-episodes": args.n_eval_episodes,
    }
    
    tensorboard_config = TensorBoardOutputConfig(
        s3_output_path=f"s3://{sagemaker_session.default_bucket()}/{args.job_name}/tensorboard",
        local_path="/opt/ml/output/tensorboard"
    )

    if args.num_cpus is not None:
        hyperparameters["num-cpus"] = args.num_cpus

    input_data_config = []

    if args.bc_data_dir:
        if not os.path.exists(args.bc_data_dir):
            raise FileNotFoundError(f"BC data directory not found: {args.bc_data_dir}")
        
        print(f"Uploading expert dataset from {args.bc_data_dir} to S3...")
        s3_bc_uri = sagemaker_session.upload_data(
            path=args.bc_data_dir,
            key_prefix=f"{args.job_name}/input/bc_data"
        )
        print(f"Uploaded to {s3_bc_uri}")
        
        # SageMaker will download the S3 folder to /opt/ml/input/data/bc_data
        hyperparameters["bc-data-dir"] = "/opt/ml/input/data/bc_data"
        
        input_data_config.append(
            InputData(
                channel_name="bc_data",
                data_source=s3_bc_uri
            )
        )

    trainer = ModelTrainer(
        sagemaker_session=sagemaker_session,
        training_image="763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.6.0-cpu-py312-ubuntu22.04-sagemaker",
        role=ROLE,
        base_job_name=args.job_name,
        source_code=SourceCode(
            source_dir="src",
            entry_script="train_advanced_hover.py",
            requirements="requirements.txt"
        ),
        compute=Compute(
            instance_count=1,
            instance_type="ml.c5.9xlarge", # Optimized for multi-CPU ParallelSAC
        ),
        hyperparameters=hyperparameters,
        input_data_config=input_data_config if input_data_config else None,
        stopping_condition=StoppingCondition(
            max_runtime_in_seconds=6 * 60 * 60 # 6 hours
        ),
        environment={
            "TF_ENABLE_ONEDNN_OPTS": "0"
        }
    )

    trainer.with_tensorboard_output_config(tensorboard_config)
    trainer.train(wait=False)

    # Manual cleanup of internal SageMaker temp dirs to avoid messy __del__ exceptions on exit
    for attr in ["_temp_recipe_train_dir", "_temp_code_dir"]:
        temp_dir = getattr(trainer, attr, None)
        if temp_dir is not None:
            temp_dir.cleanup()
            setattr(trainer, attr, None)

    print(f"SageMaker Advanced Hover Training Job submitted: {args.job_name}")

if __name__ == "__main__":
    main()
