#!/usr/bin/env python
import os
import argparse
import boto3
import sagemaker
from sagemaker.train.model_trainer import ModelTrainer, StoppingCondition
from sagemaker.core.training.configs import SourceCode, Compute, InputData

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
    parser.add_argument("--model-zip", type=str, default=None, help="Path to existing model .zip to fine-tune")
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
    }
    
    if args.num_cpus is not None:
        hyperparameters["num-cpus"] = args.num_cpus

    input_data_config = None

    if args.model_zip:
        if not os.path.exists(args.model_zip):
            raise FileNotFoundError(f"Model zip file not found: {args.model_zip}")
            
        print(f"Uploading {args.model_zip} to S3...")
        # Upload data to s3, this will upload the file to [bucket]/[prefix]/input/model/<filename>
        s3_uri = sagemaker_session.upload_data(
            path=args.model_zip,
            key_prefix=f"{args.job_name}/input/model"
        )
        print(f"Uploaded to {s3_uri}")
        
        # SageMaker will download the S3 object to /opt/ml/input/data/model (since channel name is 'model')
        # The file inside the container will be /opt/ml/input/data/model/<filename>
        filename = os.path.basename(args.model_zip)
        hyperparameters["model-zip"] = f"/opt/ml/input/data/model/{filename}"
        
        input_data_config = [
            InputData(
                channel_name="model",
                data_source=s3_uri
            )
        ]

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
        input_data_config=input_data_config,
        stopping_condition=StoppingCondition(
            max_runtime_in_seconds=3 * 60 * 60 # 3 hours
        ),
        environment={
            "TF_ENABLE_ONEDNN_OPTS": "0"
        }
    )

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