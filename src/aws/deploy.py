"""Create or update a SageMaker endpoint for the Role 1 PPO model.

The inference container must include this repository's ``src/aws/inference.py``
and the runtime dependencies from ``requirements.txt``. Cloud credentials are
resolved by boto3's normal credential chain; no credentials are read from files.
"""

from __future__ import annotations

import argparse
import os
import time


def deploy(
    *,
    model_data: str,
    image: str,
    role: str,
    endpoint_name: str,
    instance_type: str = "ml.m5.large",
    region: str | None = None,
) -> str:
    import boto3

    session = boto3.session.Session(region_name=region)
    client = session.client("sagemaker")
    suffix = str(int(time.time()))
    model_name = f"{endpoint_name}-model-{suffix}"
    config_name = f"{endpoint_name}-config-{suffix}"
    client.create_model(
        ModelName=model_name,
        ExecutionRoleArn=role,
        PrimaryContainer={"Image": image, "ModelDataUrl": model_data},
    )
    client.create_endpoint_config(
        EndpointConfigName=config_name,
        ProductionVariants=[
            {
                "VariantName": "AllTraffic",
                "ModelName": model_name,
                "InitialInstanceCount": 1,
                "InstanceType": instance_type,
                "InitialVariantWeight": 1.0,
            }
        ],
    )
    try:
        client.describe_endpoint(EndpointName=endpoint_name)
    except client.exceptions.ResourceNotFound:
        client.create_endpoint(EndpointName=endpoint_name, EndpointConfigName=config_name)
    else:
        client.update_endpoint(EndpointName=endpoint_name, EndpointConfigName=config_name)
    return endpoint_name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-data", default=os.getenv("SAGEMAKER_MODEL_DATA"))
    parser.add_argument("--image", default=os.getenv("SAGEMAKER_INFERENCE_IMAGE"))
    parser.add_argument("--role", default=os.getenv("SAGEMAKER_EXECUTION_ROLE_ARN"))
    parser.add_argument("--endpoint-name", default=os.getenv("SAGEMAKER_ENDPOINT_NAME", "role1-ppo"))
    parser.add_argument("--instance-type", default=os.getenv("SAGEMAKER_INSTANCE_TYPE", "ml.m5.large"))
    parser.add_argument("--region", default=os.getenv("AWS_REGION"))
    args = parser.parse_args()
    missing = [name for name in ("model_data", "image", "role") if not getattr(args, name)]
    if missing:
        parser.error("missing required values: " + ", ".join(f"--{name.replace('_', '-')}" for name in missing))
    print(deploy(**vars(args)))


if __name__ == "__main__":
    main()