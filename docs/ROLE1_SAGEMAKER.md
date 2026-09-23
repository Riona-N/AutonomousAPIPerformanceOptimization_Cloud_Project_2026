# Role 1 SageMaker deployment

The trained artifact is `results/data_ppo_agent.zip`. Package that file as
`model.tar.gz`, upload it to S3, and provide the S3 URI to the deployment CLI.
The endpoint container must run `src.aws.inference` using the dependencies in
`requirements.txt`.

```powershell
New-Item -ItemType Directory -Force -Path .\dist | Out-Null
tar -czf .\dist\role1-model.tar.gz -C .\results data_ppo_agent.zip
aws s3 cp .\dist\role1-model.tar.gz s3://YOUR_BUCKET/role1/model.tar.gz

$env:SAGEMAKER_MODEL_DATA = "s3://YOUR_BUCKET/role1/model.tar.gz"
$env:SAGEMAKER_INFERENCE_IMAGE = "YOUR_ACCOUNT.dkr.ecr.YOUR_REGION.amazonaws.com/role1-inference:latest"
$env:SAGEMAKER_EXECUTION_ROLE_ARN = "arn:aws:iam::YOUR_ACCOUNT:role/YOUR_SAGEMAKER_ROLE"
$env:SAGEMAKER_ENDPOINT_NAME = "role1-ppo"
python -m src.aws.deploy
```

Invoke the endpoint with one 83-value observation. The optional context fields
`e2e_ms`, `sla_ms`, `timeout_ms`, and `fraud_risk` make the explanation use the
realized reward; without `e2e_ms`, it reports a pre-outcome P95 estimate.

```python
from src.ml_model.data_api_env import DataApiRoutingEnv

env = DataApiRoutingEnv(split="test")
observation, _ = env.reset()
payload = {
    "instances": [{
        "observation": observation.tolist(),
        "context": {"e2e_ms": 734.0, "sla_ms": 1000.0, "timeout_ms": 2000.0, "fraud_risk": 0.02},
    }]
}
```

The response contains the selected candidate for each stage and
`explanation.dominant_reward_component`, which is the reward term with the
largest absolute contribution: success bonus, latency penalty, SLA violation
penalty, or fraud-risk bonus.