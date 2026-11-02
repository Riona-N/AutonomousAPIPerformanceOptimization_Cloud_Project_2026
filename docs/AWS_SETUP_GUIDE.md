# AWS Setup & Deployment Guide (Role 2 — Anushka)

This guide provides step-by-step instructions for deploying and configuring the cloud infrastructure on AWS for the **Autonomous API Performance Optimization Framework**.

You have two ways to set this up:
- **Method A (Recommended):** Automated deployment using the **AWS SAM CLI** (takes 5 minutes, avoids manual mistakes).
- **Method B:** Manual setup using the **AWS Management Console** (step-by-step click guide, ideal for capturing report screenshots).

---

## 1. Prerequisites

1. **AWS Account:** Free tier account is sufficient.
2. **AWS CLI installed:**
   Verify by running in terminal:
   ```powershell
   aws --version
   ```
3. **AWS Credentials configured:**
   Run:
   ```powershell
   aws configure
   ```
   Enter your:
   - `AWS Access Key ID`
   - `AWS Secret Access Key`
   - `Default region name` (e.g., `us-east-1`)
   - `Default output format` (`json`)

---

## Method A: Automated Deployment via AWS SAM CLI (Fastest)

1. **Install AWS SAM CLI** (if not already installed):
   Download from: https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/install-sam-cli.html

2. **Build the Application:**
   From the repository root (`cad_project`):
   ```powershell
   sam build
   ```

3. **Deploy to AWS:**
   ```powershell
   sam deploy --guided
   ```
   Follow the interactive prompts:
   - **Stack Name:** `api-perf-optimization-stack`
   - **AWS Region:** `us-east-1` (or your chosen region)
   - **Parameter SageMakerEndpointName:** `role1-ppo` (or press enter for default)
   - **Confirm changes before deploy:** `Y`
   - **Allow SAM CLI to create IAM roles:** `Y`
   - **Disable rollback for debugging:** `N`
   - **Save arguments to configuration file:** `Y`

4. **Outputs:**
   SAM will output:
   - `ApiEndpoint`: The live URL to send transactions to.
   - `StateMachineArn`: ARN of the Step Functions workflow.
   - `TransactionLogsTableName`: Name of the DynamoDB logs table.
   - `SystemMetricsCacheTableName`: Name of the DynamoDB metrics cache table.

---

## Method B: Manual AWS Management Console Setup (Click-by-Click)

If you need to set up the resources manually in the AWS Console for screenshots or verification:

### Step 1: Create DynamoDB Tables
1. Open the [AWS DynamoDB Console](https://console.aws.amazon.com/dynamodbv2/).
2. Click **Create table**:
   - **Table name:** `TransactionLogs`
   - **Partition key:** `transaction_id` (Type: `String`)
   - **Sort key:** `timestamp` (Type: `Number`)
   - **Table settings:** Select **Customize settings** → **Capacity mode:** **On-demand** (Free tier friendly).
   - Click **Create table**.
3. Create a second table:
   - **Table name:** `SystemMetricsCache`
   - **Partition key:** `stage_candidate` (Type: `String`)
   - **Table settings:** Select **On-demand**.
   - Click **Create table**.

### Step 2: Create IAM Role for Lambdas
1. Open the [IAM Console](https://console.aws.amazon.com/iam/).
2. Click **Roles** → **Create role**.
3. Select **AWS service** → Use case: **Lambda**.
4. Attach policies:
   - `AWSLambdaBasicExecutionRole`
   - `AmazonDynamoDBFullAccess`
   - `CloudWatchFullAccess`
5. Name the role: `AutonomousPaymentLambdaRole` → Click **Create role**.

### Step 3: Create the Lambda Functions
1. Open the [AWS Lambda Console](https://console.aws.amazon.com/lambda/).
2. Create 4 functions (Runtime: **Python 3.11**, Execution Role: `AutonomousPaymentLambdaRole`):
   - **Function 1:** `StateBuilderFunction`
     - Paste code from `src/aws/lambda_state_builder.py` and `src/aws/dynamodb_helpers.py`.
     - Handler: `lambda_state_builder.lambda_handler`.
   - **Function 2:** `MicroservicePipelineFunction`
     - Paste code from `src/aws/lambda_microservices.py`.
     - Handler: `lambda_microservices.microservice_pipeline_handler`.
   - **Function 3:** `ActionExecutorFunction`
     - Paste code from `src/aws/lambda_action_executor.py`.
     - Handler: `lambda_action_executor.lambda_handler`.
   - **Function 4:** `MetricsPublisherFunction`
     - Paste code from `src/aws/lambda_metrics_publisher.py`.
     - Handler: `lambda_metrics_publisher.lambda_handler`.
3. Under **Configuration** → **General configuration**:
   - Set **Timeout:** `30 sec`
   - Set **Memory:** `512 MB`

### Step 4: Create the Step Functions State Machine
1. Open the [AWS Step Functions Console](https://console.aws.amazon.com/states/).
2. Click **Create state machine**.
3. Choose **Blank template** → Switch to **Code** tab.
4. Paste the content of `src/aws/step_function.json`.
5. Replace `${StateBuilderFunctionArn}`, `${ActionExecutorFunctionArn}`, and `${MetricsPublisherFunctionArn}` with the ARNs of your created Lambda functions.
6. Click **Next** → Name: `ClosedLoopPaymentController` → Click **Create state machine**.

### Step 5: Create Amazon API Gateway
1. Open the [API Gateway Console](https://console.aws.amazon.com/apigateway/).
2. Click **Create API** → Choose **REST API** (Build).
3. Name: `AutonomousPaymentAPI` → Click **Create API**.
4. Click **Create Resource**:
   - Resource Name: `transaction`
   - Resource Path: `/transaction`
5. Click **Create Method** on `/transaction`:
   - Method type: **POST**
   - Integration type: **Lambda Function**
   - Select `StateBuilderFunction` (or StartExecution on Step Function).
6. Enable CORS: Click **Enable CORS** → Save.
7. Click **Deploy API** → Stage name: `Prod`.
8. Copy the **Invoke URL** (e.g., `https://xxxx.execute-api.us-east-1.amazonaws.com/Prod/transaction`).

### Step 6: Create the CloudWatch Dashboard
1. Open the [CloudWatch Console](https://console.aws.amazon.com/cloudwatch/).
2. Click **Dashboards** → **Create dashboard**.
3. Name: `AutonomousPaymentGateway-Dashboard`.
4. Click **Actions** → **View/edit source**.
5. Paste the JSON from `src/aws/cloudwatch_dashboard.json`.
6. Click **Update** → **Save dashboard**.

---

## 3. Testing the Deployed System

### Test via cURL or PowerShell:
```powershell
curl -X POST "https://YOUR_API_ID.execute-api.us-east-1.amazonaws.com/Prod/transaction" `
     -H "Content-Type: application/json" `
     -d '{
       "transaction_id": "test_txn_999",
       "type": "PAYMENT",
       "amount": 2500.0,
       "fraud_risk": 0.02
     }'
```

### Expected Response:
```json
{
  "statusCode": 200,
  "transaction_id": "test_txn_999",
  "action": {
    "ingress": 0,
    "auth": 1,
    "routing": 0,
    "processor": 2,
    "db": 1
  },
  "explanation": {
    "dominant_reward_component": "success_bonus",
    "dominant_value": 1.0,
    "e2e_ms": 142.5
  }
}
```

---

## 4. Screenshots to Take for Your Report & Presentation

For your project deliverables and viva, take screenshots of the following:

1. **API Gateway Console:** Showing the `POST /transaction` method and resource hierarchy.
2. **Step Functions Console:** Showing the state machine graph (`StateBuilder` → `ActionExecutor` → `MetricsPublisher`) with a successful green execution run.
3. **DynamoDB Console:** Showing items in the `TransactionLogs` table (revealing recorded routing decisions, stage latencies, SLA flags, and explanations).
4. **CloudWatch Dashboard:** Showing the graphs for P95 latency vs SLA threshold and success rates.
5. **Terminal / Test Run:** Screenshot of running `python -m unittest discover -s tests -p "test_aws_integration.py"` showing `Ran 9 tests ... OK`.

---

## 5. Free-Tier Safety & Clean Up

When you are done with testing or demonstration:
```powershell
sam delete --stack-name api-perf-optimization-stack
```
This automatically tears down all created resources, ensuring zero unexpected cloud costs.
