# Guide 3: Deploy Using the Three-Object Pattern

SageMaker deployments use three distinct objects that reference each other in a chain: **Model** (what to run), **Endpoint Configuration** (how to run it), and **Endpoint** (where to run it). You saw this pattern briefly in Module 1 when deploying a JumpStart model. Now you will create each object manually to understand exactly what each one does.

---

## Steps

### Step 1 -- Create the Model Object

1. In the **SageMaker console**, go to **Inference** -> **Models**.
2. Click **Create model**.
3. Configure:
   - **Model name:** `fraud-rf-v1`
   - **IAM role:** Select the execution role used by your Studio Domain (the `AmazonSageMaker-ExecutionRole-*` auto-created role)
   - **Container input options:** Select **Provide model artifacts and inference image location**
   - **Inference image URI:** Paste the ECR container image URI from your training job (same value you used during model registration)
   - **Model data URL:** Paste the S3 URI to your `model.tar.gz`
4. Click **Create model**.

The Model object tells SageMaker: "Here is the container that knows how to load and serve predictions from this specific model artifact."

### Step 2 -- Create the Endpoint Configuration

1. Go to **Inference** -> **Endpoint configurations**.
2. Click **Create endpoint configuration**.
3. Configure:
   - **Endpoint configuration name:** `fraud-rf-v1-config`
4. Under **Production variants**, click **Add model**:
   - Select `fraud-rf-v1` (the model you just created)
   - **Variant name:** `AllTraffic` (default -- all requests go to this model)
   - **Instance type:** `ml.m5.xlarge`
   - **Initial instance count:** `1`
   - **Initial weight:** `1` (100% of traffic)
5. Click **Create endpoint configuration**.

The Endpoint Configuration tells SageMaker: "Run this model on this instance type with this many instances."

### Step 3 -- Create the Endpoint

1. Go to **Inference** -> **Endpoints**.
2. Click **Create endpoint**.
3. Configure:
   - **Endpoint name:** `fraud-rf-v1-endpoint`
   - **Endpoint configuration:** Select `fraud-rf-v1-config`
4. Click **Create endpoint**.

The Endpoint is the live HTTPS service. SageMaker will now:
1. Provision an `ml.m5.xlarge` instance
2. Pull the container image from ECR
3. Download `model.tar.gz` from S3 and extract it
4. Start the inference server
5. Begin accepting prediction requests

### Step 4 -- Wait for InService

1. The endpoint status will show **Creating**. This typically takes 5-10 minutes.
2. Refresh the page periodically until the status changes to **InService** (green).
3. While waiting, explore the endpoint details:
   - **Production variants:** Shows the model, instance type, and traffic weight
   - **Monitoring:** Will show CloudWatch metrics once the endpoint is active
   - **Endpoint configuration:** Links back to the configuration you created

### Step 5 -- Verify the Three-Object Chain

Navigate through all three console sections and verify the references:

```
Inference -> Endpoints
  └── fraud-rf-v1-endpoint
       └── references: fraud-rf-v1-config

Inference -> Endpoint configurations
  └── fraud-rf-v1-config
       └── references: fraud-rf-v1 (model)

Inference -> Models
  └── fraud-rf-v1
       └── references: ECR image + S3 model.tar.gz
```

This chain is why cleanup order matters (Guide 5): you must delete from the outside in -- Endpoint first, then Configuration, then Model.

### Step 6 -- Understand Why Three Objects

The three-object pattern exists for flexibility:

| Scenario | What Changes | What Stays |
|----------|-------------|------------|
| Deploy a new model version | Create new Model + Config | Endpoint name stays the same (update in place) |
| Change instance type | Create new Config | Model and Endpoint name stay the same |
| A/B test two models | Config with two production variants | Same Endpoint serves both |
| Blue/green deployment | New Config referencing new Model | Switch Endpoint to new Config with zero downtime |

In all cases, the Endpoint name (the URL that clients call) can remain stable while the underlying Model and Configuration change.

---

## Presentation Checkpoint

Be prepared to show:
- All three objects in the console: Model (`fraud-rf-v1`), Config (`fraud-rf-v1-config`), Endpoint (`fraud-rf-v1-endpoint`)
- The reference chain: Endpoint -> Config -> Model -> S3/ECR
- The endpoint with **InService** status
- Explain: Why does SageMaker separate deployment into three objects? (Flexibility -- you can update the model without changing the endpoint URL, change instance types without redeploying, or A/B test multiple models behind the same endpoint)
- Explain: What happens behind the scenes when an endpoint reaches InService? (SageMaker provisions the instance, pulls the container, downloads and extracts model.tar.gz, starts the inference server, and begins accepting requests)

---

## Cost Warning

Your endpoint is now **running and incurring charges**. Complete Guide 4 (Invoke) and Guide 5 (Cleanup) promptly. Do not leave the endpoint running overnight.

---

## Key Concepts

- **Three-Object Pattern:** Model + Endpoint Configuration + Endpoint. Each object has a distinct responsibility and can be updated independently.
- **Production Variant:** A model deployed within an endpoint configuration. Multiple variants enable A/B testing and traffic splitting.
- **Endpoint URL:** Each endpoint gets a unique HTTPS URL that clients call for predictions. This URL remains stable even as the underlying model changes.
