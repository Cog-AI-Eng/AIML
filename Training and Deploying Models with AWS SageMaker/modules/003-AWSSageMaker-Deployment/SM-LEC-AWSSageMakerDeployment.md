# AWSSageMaker-Deployment Lecture - Instructor Guide

**Total Duration:** 180 Minutes (3 Stages)
**Consolidated Activities:** Model Registry & Versioning (SM-CT-ModelRegistry&Versioning), Approval Workflows (SM-CT-ApprovalWorkflows), Real-time Inference Endpoints (SM-CT-RealtimeInferenceEndpoints), Invoking Endpoints (SM-CT-InvokingEndpoints)

| Block | Content | Minutes |
|-------|---------|---------|
| Stage 1 | Model Registry: Registering Versions and Designing Approval Criteria | 45 |
| Break 1 | Stretch / Questions | 10 |
| Stage 2 | Deploying to a Real-time Endpoint: Console and SDK | 45 |
| Break 2 | Stretch / Questions | 10 |
| Stage 3 | Invoking the Endpoint, Debugging, and Mandatory Cleanup | 45 |
| Buffer | Open Q&A, Git Branch Activity, Wrap-Up | 25 |

---

## Lecture Overview

**Unified Scenario -- FraudShield Risk Analytics (continued)**

The FraudShield data science team has completed Module 2: they adapted a local scikit-learn training script for SageMaker Script Mode, launched a training job, and verified the `model.tar.gz` artifact in S3. Leadership has reviewed the training accuracy and given the green light: "Put this model where other teams can use it."

Today the team takes the model from artifact to production-ready:

1. **"How do we track which model version is approved for production?"** (Model Registry, approval workflow)
2. **"How do we deploy the approved model so applications can call it?"** (Three-object pattern, real-time endpoint)
3. **"How do we actually send data and get predictions back?"** (Invocation with `boto3` and `Predictor`, cleanup)

This scenario threads through every Module 3 exit criterion: Associates will register a model in the Model Registry, set approval criteria, deploy a real-time endpoint using the console, invoke it with `boto3`, and clean up all resources.

**Console + SDK:** This lecture starts console-first (Registry, approval, deployment) and transitions to SDK for invocation. Associates see both paths for every operation.

---

## Pre-Lecture Setup

### Instructor Checklist

- [ ] Completed Module 2 training job with a successful `model.tar.gz` artifact in S3
- [ ] S3 artifact path noted (e.g., `s3://<bucket>/fraudshield/output/<job-name>/output/model.tar.gz`)
- [ ] SageMaker Studio open with a notebook ready
- [ ] The scikit-learn managed container image URI available (from the training job details page under "Training image")
- [ ] IAM execution role with S3 read, CloudWatch write, and SageMaker invoke permissions
- [ ] Screen sharing enabled, font increased for projector readability
- [ ] This instructor guide open in a second tab

### Quick Reference: Locating the Artifact

If you need to find the artifact path from Module 2:

```python
import sagemaker
session = sagemaker.Session()
bucket = session.default_bucket()

# Option A: If you still have the estimator object
print(estimator.model_data)

# Option B: From the console
# SageMaker > Training > Training jobs > click job > Output data configuration
```

### Student Prerequisites

- [ ] Completed Module 2 lecture (trained model artifact in S3)
- [ ] Completed readings: Model Registry & Versioning, Approval Workflows, Real-time Inference Endpoints, Invoking Endpoints
- [ ] Studio notebook open
- [ ] S3 artifact path from their own Module 2 training job

---

## Stage 1: Model Registry -- Registering Versions and Designing Approval Criteria

**Duration:** 45 minutes
**Exit Criteria Addressed:**
- Register a trained model version in the SageMaker Model Registry (Required)
- Design an approval workflow for model versions within the Model Registry (Required)
- Define model packages and versions in the Model Registry with metadata and tags (Preferred)

### Instructor Opening (5 minutes -- talk, no code)

> "In Module 2, your training job produced a `model.tar.gz` sitting in S3. But imagine FraudShield trains models every week. After a month, you have four artifacts in S3. Which one is the good one? Which one is currently serving customers? Which one was trained on the dataset with the corrected feature engineering? S3 stores files, but it does not answer these questions."

> "The Model Registry is SageMaker's answer. Think of it as Git for models. A Model Package Group is a repository -- one per logical model. A Model Package is a commit -- one per training run. And approval status is code review -- the gate between 'experimental' and 'production.' Today we register our fraud detection model, define approval criteria, approve it, and deploy it."

---

### STEP 1 -- Creating a Model Package Group in the Console (10 minutes)

**Pacing: live demonstration.** All Associates follow along.

1. Open the SageMaker console. Navigate to **Governance > Model registry** (or **Home > Models > Model registry**).
2. Click **Create model package group**.
3. Fill in the details:
   - **Group name:** `fraud-detection-rf`
   - **Description:** "Random Forest classifier for transaction fraud detection, trained on tabular transaction data using scikit-learn Script Mode."
4. **Tags:** Click **Add tag** for each:
   - Key: `project`, Value: `fraud-detection`
   - Key: `team`, Value: `data-science`
   - Key: `framework`, Value: `scikit-learn`

> "The group name is the stable identifier. Deployment pipelines reference this name to find the latest approved model. Tags are organizational metadata -- they help you filter when you have dozens of model groups, and they support tag-based IAM access control from Module 1."

5. Click **Create model package group**. The group appears in the Registry list.

[PAUSE FOR Q&A - Ask: "Why should you avoid putting timestamps in the group name?" (Timestamps belong on individual versions, not the group. The group represents the logical model, not a specific training run.)]

---

### STEP 2 -- Registering a Model Version in the Console (15 minutes)

**Pacing: live demonstration, step by step.**

1. Click the `fraud-detection-rf` group to open it. The version list is empty.
2. Click **Register model version** (or **Create model package**).
3. **Inference specification:**

> "Two pieces of information connect this version to a deployable model: the container image and the artifact path."

   - **Container image:** Paste the ECR URI of the scikit-learn inference container. Show where to find it:
     - Navigate to **Training > Training jobs** in a new tab. Click the completed training job from Module 2.
     - Copy the **Training image** URI (e.g., `683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3`).
     - Paste it back in the registration form.

   - **Model data URL:** Paste the S3 path to the `model.tar.gz`:
     - On the same training job details page, scroll to **Output data configuration**.
     - Copy the S3 URI and append `/output/model.tar.gz` if needed.
     - Paste it in the form.

   - **Supported instance types:** Select `ml.m5.xlarge`.

> "This metadata tells SageMaker: 'To serve this model, use this container and download this artifact.' When we deploy in Stage 2, SageMaker reads this specification."

4. **Approval status:** Select **PendingManualApproval**.

> "We always start with PendingManualApproval. This prevents accidental deployment of an untested model. We will approve it after reviewing the criteria."

5. **Tags:**
   - Key: `training-job`, Value: (paste the training job name)
   - Key: `training-accuracy`, Value: (paste the accuracy you printed during Module 2)

6. Click **Create model package**. Version 1 appears in the group list.

> "Notice the version number is auto-incremented. When we register a second model later, it becomes version 2. The Registry maintains a clear version history."

---

### STEP 3 -- Registering via the SDK (5 minutes)

**Pacing: live demonstration in notebook.**

> "Let's see the same registration in code. If you still have the `estimator` object from Module 2, this is a one-liner."

```python
# STEP 3: Register from the Estimator (SDK shortcut)
model_package = estimator.register(
    model_package_group_name="fraud-detection-rf",
    content_types=["text/csv"],
    response_types=["text/csv"],
    inference_instances=["ml.m5.xlarge"],
    approval_status="PendingManualApproval",
)
print(f"Registered: {model_package.model_package_arn}")
```

> "The estimator already knows the artifact S3 path and the container image from the training job. `register()` wraps the `create_model_package` API call. If you refresh the console, you will see a new version appear."

**Instructor Note:** If the estimator object is no longer available (new session), demonstrate the `boto3` equivalent instead:

```python
# STEP 3 (alternative): Register with boto3
import boto3

sm_client = boto3.client("sagemaker")

sm_client.create_model_package(
    ModelPackageGroupName="fraud-detection-rf",
    InferenceSpecification={
        "Containers": [{
            "Image": "<container-image-uri>",
            "ModelDataUrl": "<s3-artifact-path>",
        }],
        "SupportedContentTypes": ["text/csv"],
        "SupportedResponseMIMETypes": ["text/csv"],
        "SupportedRealtimeInferenceInstanceTypes": ["ml.m5.xlarge"],
    },
    ModelApprovalStatus="PendingManualApproval",
)
```

---

### STEP 4 -- Designing Approval Criteria and Approving (10 minutes)

**Pacing: interactive discussion followed by console demonstration.**

> "Before we click Approve, let's define what 'approved' means for FraudShield. This is where the evaluation metrics from AIML Foundations become deployment decisions."

Write the criteria on screen:

| Criterion | Threshold | Rationale |
|-----------|-----------|-----------|
| F1 score | >= 0.85 | Below this, too many missed frauds or false alarms |
| Precision | >= 0.80 | False positives (blocking legitimate transactions) must be low |
| New version >= current | F1(new) >= F1(production) | Never deploy a regression |
| Training data validated | No nulls, expected schema | Prevents models trained on corrupted data |

> "Write these down before you approve any model. When a version arrives for review, you check it against the list. This removes subjective judgment and makes approvals consistent across the team."

**Now approve version 1 in the console:**

1. Navigate to **Governance > Model registry > fraud-detection-rf**.
2. Click version 1. Review the details page.
3. Click **Update status**.
4. Select **Approved**.
5. Add description: "Initial version. Training accuracy 0.XX. Baseline model for fraud detection."
6. Click **Update**.

> "The approval description is your audit trail. Three months from now, when someone asks why version 1 was approved, this description answers the question. Think of it as a meaningful Git commit message."

**SDK equivalent:**

```python
# STEP 4: Approve via SDK
sm_client.update_model_package(
    ModelPackageArn=model_package.model_package_arn,
    ModelApprovalStatus="Approved",
    ApprovalDescription="Initial baseline model. Training accuracy meets threshold.",
)
```

[PAUSE FOR Q&A - Ask: "If version 2 has higher F1 but lower precision than version 1, should you approve it?" (Depends on the use case. For fraud detection, precision matters because false positives block legitimate customers. This is the trade-off discussion from AIML Evaluation.)]

[PAUSE FOR BREAK - 10 MINS]

---

## Stage 2: Deploying to a Real-time Endpoint -- Console and SDK

**Duration:** 45 minutes
**Exit Criteria Addressed:**
- Deploy a trained model to a Real-time Inference Endpoint using the console (Required)
- Implement resource cleanup steps (deleting endpoints and models) to avoid unnecessary billing (Required)

### Instructor Opening (3 minutes)

> "Our model is registered and approved. Now we make it available to other teams. When a payment processor at FraudShield needs to check whether a transaction is fraudulent, they call our model's endpoint. The endpoint receives transaction data, runs it through the Random Forest, and returns a fraud/not-fraud prediction in milliseconds."

> "SageMaker deploys models through three objects: Model, Endpoint Configuration, and Endpoint. Think of it as: the recipe (Model), the kitchen layout (Configuration), and the open restaurant (Endpoint). We will create all three in the console first, then see the SDK shortcut."

Draw the three-object pattern:

```
Model (artifact + container)  -->  Endpoint Config (instance type, count)  -->  Endpoint (live service)
```

---

### STEP 5 -- Creating a Model Object in the Console (10 minutes)

**Pacing: live demonstration.**

1. Navigate to **Inference > Models**. Click **Create model**.
2. **Model name:** `fraud-rf-v1`
3. **IAM role:** Select the execution role.
4. **Container definition:**
   - **Container image:** Paste the same ECR URI used during registration.
   - **Model data URL:** Paste the S3 artifact path.

> "You can copy both values from the Model Registry version details page. The Registry is the single source of truth -- you do not need to remember S3 paths or container URIs."

5. Click **Create model**. The Model appears in the list.

> "No compute is running yet. We have just told SageMaker: 'Here is an artifact and a container that knows how to serve it.' The next step defines the infrastructure."

---

### STEP 6 -- Creating an Endpoint Configuration in the Console (7 minutes)

**Pacing: live demonstration.**

1. Navigate to **Inference > Endpoint configurations**. Click **Create endpoint configuration**.
2. **Configuration name:** `fraud-rf-v1-config`
3. Click **Add model**. Select `fraud-rf-v1`.
   - **Instance type:** `ml.m5.xlarge`
   - **Initial instance count:** `1`

> "`ml.m5.xlarge` keeps us within Free Tier limits. For production at FraudShield, you might scale to multiple instances for high availability, but for learning, one instance is sufficient."

4. **Data capture:** Skip for now (covered conceptually -- feeds Model Monitor in Module 4).
5. Click **Create endpoint configuration**. Still no compute running.

> "The configuration is a blueprint. You can create multiple endpoints from the same configuration, or create new configurations when you want to change instance types."

---

### STEP 7 -- Creating an Endpoint in the Console (10 minutes)

**Pacing: live demonstration. All Associates deploy simultaneously.**

1. Navigate to **Inference > Endpoints**. Click **Create endpoint**.
2. **Endpoint name:** `fraud-rf-v1-endpoint`
3. **Endpoint configuration:** Select `fraud-rf-v1-config`.
4. Click **Create endpoint**.

> "Now SageMaker provisions compute. An `ml.m5.xlarge` instance is being allocated. The container image is being pulled from ECR. The `model.tar.gz` is being downloaded from S3, extracted, and loaded into memory. When the status changes to InService, the model is live and accepting requests."

5. Wait for **InService** status (5-10 minutes). While waiting, discuss:

> "Every second this endpoint runs, it costs money. That is the trade-off of real-time inference: low latency but constant cost. Alternative modes trade latency for cost savings."

Show the comparison table:

| Mode | Latency | Cost Model | Best For |
|------|---------|-----------|----------|
| Real-time | Milliseconds | Per-instance-hour (always on) | Steady traffic, low latency |
| Batch Transform | Minutes | Per-job (instance-hours) | Large datasets, offline scoring |
| Serverless | Seconds (cold start) | Per-request | Infrequent, unpredictable traffic |
| Asynchronous | Seconds to minutes | Per-instance-hour | Large payloads |
| Multi-Model (MME) | Milliseconds (warm) | Per-instance-hour (shared) | Many models, cost savings |

> "For FraudShield's production system with steady transaction volume, real-time is correct. For a quarterly financial report that scores millions of records, batch transform would be better. For a prototype with sporadic traffic, serverless saves money."

6. When status shows **InService**, verify:
   - Click the endpoint name. Show the endpoint ARN, model, and instance details.
   - Note the CloudWatch metrics section (invocations, latency, errors -- all zero so far).

---

### STEP 8 -- SDK Deployment Shortcut (5 minutes)

**Pacing: demonstration in notebook (do NOT run -- the console endpoint is already live).**

> "The SDK can create all three objects in one line."

```python
# STEP 8: SDK deployment shortcut (shown, not executed -- we already have an endpoint)
predictor = estimator.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.xlarge",
    endpoint_name="fraud-rf-v1-endpoint-sdk",
)
```

> "This creates a Model, an Endpoint Configuration, and an Endpoint behind the scenes. It returns a `Predictor` object ready for invocation. Convenient, but understanding the three-object pattern helps you debug when something fails. If the endpoint is stuck on 'Creating,' you can check each object independently in the console."

---

### STEP 9 -- Understanding What Just Happened (10 minutes)

**Pacing: interactive console tour.**

> "Let's verify all three objects in the console."

1. **Inference > Models** -- Show `fraud-rf-v1` with its artifact and container.
2. **Inference > Endpoint configurations** -- Show `fraud-rf-v1-config` with instance type and model reference.
3. **Inference > Endpoints** -- Show `fraud-rf-v1-endpoint` with InService status.
4. Navigate to **Governance > Model registry > fraud-detection-rf > version 1**. Show that the artifact S3 path matches the Model object.

> "Three objects, one pipeline: the Registry holds the version metadata and approval status, the Model points to the artifact, the Config defines the infrastructure, and the Endpoint serves traffic. When you want to update the model in production, you create a new Model pointing to a new artifact, create a new Config, and update the Endpoint to use the new Config. The old version stays in the Registry as a record."

[PAUSE FOR BREAK - 10 MINS]

---

## Stage 3: Invoking the Endpoint, Debugging, and Mandatory Cleanup

**Duration:** 45 minutes
**Exit Criteria Addressed:**
- Invoke a SageMaker endpoint using the `boto3` client to generate predictions (Required)
- Implement resource cleanup steps (deleting endpoints and models) to avoid unnecessary billing (Required)

### STEP 10 -- Invoking with boto3 (12 minutes)

**Pacing: line-by-line in a notebook cell.**

> "The endpoint is live. Let's talk to it. We start with `boto3` -- the lowest-level AWS SDK -- so you see exactly what is happening over the wire."

```python
# STEP 10: Invoke with boto3
import boto3

runtime = boto3.client("sagemaker-runtime")

response = runtime.invoke_endpoint(
    EndpointName="fraud-rf-v1-endpoint",
    ContentType="text/csv",
    Body="500.0,3,25.0,5,0,0.4\n",
)

result = response["Body"].read().decode("utf-8")
print(f"Prediction: {result}")
```

> "Let's break this down. `EndpointName` tells SageMaker which endpoint to route to. `ContentType` tells the inference container how to parse the body -- we are sending CSV. `Body` is the raw data: six feature values matching our training data columns (amount, hour, distance_from_home, transaction_count_24h, is_international, merchant_risk_score). The response body contains the prediction."

Run the cell. Show the output.

> "If the prediction is `0`, the model says this transaction is not fraud. If `1`, it is fraud. Compare this to `model.predict(X_test)` from AIML Foundations -- same logic, but now the model runs on remote infrastructure and any application can call it over HTTPS."

**Multiple samples in one request:**

```python
# STEP 10b: Multiple samples
multi_response = runtime.invoke_endpoint(
    EndpointName="fraud-rf-v1-endpoint",
    ContentType="text/csv",
    Body="500.0,3,25.0,5,0,0.4\n900.0,2,80.0,1,1,0.9\n",
)

multi_result = multi_response["Body"].read().decode("utf-8")
print(f"Predictions: {multi_result}")
```

> "Each line in the CSV is one sample. The second sample has a high amount, early hour, and high merchant risk score -- our training data was designed to flag these as fraud. Check if the model agrees."

[PAUSE FOR Q&A - Ask: "What would happen if you sent seven values instead of six?" (The model expects six features. You would get a ModelError because the input shape does not match the training data shape.)]

---

### STEP 11 -- Invoking with the Predictor Class (8 minutes)

**Pacing: live demonstration in notebook.**

> "The `Predictor` class handles serialization automatically. No manual CSV formatting."

```python
# STEP 11: Invoke with Predictor
from sagemaker.predictor import Predictor
from sagemaker.serializers import CSVSerializer
from sagemaker.deserializers import CSVDeserializer

predictor = Predictor(
    endpoint_name="fraud-rf-v1-endpoint",
    serializer=CSVSerializer(),
    deserializer=CSVDeserializer(),
)

result = predictor.predict([500.0, 3, 25.0, 5, 0, 0.4])
print(f"Prediction: {result}")
```

> "`CSVSerializer` converts the Python list into a CSV string. `CSVDeserializer` converts the response back into a Python list. You pass a list, you get a list -- no manual string formatting."

**Mapping to local inference:**

| Local (AIML Foundations) | SageMaker Endpoint |
|--------------------------|-------------------|
| `model = joblib.load("model.pkl")` | Model loaded by the inference container |
| `X_test` as a NumPy array | Serialized as CSV in the request body |
| `model.predict(X_test)` | `predictor.predict(data)` |
| Return value as array | Deserialized by `CSVDeserializer` |

> "The logic is identical. The difference is a network layer in between that requires serialization on the way in and deserialization on the way out."

---

### STEP 12 -- Debugging Invocation Errors (10 minutes)

**Pacing: live demonstration. Trigger errors intentionally.**

> "Let's break things on purpose so you see the debugging workflow."

**Error 1: Wrong endpoint name.**

```python
# STEP 12a: Wrong endpoint name
try:
    runtime.invoke_endpoint(
        EndpointName="nonexistent-endpoint",
        ContentType="text/csv",
        Body="500.0,3,25.0,5,0,0.4\n",
    )
except Exception as e:
    print(f"Error: {e}")
```

> "You get a `ValidationError: Endpoint not found`. Fix: check the endpoint name in the console under **Inference > Endpoints**."

**Error 2: Wrong content type or malformed body.**

```python
# STEP 12b: Wrong data shape (7 features instead of 6)
try:
    response = runtime.invoke_endpoint(
        EndpointName="fraud-rf-v1-endpoint",
        ContentType="text/csv",
        Body="500.0,3,25.0,5,0,0.4,EXTRA\n",
    )
    result = response["Body"].read().decode("utf-8")
    print(result)
except Exception as e:
    print(f"Error: {e}")
```

> "A `ModelError` means the inference container could not process your input. The error traceback appears in CloudWatch Logs."

3. Show how to find CloudWatch Logs for the endpoint:
   - Search "CloudWatch" in the console.
   - Navigate to **Log groups > /aws/sagemaker/Endpoints/fraud-rf-v1-endpoint**.
   - Open the latest log stream. Show the error traceback.

> "CloudWatch Logs is your best friend for endpoint debugging. Every error, every print statement from the inference container appears here. Bookmark this pattern: error in notebook, traceback in CloudWatch."

| Error | Cause | Fix |
|-------|-------|-----|
| Endpoint not found | Wrong name or deleted endpoint | Check **Inference > Endpoints** |
| ModelError (400/500) | Wrong input format, shape mismatch | Check ContentType, column count, CloudWatch logs |
| AccessDeniedException | Missing `sagemaker:InvokeEndpoint` permission | Add permission to IAM role |
| Timeout | Model too large or instance too small | Check instance sizing |

---

### STEP 13 -- Mandatory Cleanup (10 minutes)

**Pacing: live demonstration. EVERY student must complete this step.**

> "This is the most important step of the lecture. If you skip this, your AWS bill accumulates for every second the endpoint runs. Let's clean up together."

**Console cleanup (preferred for visibility):**

1. Navigate to **Inference > Endpoints**. Select `fraud-rf-v1-endpoint`. Click **Actions > Delete**. Confirm.

> "Endpoint first -- this stops billing immediately."

2. Navigate to **Inference > Endpoint configurations**. Select `fraud-rf-v1-config`. Click **Actions > Delete**. Confirm.
3. Navigate to **Inference > Models**. Select `fraud-rf-v1`. Click **Actions > Delete**. Confirm.

> "Delete order: Endpoint, Configuration, Model. Always verify the endpoint is gone from the list."

**SDK cleanup (for reference):**

```python
# STEP 13: SDK cleanup
predictor.delete_endpoint()
predictor.delete_model()

# Or with boto3:
sm_client = boto3.client("sagemaker")
sm_client.delete_endpoint(EndpointName="fraud-rf-v1-endpoint")
sm_client.delete_endpoint_config(EndpointConfigName="fraud-rf-v1-config")
sm_client.delete_model(ModelName="fraud-rf-v1")
```

4. **Billing verification:** Search "Billing" in the console. Open **Billing & Cost Management**. Show current charges.

> "Make this a habit: deploy, test, clean up, check billing. Every time."

**Teaching Note:** Walk around the room (or monitor screen shares) to verify every student has deleted their endpoint. If any student still has an InService endpoint, stop and help them immediately.

[PAUSE FOR Q&A]

---

## Wrap-up & Git Branch Activity

**Duration:** 25 minutes

### Summary (5 minutes)

> "Today you completed the full deployment pipeline: registered a model in the Model Registry, defined approval criteria, approved the model, deployed it through the three-object pattern, invoked it with `boto3` and the `Predictor` class, debugged errors, and cleaned up. You now have hands-on experience with every step between a trained model artifact and a live prediction service."

> "In Module 4, you will automate this pipeline. Instead of clicking through the console, you will define SageMaker Pipelines that train, register, approve, and deploy models automatically. The Registry and endpoint patterns you learned today become steps in an automated workflow."

### Git Branch Activity (20 minutes)

> "For the remainder of this session, extend the deployment workflow you built today."

**Activity Instructions:**

1. In a Studio notebook or terminal:

```bash
cd ~/fraudshield-ml
git checkout -b feature/batch-invocation
```

2. Create a new notebook or script called `invoke_batch.py` that:
   - Reads `validation.csv` from the local filesystem.
   - Sends each row to the endpoint using `boto3` `invoke_endpoint` with CSV content type.
   - Collects all predictions into a list.
   - Computes accuracy by comparing predictions to the `target` column.
   - Prints a summary: total samples, correct predictions, accuracy.

3. (Stretch) Modify the script to use the `Predictor` class with `CSVSerializer` instead of manual `boto3` calls. Compare the code complexity.

4. Commit:

```bash
git add invoke_batch.py
git commit -m "Add batch invocation script with accuracy computation"
```

> "This exercise connects the invocation patterns you learned today back to the evaluation workflow from AIML Foundations. You are computing accuracy on a holdout set -- the same metric that informed your approval decision -- but now through a live endpoint instead of a local model."

**IMPORTANT:** If you deployed a new endpoint for this activity, delete it before leaving.

---

## Instructor Notes -- Common Issues

| Issue | Resolution |
|-------|-----------|
| Cannot find container image URI | Copy from the training job details page under "Training image." The inference container is the same as the training container for Script Mode. |
| Model Registry page is empty | Check the region. The Registry is region-specific. Ensure you are in the same region where you trained. |
| Endpoint stuck on "Creating" for 15+ minutes | First deployment in an account can be slow. If it exceeds 20 minutes, check CloudWatch Logs for the endpoint for errors. |
| `ModelError` on invocation | Most common cause is wrong number of features. Check the training data column count (excluding `target`). CSV body must have exactly that many values. |
| Student deploys but forgets to clean up | Walk over and help immediately. Check billing together. This is non-negotiable. |
| `AccessDeniedException` on invoke | The IAM role or user needs `sagemaker:InvokeEndpoint` permission. Add it via IAM console. |
| Endpoint deleted but config/model remain | Remind Associates to delete all three objects. Lingering configs and models do not cost money but create clutter. |
