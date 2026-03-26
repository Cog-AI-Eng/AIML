# AdvSageMaker-Inference Lecture - Instructor Guide

**Total Duration:** 180 Minutes (3 Stages)
**Consolidated Activities:** Serverless Inference (ASM-CT-ServerlessInference), Asynchronous Inference (ASM-CT-AsynchronousInference), Inference Decision Matrix (ASM-CT-InferenceDecisionMatrix), Batch Transform Architecture (ASM-CT-BatchTransformArchitecture), Multi-Model Endpoints (ASM-CT-MultiModelEndpoints), Multi-Container Endpoints (ASM-CT-MultiContainerEndpoints)

| Block | Content | Minutes |
|-------|---------|---------|
| Stage 1 | Serverless and Async Inference Deployments | 45 |
| Break 1 | Stretch / Questions | 10 |
| Stage 2 | Batch Transform and the Inference Decision Matrix | 45 |
| Break 2 | Stretch / Questions | 10 |
| Stage 3 | Multi-Model and Multi-Container Endpoints | 45 |
| Buffer | Open Q&A, Wrap-Up | 25 |

---

## Lecture Overview

**Unified Scenario -- FraudShield Risk Analytics (Advanced)**

The FraudShield ML team has moved beyond a single real-time endpoint. Business requirements now span multiple latency profiles: the payment gateway still needs sub-second fraud scores, but the compliance team wants nightly batch reports on historical transactions, the mobile app backend cannot justify always-on infrastructure for a feature used twice per hour, and the risk modeling team trains three competing algorithms each quarter and wants a single endpoint to serve whichever model a routing rule selects.

Today the team maps each of these business needs to the correct SageMaker inference pattern. Associates will deploy Serverless and Asynchronous endpoints to understand cold-start and cost tradeoffs, run a Batch Transform job on historical data, apply the Inference Decision Matrix to categorize real FraudShield scenarios, and consolidate multiple fraud models behind a Multi-Model Endpoint and a serial inference pipeline using Multi-Container Endpoints.

**Console + SDK:** Every deployment begins in the console so Associates see the configuration surface, then transitions to SDK code for automation. All deployments created in this lecture are deleted before leaving.

---

## Pre-Lecture Setup

### Instructor Checklist

- [ ] Completed foundational SageMaker modules (Studio, IAM, training jobs, real-time endpoints)
- [ ] At least one trained `model.tar.gz` artifact in S3 (e.g., `s3://<bucket>/fraudshield/output/<job-name>/output/model.tar.gz`)
- [ ] Two additional model artifacts prepared for the Multi-Model Endpoint exercise (XGBoost, Linear Learner) -- see Pre-Lecture Model Preparation below
- [ ] SageMaker Studio open with a notebook on an `ml.t3.medium` instance
- [ ] IAM execution role with S3, SageMaker, SNS, and CloudWatch permissions
- [ ] A historical transactions CSV file in S3 for Batch Transform (at least 1000 rows, no target column)
- [ ] SNS topic created for the Async inference notification exercise (`fraudshield-async-notifications`)
- [ ] Screen sharing enabled, font increased for projector readability
- [ ] This instructor guide open in a second tab

### Pre-Lecture Model Preparation

If you only have the Random Forest model from the foundational modules, create two additional models before class. Train a quick XGBoost and Linear Learner on the same FraudShield dataset using SageMaker built-in algorithms. Store all three `model.tar.gz` files under a common S3 prefix:

```
s3://<bucket>/fraudshield/multi-model/
    random-forest/model.tar.gz
    xgboost/model.tar.gz
    linear/model.tar.gz
```

### Student Prerequisites

- [ ] Completed foundational SageMaker lectures (environment, training, deployment, MLOps)
- [ ] Completed readings: Serverless Inference, Asynchronous Inference, Inference Decision Matrix, Batch Transform Architecture, Multi-Model Endpoints, Multi-Container Endpoints
- [ ] Studio notebook open on an `ml.t3.medium` instance
- [ ] S3 artifact path for at least one trained model
- [ ] Familiarity with the three-object deployment pattern (Model, Endpoint Configuration, Endpoint)

---

## Stage 1: Serverless and Async Inference Deployments

**Duration:** 45 minutes
**Exit Criteria Addressed:**
- Deploy a Serverless Inference endpoint and measure cold-start latency (Required)
- Deploy an Asynchronous Inference endpoint with SNS notification (Required)
- Compare cost and latency profiles across inference options (Required)

### Instructor Opening (5 minutes -- talk, no code)

> "In the foundational modules, you deployed a real-time endpoint. That endpoint was always running, always billing, and always ready. But think about FraudShield's mobile fraud-check feature -- it gets called maybe twice per hour during off-peak. Paying for a 24/7 `ml.m5.xlarge` instance to serve two requests per hour is like renting a commercial kitchen to microwave lunch."

> "SageMaker offers two alternatives for intermittent and long-running workloads: Serverless Inference and Asynchronous Inference. Today we deploy both and measure the tradeoffs firsthand."

---

### STEP 1 -- Deploying a Serverless Endpoint from the Console (12 minutes)

**Pacing: live demonstration.** All Associates follow along.

1. Open the SageMaker console. Navigate to **Inference > Models**.
2. Click **Create model**.
   - **Model name:** `fraud-rf-serverless`
   - **IAM role:** Select the existing execution role.
   - **Container definition:** Choose **Provide model artifacts and inference image location**.
     - **Inference image:** Paste the scikit-learn container URI from your training job.
     - **Model data URL:** Paste the S3 path to your Random Forest `model.tar.gz`.
3. Click **Create model**.

> "This is the same Model object from the three-object pattern. The difference comes in the Endpoint Configuration."

4. Navigate to **Inference > Endpoint configurations**. Click **Create endpoint configuration**.
   - **Configuration name:** `fraud-rf-serverless-config`
   - **Type:** Select **Serverless**.

> "Notice the configuration form changes. Instead of specifying an instance type, you specify memory size and max concurrency."

   - **Memory size:** `2048 MB`
   - **Max concurrency:** `5`

> "Memory size determines the compute allocated to each invocation. Max concurrency is how many simultaneous requests the endpoint handles before throttling. For FraudShield's off-peak mobile feature, 5 concurrent is generous."

   - **Production variants:** Click **Add model** and select `fraud-rf-serverless`. Set the model to the only variant.
5. Click **Create endpoint configuration**.

6. Navigate to **Inference > Endpoints**. Click **Create endpoint**.
   - **Endpoint name:** `fraud-rf-serverless-ep`
   - **Endpoint configuration:** Select `fraud-rf-serverless-config`.
7. Click **Create endpoint**. Wait for status to reach **InService**.

> "Serverless endpoints provision faster than real-time endpoints because no persistent instance is being launched. The infrastructure spins up only when a request arrives."

---

### STEP 2 -- Measuring Cold Start vs. Warm Latency (10 minutes)

**Pacing: live demonstration in notebook.**

```python
# STEP 2: Measure Serverless cold start vs. warm latency
import boto3
import time

runtime = boto3.client("sagemaker-runtime")

sample_payload = "500.0,3,25.0,5,0,0.4\n"

# First invocation (cold start)
start = time.time()
response = runtime.invoke_endpoint(
    EndpointName="fraud-rf-serverless-ep",
    ContentType="text/csv",
    Body=sample_payload,
)
cold_latency = (time.time() - start) * 1000
result = response["Body"].read().decode("utf-8")
print(f"Cold start latency: {cold_latency:.0f} ms | Prediction: {result.strip()}")

# Wait briefly, then invoke again (warm)
time.sleep(2)

start = time.time()
response = runtime.invoke_endpoint(
    EndpointName="fraud-rf-serverless-ep",
    ContentType="text/csv",
    Body=sample_payload,
)
warm_latency = (time.time() - start) * 1000
result = response["Body"].read().decode("utf-8")
print(f"Warm latency:       {warm_latency:.0f} ms | Prediction: {result.strip()}")

print(f"\nCold start overhead: {cold_latency - warm_latency:.0f} ms")
```

> "Cold start includes container initialization and model loading. Expect 3-10 seconds for a scikit-learn model. Warm invocations return in under a second. The tradeoff is clear: you pay zero when idle, but you pay a latency penalty for the first request after a period of inactivity."

[PAUSE FOR Q&A - Ask: "Would Serverless be appropriate for FraudShield's payment gateway that needs sub-200ms latency?" (No. Cold starts make Serverless unsuitable for strict latency SLAs. The payment gateway should use a real-time endpoint.)]

---

### STEP 3 -- Deploying an Asynchronous Endpoint from the Console (10 minutes)

**Pacing: live demonstration.**

1. Navigate to **Inference > Endpoint configurations**. Click **Create endpoint configuration**.
   - **Configuration name:** `fraud-rf-async-config`
   - **Type:** Select **Asynchronous**.
   - **Production variants:** Add the same `fraud-rf-serverless` model (or create a new Model object).
   - **Instance type:** `ml.m5.xlarge`
   - **Instance count:** `1`
   - **Async inference config:**
     - **S3 output path:** `s3://<bucket>/fraudshield/async-output/`
     - **SNS success topic:** Paste the ARN of `fraudshield-async-notifications`
     - **SNS error topic:** Same ARN (for simplicity)
     - **Max concurrent invocations per instance:** `4`

> "Async endpoints accept a request, return immediately with an output location, and process the request in the background. When processing finishes, the result is written to S3 and an SNS notification is sent. This pattern is ideal for large payloads or long-running inference."

2. Click **Create endpoint configuration**.
3. Navigate to **Inference > Endpoints**. Click **Create endpoint**.
   - **Endpoint name:** `fraud-rf-async-ep`
   - **Endpoint configuration:** Select `fraud-rf-async-config`.
4. Click **Create endpoint**. Wait for **InService**.

---

### STEP 4 -- Invoking the Async Endpoint and Checking Results (8 minutes)

**Pacing: live demonstration in notebook.**

```python
# STEP 4: Invoke Async endpoint
import json

runtime = boto3.client("sagemaker-runtime")

# Upload input data to S3
s3 = boto3.client("s3")
bucket = "<your-bucket>"
input_key = "fraudshield/async-input/batch-sample.csv"
input_data = "500.0,3,25.0,5,0,0.4\n900.0,2,80.0,1,1,0.9\n100.0,5,12.0,10,0,0.1\n"
s3.put_object(Bucket=bucket, Key=input_key, Body=input_data)

# Invoke asynchronously
response = runtime.invoke_endpoint_async(
    EndpointName="fraud-rf-async-ep",
    InputLocation=f"s3://{bucket}/{input_key}",
    ContentType="text/csv",
)

output_location = response["OutputLocation"]
print(f"Request accepted. Output will be at: {output_location}")
print("Check your SNS subscription (email) for notification.")
```

> "The call returned immediately. It did not wait for inference to complete. The `OutputLocation` tells you where results will appear in S3 once processing finishes. In a production system, your application subscribes to the SNS topic and processes results when notified."

```python
# STEP 4b: Poll for results (for classroom demonstration)
import urllib.parse

parsed = urllib.parse.urlparse(output_location)
output_bucket = parsed.netloc
output_key = parsed.path.lstrip("/")

for attempt in range(12):
    try:
        obj = s3.get_object(Bucket=output_bucket, Key=output_key)
        result = obj["Body"].read().decode("utf-8")
        print(f"Results ready after ~{(attempt + 1) * 5} seconds:")
        print(result)
        break
    except s3.exceptions.NoSuchKey:
        print(f"Attempt {attempt + 1}: Not ready yet, waiting 5 seconds...")
        time.sleep(5)
```

---

### STEP 5 -- Cost and Latency Comparison Table (3 minutes)

**Pacing: instructor-led discussion. Display this table.**

| Dimension | Real-time | Serverless | Asynchronous |
|-----------|-----------|------------|--------------|
| **Billing** | Per-second (always on) | Per-invocation (pay only when called) | Per-second (can scale to zero) |
| **Cold start** | None (always warm) | 3-10 seconds | None if instance running |
| **Latency** | Lowest (sub-second) | Variable (cold vs. warm) | Not applicable (background) |
| **Max payload** | 6 MB | 4 MB | 1 GB |
| **Best for** | Strict latency SLAs | Infrequent, bursty traffic | Large payloads, tolerant latency |

> "Every inference pattern is a tradeoff. There is no universally best option. The Inference Decision Matrix in Stage 2 gives you a systematic framework for choosing."

[PAUSE FOR BREAK - 10 MINS]

---

## Stage 2: Batch Transform and the Inference Decision Matrix

**Duration:** 45 minutes
**Exit Criteria Addressed:**
- Execute a Batch Transform job on historical data (Required)
- Apply the Inference Decision Matrix to categorize inference use cases (Required)
- Justify the selection of an inference pattern for a given scenario (Preferred)

### Instructor Opening (3 minutes)

> "Serverless and Async handle individual or small-batch requests. But FraudShield's compliance team needs fraud scores for every transaction from the last quarter -- three million rows. They do not need the results in real time. They need them by tomorrow morning. That is a Batch Transform job."

---

### STEP 6 -- Running a Batch Transform Job from the Console (15 minutes)

**Pacing: live demonstration.** All Associates follow along.

1. Navigate to **Inference > Batch transform**. Click **Create batch transform job**.
2. Fill in the details:
   - **Job name:** `fraud-rf-batch-historical`
   - **Model name:** Select the `fraud-rf-serverless` model (or any registered model).
   - **Instance type:** `ml.m5.xlarge`
   - **Instance count:** `1`

> "Batch Transform provisions instances, loads the model, processes every record in the input file, writes predictions to S3, and terminates the instances. You pay only for the processing time."

3. **Input data configuration:**
   - **S3 data type:** `S3Prefix`
   - **S3 URI:** `s3://<bucket>/fraudshield/batch-input/` (directory containing the historical CSV)
   - **Content type:** `text/csv`
   - **Split type:** `Line` (each line is one record)

> "Split type Line tells SageMaker to send one line at a time to the model. For CSV data, this is the standard approach."

4. **Output data configuration:**
   - **S3 output path:** `s3://<bucket>/fraudshield/batch-output/`

5. Click **Create job**. The job appears with status **InProgress**.

6. Monitor the job:
   - Click the job name. Show the **Status**, **Instance count**, **Duration**, and **Records processed** as they update.
   - Navigate to CloudWatch Logs to show the inference container logs.

> "The job processes the entire dataset without you maintaining an endpoint. When it finishes, predictions appear in the output S3 path as a file with the same name plus `.out` extension."

---

### STEP 7 -- Batch Transform via SDK (5 minutes)

**Pacing: live demonstration in notebook.**

```python
# STEP 7: Batch Transform via SDK
import sagemaker
from sagemaker.transformer import Transformer

session = sagemaker.Session()
role = sagemaker.get_execution_role()

transformer = Transformer(
    model_name="fraud-rf-serverless",
    instance_count=1,
    instance_type="ml.m5.xlarge",
    output_path=f"s3://{session.default_bucket()}/fraudshield/batch-output-sdk/",
    sagemaker_session=session,
)

transformer.transform(
    data=f"s3://{session.default_bucket()}/fraudshield/batch-input/",
    content_type="text/csv",
    split_type="Line",
    wait=False,
)

print(f"Batch Transform job started: {transformer.latest_transform_job.name}")
```

> "Setting `wait=False` returns immediately. The job runs in the background just like the console version. Use `transformer.wait()` if you want the cell to block until completion."

---

### STEP 8 -- The Inference Decision Matrix (15 minutes)

**Pacing: interactive discussion with whiteboard or shared screen.**

> "You now have four inference patterns in your toolkit. The Decision Matrix helps you choose without guessing."

Draw or display the matrix:

| Question | Real-time | Serverless | Async | Batch |
|----------|-----------|------------|-------|-------|
| Need sub-second latency? | Yes | Sometimes | No | No |
| Traffic pattern? | Sustained | Bursty/infrequent | Variable | Scheduled |
| Payload size? | < 6 MB | < 4 MB | < 1 GB | Unlimited (file) |
| Always-on cost acceptable? | Yes | No (pay per invocation) | Optional (scale to 0) | No (transient) |
| Result needed immediately? | Yes | Yes (after cold start) | No | No |
| Processing time per request? | < 60 seconds | < 60 seconds | < 15 minutes | N/A |

> "Start at the top. If the answer to 'Need sub-second latency?' is yes and traffic is sustained, you are in the real-time column. If traffic is bursty, consider Serverless. If the result is not needed immediately, Async or Batch. If you have a file of records to process offline, Batch Transform."

---

### STEP 9 -- Student Exercise: Categorize 5 FraudShield Scenarios (7 minutes)

**Pacing: individual or pair exercise. Associates write answers, then discuss.**

Present these five scenarios:

| # | Scenario | Pattern? |
|---|----------|----------|
| 1 | Payment gateway needs fraud score in < 200ms for every card swipe, 500 requests/second during peak | ? |
| 2 | Mobile app fraud check feature used 10 times per day, latency tolerance of 5 seconds | ? |
| 3 | Compliance team needs to score 3 million historical transactions by tomorrow morning | ? |
| 4 | Risk analysis pipeline uploads a 50 MB CSV of flagged transactions every 2 hours for scoring | ? |
| 5 | Internal dashboard refreshes fraud risk scores for 1000 merchants every 15 minutes | ? |

Give Associates 3 minutes to categorize, then reveal and discuss:

| # | Pattern | Reasoning |
|---|---------|-----------|
| 1 | **Real-time** | Strict latency, sustained high traffic, always-on cost justified |
| 2 | **Serverless** | Infrequent use, 5-second tolerance absorbs cold start, zero idle cost |
| 3 | **Batch Transform** | Offline bulk processing, no latency requirement, large dataset |
| 4 | **Async** | Large payload (50 MB exceeds real-time/serverless limits), scheduled, result not immediate |
| 5 | **Real-time** or **Async** | Depends on latency needs. If dashboard can wait 30 seconds, Async with polling. If instant refresh needed, Real-time with auto-scaling |

> "Scenario 5 has no single correct answer. Production decisions often require follow-up questions about latency tolerance and cost budget. The matrix narrows the options; business context makes the final call."

[PAUSE FOR Q&A]

[PAUSE FOR BREAK - 10 MINS]

---

## Stage 3: Multi-Model and Multi-Container Endpoints

**Duration:** 45 minutes
**Exit Criteria Addressed:**
- Deploy multiple models behind a Multi-Model Endpoint (Required)
- Invoke specific models via the TargetModel parameter (Required)
- Build a serial inference pipeline with Multi-Container Endpoints (Preferred)
- Clean up all deployed resources (Required)

### Instructor Opening (3 minutes)

> "FraudShield's risk modeling team trains three algorithms every quarter: XGBoost, Random Forest, and a Linear model. Each model serves a different segment -- XGBoost for high-value transactions, Random Forest for general retail, Linear for real-time scoring where speed matters most. Deploying three separate endpoints triples infrastructure cost. A Multi-Model Endpoint hosts all three behind a single endpoint, loading models on demand."

---

### STEP 10 -- Preparing Models for Multi-Model Endpoint (5 minutes)

**Pacing: live demonstration in notebook.**

```python
# STEP 10: Verify Multi-Model S3 structure
import boto3

s3 = boto3.client("s3")
bucket = "<your-bucket>"
prefix = "fraudshield/multi-model/"

response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
for obj in response.get("Contents", []):
    print(obj["Key"])
```

Expected output:

```
fraudshield/multi-model/random-forest/model.tar.gz
fraudshield/multi-model/xgboost/model.tar.gz
fraudshield/multi-model/linear/model.tar.gz
```

> "Each model is a separate `model.tar.gz` under the same prefix. The Multi-Model Endpoint dynamically loads the requested model into memory when invoked. Models not recently used are unloaded to free memory."

---

### STEP 11 -- Deploying a Multi-Model Endpoint from the Console (12 minutes)

**Pacing: live demonstration.**

1. Navigate to **Inference > Models**. Click **Create model**.
   - **Model name:** `fraud-multi-model`
   - **IAM role:** Select the existing execution role.
   - **Container definition:**
     - **Mode:** Select **MultiModel**.
     - **Inference image:** Use the scikit-learn container URI.
     - **Model data URL:** `s3://<bucket>/fraudshield/multi-model/` (the prefix, not a specific artifact).

> "Notice the mode is set to MultiModel. The model data URL points to a prefix, not a single file. SageMaker will look for `model.tar.gz` files under this prefix."

2. Click **Create model**.

3. Navigate to **Inference > Endpoint configurations**. Click **Create endpoint configuration**.
   - **Configuration name:** `fraud-multi-model-config`
   - **Type:** Real-time
   - **Instance type:** `ml.m5.xlarge`
   - **Instance count:** `1`
   - **Production variants:** Add `fraud-multi-model`.

4. Click **Create endpoint configuration**.

5. Navigate to **Inference > Endpoints**. Click **Create endpoint**.
   - **Endpoint name:** `fraud-multi-model-ep`
   - **Endpoint configuration:** Select `fraud-multi-model-config`.
6. Click **Create endpoint**. Wait for **InService**.

---

### STEP 12 -- Invoking Different Models via TargetModel (8 minutes)

**Pacing: live demonstration in notebook.**

```python
# STEP 12: Invoke different models on the Multi-Model Endpoint
runtime = boto3.client("sagemaker-runtime")

sample_payload = "500.0,3,25.0,5,0,0.4\n"

models = [
    "random-forest/model.tar.gz",
    "xgboost/model.tar.gz",
    "linear/model.tar.gz",
]

for model_path in models:
    start = time.time()
    response = runtime.invoke_endpoint(
        EndpointName="fraud-multi-model-ep",
        ContentType="text/csv",
        Body=sample_payload,
        TargetModel=model_path,
    )
    latency = (time.time() - start) * 1000
    result = response["Body"].read().decode("utf-8")
    print(f"Model: {model_path:<35} | Prediction: {result.strip():<6} | Latency: {latency:.0f} ms")
```

> "The `TargetModel` parameter tells the endpoint which model to load and invoke. The first call to each model incurs a loading penalty (similar to Serverless cold start). Subsequent calls to the same model are fast because it stays in memory."

> "In production, a routing layer upstream decides which model to call based on transaction attributes. High-value transactions route to XGBoost, retail transactions route to Random Forest. The endpoint does not care -- it loads whatever model you specify."

[PAUSE FOR Q&A - Ask: "What happens if you invoke a TargetModel path that does not exist in S3?" (SageMaker returns a ModelError. The endpoint remains healthy -- only that specific invocation fails.)]

---

### STEP 13 -- Multi-Container Serial Inference Pipeline (10 minutes)

**Pacing: SDK demonstration in notebook. Console setup is complex for multi-container, so SDK is primary.**

> "Multi-Container Endpoints serve a different purpose: chaining containers in a pipeline. FraudShield's feature engineering is complex enough that we want a dedicated preprocessing container before the prediction container. Request flows through Container 1 (preprocessor), then Container 2 (predictor) in sequence."

```python
# STEP 13: Multi-Container Endpoint (serial pipeline)
import sagemaker
from sagemaker.model import Model
from sagemaker.pipeline_model import PipelineModel

session = sagemaker.Session()
role = sagemaker.get_execution_role()

preprocessor = Model(
    image_uri="<sklearn-container-uri>",
    model_data=f"s3://{session.default_bucket()}/fraudshield/preprocessor/model.tar.gz",
    role=role,
    sagemaker_session=session,
    name="fraud-preprocessor",
)

predictor_model = Model(
    image_uri="<sklearn-container-uri>",
    model_data=f"s3://{session.default_bucket()}/fraudshield/multi-model/random-forest/model.tar.gz",
    role=role,
    sagemaker_session=session,
    name="fraud-predictor",
)

pipeline_model = PipelineModel(
    name="fraud-serial-pipeline",
    role=role,
    models=[preprocessor, predictor_model],
    sagemaker_session=session,
)

serial_predictor = pipeline_model.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.xlarge",
    endpoint_name="fraud-serial-pipeline-ep",
)

print("Serial inference pipeline deployed.")
```

> "The `models` list defines the order. Container 1 output becomes Container 2 input. This is a serial pipeline -- SageMaker calls them in sequence. You can chain up to 15 containers."

> "Use Multi-Container when preprocessing logic is complex enough to warrant its own container, or when the preprocessor and predictor use different frameworks (e.g., a Spark feature transformer followed by a PyTorch model)."

| Pattern | Purpose | Key Feature |
|---------|---------|-------------|
| Multi-Model | Multiple models, single endpoint | `TargetModel` selects model at invocation |
| Multi-Container | Model pipeline | Containers chained in serial order |
| Multi-Variant | A/B testing | Traffic split by percentage across variants |

---

### STEP 14 -- Mandatory Cleanup (7 minutes)

**Pacing: live demonstration. EVERY student must complete this step.**

> "This is the most critical step of the lecture. We deployed four endpoints today: Serverless, Async, Multi-Model, and the Serial Pipeline. Every one of them must be deleted. Let's clean up together."

**Console cleanup (preferred for visibility):**

1. Navigate to **Inference > Endpoints**. Delete each endpoint:
   - `fraud-rf-serverless-ep`
   - `fraud-rf-async-ep`
   - `fraud-multi-model-ep`
   - `fraud-serial-pipeline-ep`

> "Delete all endpoints first. This stops billing immediately."

2. Navigate to **Inference > Endpoint configurations**. Delete:
   - `fraud-rf-serverless-config`
   - `fraud-rf-async-config`
   - `fraud-multi-model-config`
   - (The serial pipeline config -- check the auto-generated name)

3. Navigate to **Inference > Models**. Delete:
   - `fraud-rf-serverless`
   - `fraud-multi-model`
   - `fraud-preprocessor`
   - `fraud-predictor`
   - `fraud-serial-pipeline`

4. Navigate to **Inference > Batch transform**. Verify the Batch Transform job has completed and is not still running.

**SDK cleanup (for reference):**

```python
# STEP 14: Cleanup all endpoints
sm_client = boto3.client("sagemaker")

endpoints_to_delete = [
    "fraud-rf-serverless-ep",
    "fraud-rf-async-ep",
    "fraud-multi-model-ep",
    "fraud-serial-pipeline-ep",
]

for ep in endpoints_to_delete:
    try:
        sm_client.delete_endpoint(EndpointName=ep)
        print(f"Deleted endpoint: {ep}")
    except Exception as e:
        print(f"Endpoint {ep}: {e}")

configs_to_delete = [
    "fraud-rf-serverless-config",
    "fraud-rf-async-config",
    "fraud-multi-model-config",
]

for cfg in configs_to_delete:
    try:
        sm_client.delete_endpoint_config(EndpointConfigName=cfg)
        print(f"Deleted config: {cfg}")
    except Exception as e:
        print(f"Config {cfg}: {e}")

models_to_delete = [
    "fraud-rf-serverless",
    "fraud-multi-model",
    "fraud-preprocessor",
    "fraud-predictor",
    "fraud-serial-pipeline",
]

for model in models_to_delete:
    try:
        sm_client.delete_model(ModelName=model)
        print(f"Deleted model: {model}")
    except Exception as e:
        print(f"Model {model}: {e}")

print("\nAll resources cleaned up.")
```

5. **Billing verification:** Search "Billing" in the console. Open **Billing and Cost Management**. Confirm no unexpected active resources.

> "Deploy, test, clean up, check billing. Every time. No exceptions."

**Teaching Note:** Walk around the room (or monitor screen shares) to verify every student has deleted all four endpoints. If any student still has an InService endpoint, stop and help them immediately.

[PAUSE FOR Q&A]

---

## Post-Lecture Wrap-Up

**Duration:** 25 minutes

### Summary (5 minutes)

> "Today you deployed four inference patterns: Serverless for intermittent traffic, Async for background processing with notifications, Batch Transform for offline bulk scoring, and Multi-Model for consolidating multiple algorithms behind a single endpoint. You also saw how Multi-Container Endpoints chain preprocessing and prediction in a serial pipeline."

> "The Inference Decision Matrix gives you a systematic framework: start with latency requirements, then consider traffic patterns, payload sizes, and cost tolerance. Every FraudShield scenario mapped to exactly one optimal pattern -- or sometimes two, requiring business context to finalize the decision."

> "In Module 5, you will add monitoring to these deployed endpoints. You will create baselines, enable data capture, detect drift, and build automated retraining loops. The inference patterns from today become the deployment targets that monitoring protects."

### Discussion Activity (20 minutes)

> "Pair up and design an inference architecture for the following FraudShield expansion scenario:"

**Scenario:** FraudShield is expanding to cover three new fraud types: identity theft, account takeover, and synthetic identity fraud. Each fraud type has its own model. The system must:
- Score payment transactions in real time (< 200ms)
- Score identity verification requests within 10 seconds
- Generate monthly compliance reports for all three fraud types across 10 million records
- Allow the risk team to swap models without redeploying

> "Sketch the architecture on paper or in a shared document. Identify which inference pattern you would use for each requirement and justify your choice. Be ready to share."

Allow 10 minutes for design, then 10 minutes for 2-3 groups to present and receive feedback.

---

## Instructor Notes -- Common Issues

| Issue | Resolution |
|-------|-----------|
| Serverless endpoint stuck on "Creating" | Serverless endpoints are fast to create (1-2 minutes). If stuck longer, check CloudWatch Logs for container errors. |
| Cold start exceeds 30 seconds | Large model files increase cold start. Verify model.tar.gz is not excessively large. Scikit-learn models should be < 100 MB. |
| Async output never appears in S3 | Check the SNS error notification for processing errors. Also verify the input file format matches what the model expects. |
| Batch Transform job fails immediately | Most common cause: input data has a header row. Batch Transform sends every line to the model, including headers. Remove the header or use `AssembleWith` and `SplitType` configuration. |
| Multi-Model TargetModel not found | The TargetModel path is relative to the model data URL prefix configured on the Model object. Verify the S3 structure matches. |
| Multi-Container deploy fails | Verify both container images are accessible and both model.tar.gz files exist. Check IAM role has access to both S3 paths. |
| Student forgets to delete endpoints | Walk over immediately. Check billing together. This is non-negotiable in every lecture. |
| `invoke_endpoint_async` returns 500 | Verify the input file exists at the S3 location. The endpoint fetches the file asynchronously -- a missing file causes a processing error. |
