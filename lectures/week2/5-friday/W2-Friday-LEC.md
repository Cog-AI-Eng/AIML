# Week 2 Friday -- Advanced SageMaker Inference Patterns + Model Monitoring

**Total Duration:** 185 Minutes (3 Stages)
**Consolidated Activities:**
- SM Inference: Inference Decision Matrix, Serverless Inference, Asynchronous Inference, Batch Transform Architecture, Multi-Model Endpoints, Multi-Container Endpoints
- SM Monitoring: Model Monitor Baselines, Data Quality Monitoring, Model Quality Monitoring, Bias & Attribution Drift, Drift Statistical Tests, Monitor & Pipeline Automation

| Block | Content | Minutes |
|-------|---------|---------|
| Stage 1 | Inference Patterns: Decision Matrix, Serverless, Async, Batch Transform | 55 |
| Break 1 | Stretch / Questions | 5 |
| Stage 2 | Model Monitoring Setup: Data Capture, Baselines, Quality Schedules | 55 |
| Break 2 | Stretch / Questions | 5 |
| Stage 3 | Drift Detection and Automation | 55 |
| Buffer | Open Q&A, Summary, W3 Monday Preview | 10 |

---

## Lecture Overview

**Unified Scenario -- FraudShield Risk Analytics**

Monday deployed a single real-time endpoint. But real-time is just one of five inference patterns. Today we explore the full inference decision matrix: serverless (scale to zero), async (large payloads), batch transform (bulk scoring), and multi-model endpoints (shared infrastructure). Then we add monitoring -- because a deployed model that degrades silently is worse than no model at all.

Associates continue as ML engineers at FraudShield. Thursday's best XGBoost model (from hyperparameter optimization) is the artifact we deploy with different inference patterns. Monday's real-time deployment was just the starting point -- now we see the full inference spectrum and learn to watch models after they go live.

1. **"Which inference pattern fits our workload?"** (Decision matrix: real-time, serverless, async, batch, multi-model)
2. **"How do we deploy the same model in different ways?"** (Serverless endpoints, async endpoints, batch transform)
3. **"How do we know when a deployed model starts degrading?"** (Data capture, baselines, monitoring schedules)
4. **"What happens when the data changes under the model?"** (Drift detection, statistical tests, automated response)

Each stage builds on the previous: deploy the model with different patterns, then monitor it, then detect when things go wrong.

---

## Pre-Lecture Setup

### Instructor Checklist

- [ ] Thursday's best XGBoost model artifact confirmed in S3 (`model.tar.gz` from HPO)
- [ ] Validation data (`validation.csv`) available in S3 from earlier sessions
- [ ] SageMaker execution role ARN ready
- [ ] Companion lecture notebook (`W2-Friday-notebook.ipynb`) open and tested
- [ ] AWS account with SageMaker access verified
- [ ] Budget verified and active
- [ ] S3 output paths configured for async inference and batch transform
- [ ] This instructor guide open in a second tab

### Student Prerequisites

- [ ] Completed readings: Inference Decision Matrix CT, Serverless Inference CT, Asynchronous Inference CT, Batch Transform Architecture CT, Multi-Model Endpoints CT, Multi-Container Endpoints CT, Model Monitor Baselines CT, Data Quality Monitoring CT, Model Quality Monitoring CT, Bias & Attribution Drift CT, Drift Statistical Tests CT, Monitor & Pipeline Automation CT
- [ ] Thursday's notebook completed (XGBoost model trained, best HPO artifact in S3)
- [ ] AWS credentials configured, SageMaker SDK installed
- [ ] Familiarity with real-time endpoint deployment from Monday

---

# STAGE 1 -- Inference Patterns (55 min)

> **Goal:** Understand the five SageMaker inference patterns, deploy Thursday's XGBoost model using serverless, async, and batch transform, and know when to use each pattern.

**Exit Criteria Addressed:**
- Compare the five SageMaker inference patterns across latency, traffic, payload, scaling, and cost dimensions (Required)
- Deploy a model to a serverless inference endpoint with appropriate memory and concurrency configuration (Required)
- Configure an asynchronous inference endpoint for large-payload workloads (Required)
- Execute a batch transform job to score data in bulk without a persistent endpoint (Required)
- Describe multi-model endpoints and when shared infrastructure is appropriate (Required)

### Instructor Opening (3 minutes -- talk, no code)

> "Monday you deployed a real-time endpoint. It worked. But what if your traffic is bursty and you do not want to pay for an always-on instance? What if your payloads are too large for the 6 MB real-time limit? What if you need to score 10 million records overnight? Real-time is one tool -- today you learn the full toolbox."

---

## STEP 1 -- Inference Decision Matrix (10 minutes)

**Pacing: conceptual with notebook markdown. Walk through the decision matrix table.**

Present the five inference patterns side by side:

| Dimension | Real-Time | Serverless | Async | Batch Transform | Multi-Model |
|-----------|-----------|------------|-------|-----------------|-------------|
| **Latency** | Low (ms) | Variable (cold start) | Minutes (queued) | Minutes-hours | Low (ms, after load) |
| **Traffic Pattern** | Steady, high | Bursty, unpredictable | Large/slow jobs | One-time or scheduled | Many models, moderate each |
| **Max Payload** | 6 MB | 4 MB | 1 GB | Unlimited (S3) | 6 MB |
| **Scale to Zero** | No | Yes | No | N/A (ephemeral) | No |
| **GPU Support** | Yes | No | Yes | Yes | Yes |
| **Billing** | Per-instance-hour | Per-ms of compute | Per-instance-hour | Per-instance-hour | Per-instance-hour (shared) |

> "This table is your cheat sheet. When a stakeholder says 'deploy this model,' your first question is not 'which instance type.' It is 'what does the traffic look like?' The answer determines the pattern."

Present the decision flowchart in text:

```
Is it a one-time bulk scoring job?
  YES --> Batch Transform
  NO  --> Is the traffic pattern bursty with long idle periods?
            YES --> Does the payload exceed 4 MB?
                      YES --> Async Inference
                      NO  --> Serverless Inference
            NO  --> Do you need to serve many models on shared infrastructure?
                      YES --> Multi-Model Endpoint
                      NO  --> Real-Time Endpoint
```

**Discussion Prompt:** "FraudShield processes card transactions. During business hours, traffic is steady at 500 TPS. Between 2 AM and 6 AM, traffic drops to near zero. Which pattern would you choose?" (Real-time during the day is fine, but the overnight lull suggests serverless or auto-scaling could save cost. Discuss trade-offs.)

---

## STEP 2 -- Create a Model Object (5 minutes)

**Pacing: live code in notebook.**

> "Before we deploy with different patterns, we need a SageMaker Model object. This is the same three-object pattern from Monday -- Model links the artifact to the container. The difference is what we do with it next."

Run the cell that creates a `sagemaker.model.Model` object using the XGBoost container image and Thursday's `model.tar.gz` artifact.

---

## STEP 3 -- Serverless Inference (12 minutes)

**Pacing: live code, step by step.**

> "Serverless inference is the 'Lambda for models.' You configure memory and max concurrency. SageMaker provisions compute on demand and scales to zero when idle. You pay only for the milliseconds your inference code runs."

Key configuration:
- `ServerlessInferenceConfig(memory_size_in_mb=2048, max_concurrency=5)`
- Memory sizes: 1024, 2048, 3072, 4096, 5120, 6144 MB
- Max concurrency: 1-200 (how many simultaneous invocations)

Deploy the model with serverless config. While deploying, explain:
- Cold start: first invocation after idle takes 1-2 seconds (container provisioning)
- Subsequent invocations: tens of milliseconds
- No GPU support -- CPU only
- 4 MB payload limit (vs 6 MB for real-time)

Once deployed, invoke the endpoint:

> "Notice the first invocation is slow -- that is the cold start. Run it again immediately and see the difference."

Run two invocations. Show the latency difference.

**Discussion Prompt:** "When would cold start latency be unacceptable?" (Real-time fraud scoring at checkout -- customers will not wait 2 seconds. Serverless is better for internal dashboards, batch-adjacent workflows, or dev/test environments.)

---

## STEP 4 -- Async Inference (10 minutes)

**Pacing: conceptual explanation then live code.**

> "Async inference handles payloads up to 1 GB and requests that take up to 15 minutes to process. The client sends the request, gets back a token, and polls for the result in S3. Think of it as a job queue for inference."

Key concepts:
- Request goes to an internal queue
- SageMaker processes it and writes the result to S3
- Client polls S3 (or gets an SNS notification)
- Supports auto-scaling, including scale-to-zero with a custom policy

Deploy with `AsyncInferenceConfig`:
- Output path in S3 for results
- Optional: SNS topics for success/failure notifications

> "We will not wait for the full async cycle in class. The important thing is that you understand when to use it: large payloads, long processing times, or when you want to decouple the request from the response."

---

## STEP 5 -- Batch Transform (10 minutes)

**Pacing: live code.**

> "Batch transform is not an endpoint at all. It is an ephemeral job: SageMaker spins up instances, reads your input from S3, runs inference on every record, writes the output to S3, and tears down the instances. No persistent infrastructure, no billing after the job ends."

Configure and run a batch transform on the validation data:
- Input: `s3://<bucket>/fraudshield/data/validation/`
- Output: `s3://<bucket>/fraudshield/batch-output/`
- Instance type: `ml.m5.xlarge`
- Strategy: `MultiRecord` (processes multiple records per request for throughput)

> "Batch transform is the right choice for monthly scoring runs, data backfills, or any scenario where you have all the data upfront and do not need real-time responses."

While the transform runs, discuss `join_source` (appending predictions to input) and `split_type` (how SageMaker chunks the input file).

---

## STEP 6 -- Multi-Model Endpoints: Conceptual Overview (5 minutes)

**Pacing: conceptual only, no deployment.**

> "What if FraudShield has 50 regional fraud models -- one per state? Deploying 50 endpoints is expensive. A multi-model endpoint hosts all 50 models on a single endpoint. SageMaker loads and unloads models dynamically based on traffic."

Key points:
- All models share the same container and instance
- SageMaker loads models on demand from S3
- Least-recently-used eviction when memory is full
- `TargetModel` parameter in the invoke call specifies which model
- Multi-container endpoints are different: multiple containers on one endpoint (e.g., preprocessing + inference)

> "We will not deploy a multi-model endpoint today because it requires multiple model artifacts. The concept is important for the exam and for production architectures."

**Discussion Prompt:** "What is the trade-off of multi-model endpoints vs separate endpoints?" (Shared infrastructure is cheaper but introduces latency for model loading and potential resource contention between models.)

---

## Cleanup Stage 1 Endpoints (3 minutes)

**Pacing: live code. Mandatory.**

> "Delete the serverless and async endpoints now. We will deploy a fresh real-time endpoint in Stage 2 with data capture enabled."

Delete: serverless endpoint, async endpoint. Verify both are gone.

[PAUSE FOR BREAK - 5 MINS]

---

# STAGE 2 -- Model Monitoring Setup (55 min)

> **Goal:** Deploy a real-time endpoint with data capture, create a baseline from training data, schedule monitoring, and understand data quality vs model quality monitoring.

**Exit Criteria Addressed:**
- Enable DataCaptureConfig on a real-time endpoint to log inference requests and responses (Required)
- Create a monitoring baseline from training data using DefaultModelMonitor (Required)
- Schedule a data quality monitoring job with appropriate frequency and constraints (Required)
- Compare data quality monitoring and model quality monitoring (Required)

### Instructor Opening (3 minutes -- talk, no code)

> "You now know five ways to deploy a model. But deployment is not the finish line. Fraud patterns change. Data distributions shift. A model that was 95% accurate last month might be 70% accurate today -- and you will not know unless you monitor it. SageMaker Model Monitor is the safety net."

---

## STEP 7 -- Why Monitoring Matters (5 minutes)

**Pacing: conceptual, notebook markdown.**

> "Imagine FraudShield deploys the XGBoost model in January. It performs well because the training data matches the live traffic. By March, a new type of fraud emerges -- cryptocurrency-related transactions with different feature distributions. The model has never seen these patterns. It starts misclassifying them as legitimate. Without monitoring, nobody notices until the quarterly review reveals millions in losses."

Key motivations:
- Data drift: input distributions change over time
- Concept drift: the relationship between features and target changes
- Model degradation: accuracy drops silently
- Regulatory requirements: some industries mandate ongoing model validation

---

## STEP 8 -- Model Monitor Architecture (5 minutes)

**Pacing: conceptual diagram in markdown.**

The monitoring pipeline has four components:

```
Live Endpoint
    |
    v
Data Capture (logs requests + responses to S3)
    |
    v
Baseline (statistical profile of training data)
    |
    v
Monitoring Schedule (periodic job compares captured data to baseline)
    |
    v
Violations Report (JSON listing which features violated constraints)
```

> "Data capture is the raw material. The baseline is the reference. The schedule is the engine. The violations report is the output. Today we set up all four."

---

## STEP 9 -- Deploy with Data Capture (10 minutes)

**Pacing: live code.**

> "We deploy a fresh real-time endpoint, but this time we enable DataCaptureConfig. This tells SageMaker to log every request and response to S3 in JSON Lines format."

Key configuration:
- `DataCaptureConfig(enable_capture=True, sampling_percentage=100, capture_options=["Input", "Output"])`
- `destination_s3_uri`: where captured data lands in S3
- Sampling percentage: 100% for learning, 10-20% typical in production

Deploy with the capture config. While deploying:

> "In production you would not capture 100% of traffic -- that creates too much data and cost. 10-20% sampling gives you a statistically representative sample. For learning purposes, we capture everything."

---

## STEP 10 -- Create a Baseline (10 minutes)

**Pacing: live code.**

> "The baseline is a statistical fingerprint of your training data. Model Monitor will compare incoming data against this fingerprint. If the incoming data looks significantly different, it raises a violation."

Use `DefaultModelMonitor` to create a baseline:
- Input: the training CSV (the same data the model was trained on)
- Output: a `statistics.json` (means, medians, distributions) and a `constraints.json` (thresholds for each feature)

```
monitor.suggest_baseline(
    baseline_dataset=training_data_s3_uri,
    dataset_format=DatasetFormat.csv(header=True),
    output_s3_uri=baseline_s3_uri,
)
```

> "The baselining job runs a SageMaker Processing job under the hood. It computes statistics for every feature: mean, standard deviation, min, max, quantiles, and the distribution type. It also generates constraints -- the acceptable ranges for each statistic."

While the baselining job runs (3-5 minutes), walk through what `statistics.json` and `constraints.json` contain:

| File | Contents | Purpose |
|------|----------|---------|
| `statistics.json` | Per-feature: mean, std, min, max, quantiles, distribution | Reference distribution |
| `constraints.json` | Per-feature: data type, completeness, allowed range | Violation thresholds |

---

## STEP 11 -- Schedule Monitoring (10 minutes)

**Pacing: live code.**

> "Now we set up a monitoring schedule. This is a recurring job that runs on a cron schedule, grabs the captured data since the last run, compares it to the baseline, and produces a violations report."

Create the schedule:
- Schedule expression: hourly for learning (production: daily or every 6 hours)
- Endpoint: the data-capture-enabled endpoint from Step 9
- Baseline: the statistics and constraints from Step 10

> "In practice, hourly monitoring is aggressive. Most teams run daily monitoring. We use hourly so you can see results within the lecture timeframe."

---

## STEP 12 -- Data Quality vs Model Quality Monitoring (5 minutes)

**Pacing: conceptual, notebook markdown.**

| Dimension | Data Quality Monitoring | Model Quality Monitoring |
|-----------|------------------------|------------------------|
| **What it monitors** | Feature distributions (inputs) | Prediction accuracy (outputs vs ground truth) |
| **Baseline source** | Training data statistics | Metrics from a baseline evaluation |
| **Detects** | Data drift, missing values, type changes | Accuracy degradation, precision/recall drop |
| **Ground truth needed?** | No | Yes (must be provided separately) |
| **SageMaker class** | `DefaultModelMonitor` | `ModelQualityMonitor` |

> "Data quality monitoring asks 'does the incoming data look like the training data?' Model quality monitoring asks 'is the model still making good predictions?' You need both, but data quality monitoring is simpler because it does not require ground truth labels."

**Discussion Prompt:** "For FraudShield, how would you get ground truth labels for model quality monitoring?" (Fraud investigations take weeks. Ground truth arrives with a delay. This means model quality monitoring lags behind data quality monitoring.)

---

## STEP 13 -- Generate Capture Data (7 minutes)

**Pacing: live code.**

> "The monitoring schedule needs captured data to analyze. Let us invoke the endpoint with sample data to generate some capture logs."

Send the validation data through the endpoint in batches. After invocations, verify that captured data appears in S3 under the capture destination.

> "Each captured record is a JSON Lines entry with the request payload, the response payload, a timestamp, and the inference ID. The monitoring job parses these to compute statistics on the incoming feature distributions."

[PAUSE FOR BREAK - 5 MINS]

---

# STAGE 3 -- Drift Detection and Automation (55 min)

> **Goal:** Understand data drift, simulate it, interpret monitoring violations, and learn how EventBridge automation closes the loop from detection to retraining.

**Exit Criteria Addressed:**
- Differentiate between data drift, concept drift, and prediction drift (Required)
- Simulate data drift by modifying feature distributions and observing monitoring violations (Required)
- Describe the statistical tests used for drift detection (KS test, chi-squared, L-infinity) (Required)
- Explain bias drift and feature attribution drift concepts (Required)
- Design an EventBridge-based automation pattern that triggers retraining when drift is detected (Required)

### Instructor Opening (3 minutes -- talk, no code)

> "The monitoring schedule is running. But monitoring is only useful if you can interpret the results and act on them. In this stage, we simulate what happens when the data changes -- a new fraud pattern, a seasonal shift, a data pipeline bug. We will see how Model Monitor flags the change and how EventBridge can automate the response."

---

## STEP 14 -- Types of Drift (8 minutes)

**Pacing: conceptual, notebook markdown.**

Three types of drift that degrade model performance:

| Drift Type | What Changes | Example | Detection Method |
|------------|-------------|---------|-----------------|
| **Data Drift** | Input feature distributions shift | Transaction amounts increase due to inflation | Compare feature statistics to baseline |
| **Concept Drift** | Relationship between features and target changes | A previously safe merchant category becomes high-risk | Monitor prediction accuracy over time |
| **Prediction Drift** | Model output distribution shifts | Model suddenly predicts more fraud than usual | Track prediction distribution |

> "Data drift is the canary in the coal mine. You can detect it without ground truth labels. Concept drift requires ground truth, which arrives with a delay. If you catch data drift early, you can investigate and retrain before concept drift causes real damage."

---

## STEP 15 -- Simulate Data Drift (12 minutes)

**Pacing: live code.**

> "Let us simulate a real-world scenario: three months after deployment, transaction amounts have increased by 40% due to inflation, and the merchant risk score distribution has shifted because of a new partner onboarding."

Modify the validation data:
- Increase `amount` by 40%: `amount * 1.4`
- Shift `merchant_risk_score` distribution: add 0.2 (clipped to [0, 1])
- Keep other features unchanged

Send the modified data through the endpoint.

> "We are now generating captured data that looks different from the training data. The monitoring job will compare these new distributions to the baseline and flag the deviations."

After invocations, explain what the monitoring job will find:
- `amount`: mean and distribution shifted significantly
- `merchant_risk_score`: distribution shifted
- Other features: within baseline constraints

**Discussion Prompt:** "If only one feature drifts, should you retrain? What if three features drift?" (Depends on feature importance. If `amount` is the top predictor and drifts significantly, retrain. If a low-importance feature drifts slightly, investigate but may not need immediate action.)

---

## STEP 16 -- Statistical Tests for Drift (8 minutes)

**Pacing: conceptual, notebook markdown.**

Model Monitor uses statistical tests to quantify drift:

| Test | Used For | What It Measures | Null Hypothesis |
|------|----------|-----------------|-----------------|
| **Kolmogorov-Smirnov (KS)** | Continuous features | Max distance between two CDFs | Distributions are identical |
| **Chi-Squared** | Categorical features | Divergence between observed and expected frequencies | No significant difference |
| **L-Infinity Norm** | Continuous or categorical | Max absolute difference in bin frequencies | Distributions are identical |

> "Model Monitor computes these tests for each feature and compares the test statistic to the threshold in `constraints.json`. If the statistic exceeds the threshold, the feature is flagged as a violation."

Example interpretation:
- KS statistic for `amount` = 0.42 (threshold: 0.10) --> VIOLATION
- KS statistic for `hour` = 0.03 (threshold: 0.10) --> within bounds

> "The violations report lists every feature that failed its statistical test, along with the test statistic and the threshold. This is the evidence you use to decide whether to retrain."

---

## STEP 17 -- Bias and Attribution Drift Concepts (7 minutes)

**Pacing: conceptual, no code.**

> "Beyond feature distributions, two advanced drift types matter for production ML: bias drift and feature attribution drift."

**Bias Drift:**
- Model starts treating demographic groups differently over time
- Example: fraud model becomes more aggressive toward transactions from certain geographic regions as data distribution shifts
- SageMaker Clarify monitors bias metrics over time
- Regulatory requirement in many financial services contexts

**Feature Attribution Drift:**
- The importance of features changes over time
- Example: `merchant_risk_score` was the top predictor at training time, but after drift, `amount` becomes more important
- SageMaker Clarify computes SHAP values on incoming data and compares to baseline attributions
- If feature importance shifts significantly, the model's decision logic may no longer match the current data

> "Clarify provides both bias monitoring and explainability monitoring as optional add-ons to Model Monitor. They require additional configuration and compute, but for regulated industries they are essential."

---

## STEP 18 -- EventBridge Automation Pattern (10 minutes)

**Pacing: conceptual with architecture diagram.**

> "Detection without action is just expensive logging. The real value of monitoring is automated response. EventBridge bridges the gap between 'drift detected' and 'do something about it.'"

Architecture:

```
Model Monitor Schedule
    |
    v
Violations Detected
    |
    v
CloudWatch Alarm (threshold on violation count)
    |
    v
EventBridge Rule (matches alarm state change)
    |
    v
Target: Lambda / Step Functions / SageMaker Pipeline
    |
    v
Automated Retraining Pipeline
    |
    v
New Model --> Registry --> Approval --> Deployment
```

Key concepts:
- Model Monitor emits CloudWatch metrics (e.g., violation count per feature)
- CloudWatch Alarm triggers when violations exceed threshold
- EventBridge rule matches the alarm and routes to a target
- Target can be:
  - Lambda function (lightweight: send Slack alert, trigger pipeline)
  - Step Functions (complex: orchestrate retrain + evaluate + deploy)
  - SageMaker Pipeline (direct: kick off a pre-built retraining DAG)

> "You built pipeline concepts on Monday. Now imagine that pipeline runs automatically whenever drift is detected. That is the end-to-end MLOps loop: deploy, monitor, detect, retrain, redeploy."

**Discussion Prompt:** "Should automated retraining deploy the new model automatically, or should it require human approval?" (Depends on risk tolerance. For FraudShield, financial impact is high -- require human approval for deployment but automate everything up to the approval gate.)

---

## STEP 19 -- Mandatory Cleanup (7 minutes)

**Pacing: live code. EVERY student must complete this.**

> "Delete everything we created today. Endpoints, monitoring schedules, models. Check billing. Make it a habit."

Cleanup order:
1. Stop and delete the monitoring schedule
2. Delete the real-time endpoint (with data capture)
3. Delete endpoint configurations
4. Delete model objects
5. Optionally clean up S3 capture data and batch output
6. Verify no endpoints remain in the console

> "Run the cleanup cell. Then open the AWS Console, navigate to Inference > Endpoints, and verify the list is empty. Check billing."

**Teaching Note:** Walk around the room (or monitor screen shares) to verify every student has deleted their resources. This is non-negotiable.

---

## Wrap-up & Q&A Buffer (10 minutes)

### Summary (5 minutes)

> "Today you accomplished three things. First, you learned the full inference spectrum: serverless endpoints scale to zero for bursty traffic, async endpoints handle large payloads, batch transform scores data in bulk without persistent infrastructure, and multi-model endpoints share a single instance across many models. You now have a decision matrix to match any workload to the right inference pattern. Second, you set up model monitoring end to end: data capture on the endpoint, a statistical baseline from training data, and a recurring monitoring schedule that flags violations. Third, you simulated data drift, saw how statistical tests detect it, and designed an EventBridge automation pattern that closes the loop from detection to retraining."

### W3 Monday Preview (2 minutes)

> "Next week shifts to advanced ML and genertic AI. Monday introduces foundation models, transfer learning at scale, and Amazon Bedrock. The question changes from 'how do we train and deploy our own models' to 'how do we leverage pre-trained foundation models for our tasks.' Read the Foundation Models and Bedrock CTs before Monday."

### Open Q&A (3 minutes)

---

## Instructor Notes -- Common Issues

| Issue | Resolution |
|-------|-----------|
| Thursday's XGBoost model artifact not in S3 | Use the fallback cell in the notebook that retrains a quick XGBoost model. Takes ~5 minutes. |
| Serverless endpoint cold start takes too long | Expected behavior (1-2 seconds). Explain this is the trade-off for scale-to-zero. |
| Async endpoint invoke returns no immediate result | By design -- check S3 output path for the result. Async is fire-and-forget. |
| Batch transform job fails with `ClientError` | Check that the input data format matches what the model expects (CSV, no header row for XGBoost). |
| `DataCaptureConfig` not capturing data | Verify `enable_capture=True` and `sampling_percentage > 0`. Check S3 destination path permissions. |
| Baselining job takes too long | Processing job provisioning can take 5-7 minutes. Normal for first run. |
| Monitoring schedule shows no violations | Invoke the endpoint with enough data first. Hourly schedule may not have run yet -- wait or trigger manually. |
| Student forgets to delete endpoints/schedules | Walk over immediately. Monitoring schedules incur costs even without endpoints. |
| `ValidationError` on serverless deploy | Check that the memory size is one of the allowed values (1024-6144 in 1024 increments). |
| Drift simulation does not trigger violations | Increase the magnitude of the shift, or lower the constraint thresholds in `constraints.json`. |
