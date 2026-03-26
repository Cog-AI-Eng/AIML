# AdvSageMaker-Monitoring Lecture - Instructor Guide

**Total Duration:** 180 Minutes (3 Stages)
**Consolidated Activities:** Model Monitor Baselines (ASM-CT-ModelMonitorBaselines), Data Quality Monitoring (ASM-CT-DataQualityMonitoring), Model Quality Monitoring (ASM-CT-ModelQualityMonitoring), Bias & Attribution Drift (ASM-CT-BiasAttributionDrift), Drift Statistical Tests (ASM-CT-DriftStatisticalTests), Monitor & Pipeline Automation (ASM-CT-MonitorPipelineAutomation)

| Block | Content | Minutes |
|-------|---------|---------|
| Stage 1 | Baseline Creation and Data Quality Monitoring | 45 |
| Break 1 | Stretch / Questions | 10 |
| Stage 2 | Model Quality and Bias Monitoring | 45 |
| Break 2 | Stretch / Questions | 10 |
| Stage 3 | Automated Retraining Pipeline from Monitoring Alerts | 45 |
| Buffer | Open Q&A, Wrap-Up | 25 |

---

## Lecture Overview

**Unified Scenario -- FraudShield Risk Analytics (Advanced)**

FraudShield's fraud detection model is deployed and serving production traffic. For the first three months, accuracy held steady at 94%. Then the compliance team noticed something troubling: the model was approving transactions that should have been flagged. Investigation revealed that FraudShield had expanded into a new market segment -- international wire transfers -- with significantly different feature distributions than the domestic card transactions the model was trained on. Average transaction amounts doubled, time-of-day patterns shifted across time zones, and merchant risk profiles changed entirely. The model never saw this data during training.

This is the canonical production ML failure: the world changed, but the model did not. Model Monitor exists to catch this before the compliance team does. Today the team builds a complete monitoring stack: baselines from training data, data quality monitoring to detect feature drift, model quality monitoring to track prediction accuracy over time, and bias monitoring to ensure protected attributes are not driving disparate outcomes.

In Stage 3, the team closes the loop by connecting monitoring alerts to an automated retraining pipeline. When Model Monitor detects drift, a CloudWatch Alarm fires, EventBridge routes the event to a Lambda function, and Lambda triggers the SageMaker Pipeline from the foundational MLOps module. The model retrains itself.

**Console + SDK:** Baseline creation and monitoring schedules begin in the console for visibility, then transition to SDK for automation. The automated retraining pipeline in Stage 3 is primarily a walkthrough with targeted console demonstrations.

---

## Pre-Lecture Setup

### Instructor Checklist

- [ ] A real-time endpoint deployed and InService (e.g., `fraud-rf-production-ep` on `ml.m5.xlarge`)
- [ ] Data capture enabled on the endpoint (or be prepared to enable it in Step 2)
- [ ] FraudShield training dataset available in S3 (`s3://<bucket>/fraudshield/data/train.csv`)
- [ ] A "drifted" synthetic dataset prepared for injection (see Drift Injection Data below)
- [ ] Ground truth labels file prepared for model quality monitoring (see Ground Truth Data below)
- [ ] SageMaker Studio open with a notebook on an `ml.t3.medium` instance
- [ ] IAM execution role with SageMaker, S3, CloudWatch, EventBridge, Lambda, and SNS permissions
- [ ] SageMaker Pipeline from the foundational MLOps module available (`fraudshield-detection-pipeline`)
- [ ] Screen sharing enabled, font increased for projector readability
- [ ] This instructor guide open in a second tab

### Drift Injection Data

Create a CSV file with deliberately shifted feature distributions. This simulates the international wire transfer scenario:

```python
import numpy as np
import pandas as pd

np.random.seed(99)
n = 200

drifted_data = pd.DataFrame({
    "amount": np.random.normal(2500, 800, n),         # shifted up from ~500
    "num_items": np.random.randint(1, 3, n),           # narrower range
    "hour_of_day": np.random.normal(3, 2, n).clip(0, 23).astype(int),  # shifted to early morning
    "customer_age_months": np.random.randint(1, 6, n), # newer accounts
    "is_international": np.ones(n, dtype=int),          # all international
    "merchant_risk_score": np.random.uniform(0.5, 1.0, n),  # higher risk
})

drifted_data.to_csv("drifted_transactions.csv", index=False, header=False)
```

Upload to S3: `s3://<bucket>/fraudshield/monitoring/drifted-input/drifted_transactions.csv`

### Ground Truth Data

For model quality monitoring, prepare a CSV mapping inference IDs to actual outcomes:

```csv
inference_id,label
inf-001,1
inf-002,0
inf-003,0
inf-004,1
```

Upload to S3: `s3://<bucket>/fraudshield/monitoring/ground-truth/`

### Student Prerequisites

- [ ] Completed foundational SageMaker modules and Module 4 (Inference)
- [ ] Completed readings: Model Monitor Baselines, Data Quality Monitoring, Model Quality Monitoring, Bias & Attribution Drift, Drift Statistical Tests, Monitor & Pipeline Automation
- [ ] Studio notebook open on an `ml.t3.medium` instance
- [ ] A deployed real-time endpoint (or willingness to deploy one at lecture start)

---

## Stage 1: Baseline Creation and Data Quality Monitoring

**Duration:** 45 minutes
**Exit Criteria Addressed:**
- Create a monitoring baseline from training data (Required)
- Enable data capture on a SageMaker endpoint (Required)
- Configure and run a Data Quality monitoring schedule (Required)
- Interpret violation reports (Required)

### Instructor Opening (5 minutes -- talk, no code)

> "In the foundational MLOps module, we discussed Model Monitor conceptually -- the four-step process of baseline, data capture, monitoring schedule, and alerts. Today we build every step for real. By the end of Stage 1, you will have a Data Quality monitor running on a schedule that detects when incoming transaction data no longer matches what the model was trained on."

> "Think of the baseline as the model's birth certificate. It records what the world looked like when the model was trained: feature means, standard deviations, distributions, data types. Everything afterward is compared against this baseline."

---

### STEP 1 -- Creating a Data Quality Baseline from the Console (12 minutes)

**Pacing: live demonstration.** All Associates follow along.

1. Open the SageMaker console. Navigate to **Inference > Model monitoring > Data quality**.

> "Data quality monitoring answers: 'Does the incoming data look like the training data?' If feature distributions shift, Model Monitor flags it."

2. Click **Create monitoring** (or **Suggest baseline**).
3. **Baseline job configuration:**
   - **Job name:** `fraud-dq-baseline`
   - **Baseline dataset:** `s3://<bucket>/fraudshield/data/train.csv`
   - **Dataset format:** CSV
   - **S3 output path:** `s3://<bucket>/fraudshield/monitoring/baseline-output/`
   - **IAM role:** Select the execution role.
   - **Instance type:** `ml.m5.xlarge`
   - **Instance count:** `1`

> "The baseline job reads your training data, computes statistics for every feature (mean, median, standard deviation, min, max, unique count, distribution type), and writes two files: `statistics.json` and `constraints.json`. These become the reference for all future monitoring."

4. Click **Create**. The baseline job starts.

> "This job takes 3-5 minutes for a typical dataset. While it runs, let's enable data capture on the endpoint."

---

### STEP 2 -- Enabling Data Capture on an Existing Endpoint (8 minutes)

**Pacing: SDK demonstration in notebook (data capture configuration requires endpoint update).**

> "Data capture tells SageMaker to log incoming requests and model responses to S3. This is the raw material that monitoring jobs compare against the baseline."

```python
# STEP 2: Enable data capture on the endpoint
import boto3
import sagemaker

sm_client = boto3.client("sagemaker")
session = sagemaker.Session()
bucket = session.default_bucket()

sm_client.update_endpoint(
    EndpointName="fraud-rf-production-ep",
    EndpointConfigName="fraud-rf-production-config-dc",
    # First, create a new endpoint config with data capture enabled
)
```

> "Updating data capture requires a new endpoint configuration. Let's create one."

```python
# STEP 2b: Create endpoint config with data capture
from sagemaker.model_monitor import DataCaptureConfig

data_capture_config = DataCaptureConfig(
    enable_capture=True,
    sampling_percentage=100,
    destination_s3_uri=f"s3://{bucket}/fraudshield/monitoring/data-capture/",
    capture_options=["Input", "Output"],
    csv_content_types=["text/csv"],
)

# Using the SageMaker Predictor to update (simpler approach)
from sagemaker.predictor import Predictor

predictor = Predictor(
    endpoint_name="fraud-rf-production-ep",
    sagemaker_session=session,
)

predictor.update_data_capture_config(data_capture_config=data_capture_config)
print("Data capture enabled. Requests will be logged to S3.")
```

> "We set sampling to 100% for this exercise so every request is captured. In production, 10-20% is typical -- enough to detect drift without excessive S3 costs."

**Verify in the console:**

1. Navigate to **Inference > Endpoints**. Click `fraud-rf-production-ep`.
2. Scroll to **Data capture** section. Confirm it shows **Enabled** with the S3 destination.

---

### STEP 3 -- Generating Traffic for Data Capture (5 minutes)

**Pacing: live demonstration in notebook.**

```python
# STEP 3: Send normal traffic to populate data capture
import time
from sagemaker.serializers import CSVSerializer
from sagemaker.deserializers import CSVDeserializer

predictor.serializer = CSVSerializer()
predictor.deserializer = CSVDeserializer()

normal_samples = [
    [500.0, 3, 14.0, 24, 0, 0.3],
    [120.0, 1, 10.0, 36, 0, 0.1],
    [800.0, 5, 22.0, 12, 0, 0.5],
    [50.0, 2, 9.0, 48, 0, 0.05],
    [1200.0, 1, 2.0, 3, 0, 0.8],
]

for i, sample in enumerate(normal_samples):
    result = predictor.predict(sample)
    print(f"Sample {i+1}: {sample} => {result}")
    time.sleep(1)

print("\nNormal traffic sent. Data capture files will appear in S3 within 5 minutes.")
```

> "Data capture writes to S3 in near real time, but there can be a short delay. The capture files are JSONL format, each line containing the request payload, response, timestamp, and inference ID."

---

### STEP 4 -- Reviewing the Baseline Output (5 minutes)

**Pacing: return to the console to check the baseline job.**

1. Navigate to **Inference > Model monitoring > Data quality**.
2. Click the `fraud-dq-baseline` job. Verify it shows **Completed**.
3. Click through to the output S3 location. Show the two key files:
   - `statistics.json` -- computed statistics for each feature
   - `constraints.json` -- inferred constraints (data types, non-null expectations)

```python
# STEP 4: Review baseline statistics
import json

s3 = boto3.client("s3")

stats_obj = s3.get_object(
    Bucket=bucket,
    Key="fraudshield/monitoring/baseline-output/statistics.json",
)
stats = json.loads(stats_obj["Body"].read().decode("utf-8"))

for feature in stats.get("features", []):
    name = feature.get("name", "unknown")
    mean = feature.get("numerical_statistics", {}).get("mean", "N/A")
    stddev = feature.get("numerical_statistics", {}).get("std_dev", "N/A")
    print(f"Feature: {name:<25} Mean: {str(mean):<10} StdDev: {str(stddev):<10}")
```

> "These statistics are the benchmark. When monitoring runs, it computes the same statistics on captured data and compares. If the mean of 'amount' jumps from 500 to 2500, that is a violation."

---

### STEP 5 -- Configuring a Data Quality Monitoring Schedule (8 minutes)

**Pacing: SDK demonstration in notebook.**

```python
# STEP 5: Create a Data Quality monitoring schedule
from sagemaker.model_monitor import DefaultModelMonitor, CronExpressionGenerator

monitor = DefaultModelMonitor(
    role=sagemaker.get_execution_role(),
    instance_count=1,
    instance_type="ml.m5.xlarge",
    volume_size_in_gb=10,
)

monitor.create_monitoring_schedule(
    monitor_schedule_name="fraud-dq-schedule",
    endpoint_input="fraud-rf-production-ep",
    output_s3_uri=f"s3://{bucket}/fraudshield/monitoring/dq-results/",
    statistics=f"s3://{bucket}/fraudshield/monitoring/baseline-output/statistics.json",
    constraints=f"s3://{bucket}/fraudshield/monitoring/baseline-output/constraints.json",
    schedule_cron_expression=CronExpressionGenerator.hourly(),
)

print("Data Quality monitoring schedule created (hourly).")
```

> "The schedule runs every hour. It pulls captured data from the last hour, computes statistics, compares against the baseline, and generates a violations report if drift is detected."

**Verify in the console:**

1. Navigate to **Inference > Model monitoring > Monitoring schedules**.
2. Show `fraud-dq-schedule` in the list with status **Scheduled**.

---

### STEP 6 -- Injecting Drift and Observing Violations (7 minutes)

**Pacing: live demonstration in notebook.**

> "Let's simulate the international wire transfer scenario. We send traffic with deliberately shifted features and trigger a monitoring execution."

```python
# STEP 6: Inject drifted traffic
import pandas as pd

drifted = pd.read_csv("drifted_transactions.csv", header=None)
print(f"Sending {len(drifted)} drifted samples...")

for idx, row in drifted.head(20).iterrows():
    try:
        result = predictor.predict(row.tolist())
        if idx < 3:
            print(f"Drifted sample {idx}: {row.tolist()} => {result}")
    except Exception as e:
        print(f"Error on sample {idx}: {e}")
    time.sleep(0.5)

print("\nDrifted traffic injected. Trigger a manual monitoring execution or wait for the schedule.")
```

```python
# STEP 6b: Trigger a manual monitoring execution (instead of waiting for hourly schedule)
monitor.create_monitoring_schedule(
    monitor_schedule_name="fraud-dq-manual-run",
    endpoint_input="fraud-rf-production-ep",
    output_s3_uri=f"s3://{bucket}/fraudshield/monitoring/dq-manual-results/",
    statistics=f"s3://{bucket}/fraudshield/monitoring/baseline-output/statistics.json",
    constraints=f"s3://{bucket}/fraudshield/monitoring/baseline-output/constraints.json",
    schedule_cron_expression="cron(0/1 * ? * * *)",  # every minute for demo
)
```

> "In practice, you would wait for the hourly schedule. For the classroom, we trigger it manually or set a very short interval. The violation report appears in S3 within minutes."

**Show the violation report structure:**

```python
# STEP 6c: Read violation report (after monitoring execution completes)
# Navigate in the console: Model monitoring > Monitoring schedules > click schedule > Executions
# Or read from S3:
violation_report = {
    "violations": [
        {
            "feature_name": "amount",
            "constraint_check_type": "baseline_drift_check",
            "description": "Numerical feature amount has baseline mean 487.32 but observed mean 2534.17"
        },
        {
            "feature_name": "hour_of_day",
            "constraint_check_type": "baseline_drift_check",
            "description": "Numerical feature hour_of_day has baseline mean 12.5 but observed mean 3.2"
        }
    ]
}

print("Example violation report:")
for v in violation_report["violations"]:
    print(f"  Feature: {v['feature_name']}")
    print(f"  Check:   {v['constraint_check_type']}")
    print(f"  Detail:  {v['description']}")
    print()
```

> "The violation report names the drifted features and quantifies the shift. The 'amount' feature jumped from a mean of 487 to 2534. The 'hour_of_day' shifted from midday to early morning. These are exactly the international wire transfer patterns we injected."

[PAUSE FOR Q&A - Ask: "If only one feature drifts but the model's accuracy remains unchanged, should you retrain?" (Not necessarily. Data drift is a warning signal, not a definitive problem. Check model quality metrics before retraining. A single drifted feature may not affect predictions if the model does not rely heavily on it.)]

[PAUSE FOR BREAK - 10 MINS]

---

## Stage 2: Model Quality and Bias Monitoring

**Duration:** 45 minutes
**Exit Criteria Addressed:**
- Create a model quality baseline with evaluation metrics (Required)
- Configure model quality monitoring with ground truth ingestion (Required)
- Set up bias monitoring with SageMaker Clarify (Preferred)
- Interpret K-S test and Chi-Squared test results (Preferred)

### Instructor Opening (3 minutes)

> "Data quality monitoring tells you that the input data has changed. Model quality monitoring tells you that the model's predictions have degraded. These are complementary signals. Data drift can occur without accuracy loss (the model is robust). Accuracy can degrade without data drift (concept drift -- the relationship between features and labels changed). You need both monitors."

---

### STEP 7 -- Creating a Model Quality Baseline (10 minutes)

**Pacing: SDK demonstration in notebook.**

> "A model quality baseline establishes the expected prediction performance. We provide the model's evaluation metrics from training time -- the same metrics we computed in the foundational evaluation module."

```python
# STEP 7: Create a model quality baseline
from sagemaker.model_monitor import ModelQualityMonitor

mq_monitor = ModelQualityMonitor(
    role=sagemaker.get_execution_role(),
    instance_count=1,
    instance_type="ml.m5.xlarge",
    volume_size_in_gb=10,
    sagemaker_session=session,
)

mq_monitor.suggest_baseline(
    baseline_dataset=f"s3://{bucket}/fraudshield/monitoring/ground-truth/baseline-with-predictions.csv",
    dataset_format=sagemaker.model_monitor.DatasetFormat.csv(header=True),
    output_s3_uri=f"s3://{bucket}/fraudshield/monitoring/mq-baseline-output/",
    problem_type="BinaryClassification",
    inference_attribute="prediction",
    ground_truth_attribute="label",
    probability_attribute="probability",
)

print("Model quality baseline job started.")
```

> "The baseline dataset contains predictions alongside ground truth labels. Model Monitor computes accuracy, F1, precision, recall, and AUC. These become the reference metrics. When production accuracy drops below the baseline, a violation fires."

**Verify in the console:**

1. Navigate to **Inference > Model monitoring > Model quality**.
2. Show the baseline job running or completed.
3. Click through to the output to see the baseline metrics.

---

### STEP 8 -- Configuring Model Quality Monitoring (10 minutes)

**Pacing: SDK demonstration in notebook.**

> "Model quality monitoring requires ground truth. Unlike data quality (which only looks at inputs), model quality needs to know the correct answer. In FraudShield's case, ground truth arrives when the compliance team confirms whether a transaction was actually fraudulent -- often days or weeks after the initial prediction."

```python
# STEP 8: Create model quality monitoring schedule
mq_monitor.create_monitoring_schedule(
    monitor_schedule_name="fraud-mq-schedule",
    endpoint_input="fraud-rf-production-ep",
    output_s3_uri=f"s3://{bucket}/fraudshield/monitoring/mq-results/",
    problem_type="BinaryClassification",
    ground_truth_input=f"s3://{bucket}/fraudshield/monitoring/ground-truth/",
    constraints=f"s3://{bucket}/fraudshield/monitoring/mq-baseline-output/constraints.json",
    schedule_cron_expression=CronExpressionGenerator.daily(),
)

print("Model quality monitoring schedule created (daily).")
```

> "The schedule runs daily. It merges captured predictions with ground truth labels uploaded to the ground truth S3 path. The merge key is the inference ID -- each prediction has a unique ID from data capture, and the ground truth file maps those IDs to actual labels."

**Ground truth ingestion flow:**

```
Endpoint prediction (with inference ID)
    |
    v
Data Capture logs to S3 (inference_id, prediction, timestamp)
    |                                                    |
    v                                                    v
Model Quality Monitor  <--  Ground Truth Upload (inference_id, label)
    |
    v
Compare accuracy to baseline
```

> "The lag between prediction and ground truth upload is expected. FraudShield's compliance team confirms fraud cases within 48 hours. The daily monitoring schedule accommodates this lag."

[PAUSE FOR Q&A - Ask: "What happens if ground truth is not available for some predictions?" (Model Monitor computes metrics on the subset that has ground truth. Incomplete coverage is normal in production. If coverage is too low, the metrics may not be statistically significant.)]

---

### STEP 9 -- Bias Monitoring with SageMaker Clarify (12 minutes)

**Pacing: SDK demonstration with console exploration.**

> "Bias monitoring answers a different question: 'Is the model treating protected groups differently?' FraudShield's legal team wants assurance that the fraud model does not disproportionately flag transactions from specific demographic groups."

```python
# STEP 9: Configure bias monitoring with Clarify
from sagemaker.clarify import (
    BiasConfig,
    DataConfig,
    ModelConfig,
    ModelPredictedLabelConfig,
)
from sagemaker.model_monitor import ClarifyModelMonitor

clarify_monitor = ClarifyModelMonitor(
    role=sagemaker.get_execution_role(),
    instance_count=1,
    instance_type="ml.m5.xlarge",
    sagemaker_session=session,
)

bias_config = BiasConfig(
    label_values_or_threshold=[1],
    facet_name="is_international",
    facet_values_or_threshold=[1],
)

# Create bias baseline
clarify_monitor.suggest_baseline(
    data_config=DataConfig(
        s3_data_input_path=f"s3://{bucket}/fraudshield/data/train.csv",
        s3_output_path=f"s3://{bucket}/fraudshield/monitoring/bias-baseline-output/",
        dataset_type="text/csv",
        label="target",
    ),
    bias_config=bias_config,
    model_config=ModelConfig(
        model_name="fraud-rf-serverless",
        instance_type="ml.m5.xlarge",
        instance_count=1,
    ),
    model_predicted_label_config=ModelPredictedLabelConfig(probability_threshold=0.5),
)

print("Bias baseline job started.")
```

> "The `facet_name` is the protected attribute we are monitoring. Here we check whether international transactions (`is_international=1`) receive disproportionately different fraud scores than domestic transactions. The baseline computes pre-training and post-training bias metrics like Demographic Parity Difference and Disparate Impact."

> "Clarify computes metrics such as DPL (Difference in Positive Proportions in Labels) and DI (Disparate Impact). DPL measures whether one group is labeled positive more often. DI measures the ratio of positive prediction rates between groups. A DI of 1.0 means equal treatment; values far from 1.0 indicate potential bias."

---

### STEP 10 -- Statistical Tests for Drift Detection (10 minutes)

**Pacing: interactive discussion with whiteboard.**

> "Model Monitor uses statistical tests to decide whether drift is real or just noise. Two tests dominate: the Kolmogorov-Smirnov (K-S) test for numerical features and the Chi-Squared test for categorical features."

**K-S Test (Numerical Features):**

> "The K-S test compares two distributions by measuring the maximum distance between their cumulative distribution functions. If the distance exceeds a threshold, the test rejects the null hypothesis that the distributions are the same."

Draw two CDFs on the whiteboard -- one baseline, one drifted. Mark the maximum vertical distance.

| Property | K-S Test |
|----------|----------|
| **Input** | Two samples of numerical values |
| **Null hypothesis** | Samples come from the same distribution |
| **Output** | Test statistic (0 to 1) and p-value |
| **Reject when** | p-value < significance level (typically 0.05) |
| **FraudShield example** | Baseline `amount` mean=487, observed `amount` mean=2534 |

**Chi-Squared Test (Categorical Features):**

> "The Chi-Squared test compares observed frequencies of categories against expected frequencies from the baseline. If a category that appeared 5% of the time in training now appears 40% of the time, the test detects it."

| Property | Chi-Squared Test |
|----------|------------------|
| **Input** | Observed and expected category frequencies |
| **Null hypothesis** | Observed frequencies match expected |
| **Output** | Test statistic and p-value |
| **Reject when** | p-value < significance level |
| **FraudShield example** | Baseline `is_international` ratio=10%, observed=95% |

> "Model Monitor applies these tests automatically. You configure the significance level (default 0.05). The violation report includes the test statistic and p-value for each feature that drifted. Understanding these tests helps you interpret whether a violation is a genuine shift or a statistical fluke from a small sample."

[PAUSE FOR Q&A - Ask: "If your data capture only has 10 samples, can the K-S test reliably detect drift?" (No. Small sample sizes reduce statistical power. The test may miss real drift or produce false positives. Ensure sufficient data capture volume before relying on statistical test results.)]

[PAUSE FOR BREAK - 10 MINS]

---

## Stage 3: Automated Retraining Pipeline from Monitoring Alerts

**Duration:** 45 minutes
**Exit Criteria Addressed:**
- Create a CloudWatch Alarm on Model Monitor violations (Required)
- Configure an EventBridge rule to route monitoring events (Required)
- Connect monitoring alerts to automated pipeline execution (Required)
- Describe the closed-loop MLOps architecture (Required)

### Instructor Opening (3 minutes)

> "You now have two monitors running: data quality and model quality. When drift occurs, violation reports appear in S3. But nobody is watching S3 at 3 AM. The next step is automation: monitoring alerts trigger retraining without human intervention."

> "This is the difference between Level 1 and Level 2 MLOps. Level 1 automates training pipelines but triggers them manually. Level 2 automates the trigger itself -- monitoring closes the loop."

---

### STEP 11 -- Creating a CloudWatch Alarm on Monitor Violations (10 minutes)

**Pacing: console demonstration.** All Associates follow along.

1. Open the CloudWatch console. Navigate to **Alarms > All alarms**. Click **Create alarm**.
2. Click **Select metric**.
3. Navigate to **SageMaker > Monitoring Schedule Metrics** (or search for the metric namespace `aws/sagemaker/Endpoints/data-metrics`).

> "Model Monitor publishes metrics to CloudWatch after each monitoring execution. The key metric is the number of violations detected."

4. Select the metric for your monitoring schedule. If the metric is not yet available (no monitoring execution has completed), demonstrate the configuration:
   - **Metric name:** `data_quality_violations`
   - **Statistic:** `Maximum`
   - **Period:** `1 hour`

5. **Conditions:**
   - **Threshold type:** Static
   - **Whenever data_quality_violations is:** Greater than 0
   - **Threshold value:** `0`

> "Any violation count greater than zero triggers the alarm. In production, you might set a higher threshold to avoid reacting to minor fluctuations -- for example, alarm only when violations exceed 3 features."

6. **Actions:**
   - **Alarm state trigger:** In alarm
   - **SNS topic:** Select or create `fraudshield-monitoring-alerts`

7. **Name:** `fraud-dq-violation-alarm`
8. Click **Create alarm**.

> "When the alarm fires, it sends a notification to SNS. But we want more than a notification -- we want to trigger the retraining pipeline. That is where EventBridge comes in."

---

### STEP 12 -- Configuring EventBridge to Route Monitoring Events (10 minutes)

**Pacing: console demonstration.**

1. Open the EventBridge console. Navigate to **Rules**. Click **Create rule**.
2. **Rule details:**
   - **Name:** `fraud-monitor-retrain-trigger`
   - **Event bus:** Default
   - **Rule type:** Rule with an event pattern

3. **Event pattern:**

> "SageMaker Model Monitor emits events when monitoring executions complete. We match on events that indicate violations were found."

```json
{
  "source": ["aws.sagemaker"],
  "detail-type": ["SageMaker Model Monitor Alert"],
  "detail": {
    "MonitoringScheduleName": ["fraud-dq-schedule"],
    "CurrentStatus": ["CompletedWithViolations"]
  }
}
```

> "This pattern matches only events from our specific monitoring schedule where violations were detected. Events with no violations are ignored."

4. **Target:**
   - **Target type:** AWS service
   - **Service:** Lambda function
   - **Function:** `fraud-retrain-trigger` (we will create this next)

5. Click **Create rule** (or save as draft if Lambda is not yet ready).

---

### STEP 13 -- Lambda Function to Trigger SageMaker Pipeline (12 minutes)

**Pacing: console demonstration with code walkthrough.**

1. Open the Lambda console. Click **Create function**.
   - **Function name:** `fraud-retrain-trigger`
   - **Runtime:** Python 3.12
   - **Execution role:** Create a new role with SageMaker permissions (or use an existing role with `sagemaker:StartPipelineExecution`)

2. Replace the default handler with:

```python
import json
import boto3
from datetime import datetime


def lambda_handler(event, context):
    sm_client = boto3.client("sagemaker")
    pipeline_name = "fraudshield-detection-pipeline"
    
    execution_id = f"monitor-triggered-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    response = sm_client.start_pipeline_execution(
        PipelineName=pipeline_name,
        PipelineExecutionDisplayName=execution_id,
        PipelineParameters=[
            {"Name": "ApprovalStatus", "Value": "PendingManualApproval"},
        ],
    )
    
    pipeline_arn = response["PipelineExecutionArn"]
    
    print(f"Monitor alert received. Pipeline execution started: {pipeline_arn}")
    print(f"Event detail: {json.dumps(event.get('detail', {}))}")
    
    return {
        "statusCode": 200,
        "body": json.dumps({
            "message": "Retraining pipeline triggered",
            "execution_arn": pipeline_arn,
        }),
    }
```

> "The Lambda function is simple: receive the event, start the pipeline, log the result. The pipeline name matches the one we built in the foundational MLOps module. We set ApprovalStatus to PendingManualApproval so a human still reviews the retrained model before deployment."

3. Click **Deploy** to save the function.

4. **Test the function** with a sample event:

```json
{
  "source": "aws.sagemaker",
  "detail-type": "SageMaker Model Monitor Alert",
  "detail": {
    "MonitoringScheduleName": "fraud-dq-schedule",
    "CurrentStatus": "CompletedWithViolations"
  }
}
```

> "If the SageMaker Pipeline exists, the test invocation should succeed and start a pipeline execution. Check the SageMaker Pipelines console to verify."

---

### STEP 14 -- The Closed-Loop Architecture Walkthrough (8 minutes)

**Pacing: whiteboard diagram with discussion.**

Draw the complete closed-loop architecture:

```
Production Endpoint (fraud-rf-production-ep)
    |
    |--- Data Capture ---> S3 (request/response logs)
    |
    v
Model Monitor (hourly schedule)
    |
    |--- statistics.json vs. baseline
    |
    v
Violation detected? ----No----> Continue monitoring
    |
    Yes
    |
    v
CloudWatch Metric (data_quality_violations > 0)
    |
    v
CloudWatch Alarm fires
    |
    v
EventBridge Rule matches SageMaker Model Monitor Alert
    |
    v
Lambda: fraud-retrain-trigger
    |
    v
SageMaker Pipeline: Preprocess --> Train --> Evaluate --> CheckQuality --> Register
    |
    v
Model Registry (PendingManualApproval)
    |
    v
Human reviews and approves
    |
    v
Deploy new model to Endpoint
    |
    v
(cycle repeats)
```

> "This is the complete MLOps feedback loop. Every component maps to a service you have used in this curriculum. Data capture is configuration. Model Monitor is a scheduled job. CloudWatch and EventBridge are AWS infrastructure services. Lambda is a glue function. The SageMaker Pipeline is the automated workflow from Module 4 of the foundational skill."

> "The only manual step remaining is human approval in the Model Registry. At Level 2 MLOps maturity, even that step can be automated -- Lambda checks whether the retrained model exceeds the baseline metrics and auto-approves if it does. But for FraudShield, a fraud detection model affecting financial decisions, human review is a regulatory requirement."

[PAUSE FOR Q&A - Ask: "What would happen if the retrained model also fails quality checks?" (The ConditionStep in the pipeline rejects it. No model is registered. The monitoring loop continues on the old model. The team investigates why retraining did not improve quality -- possibly the training data itself needs curation.)]

---

### STEP 15 -- Mandatory Cleanup (5 minutes)

**Pacing: live demonstration. EVERY student must complete this step.**

> "We need to clean up the monitoring schedules, the endpoint, and the Lambda function."

```python
# STEP 15: Cleanup all monitoring and endpoint resources
sm_client = boto3.client("sagemaker")

# Delete monitoring schedules
schedules_to_delete = ["fraud-dq-schedule", "fraud-dq-manual-run", "fraud-mq-schedule"]
for schedule in schedules_to_delete:
    try:
        sm_client.delete_monitoring_schedule(MonitoringScheduleName=schedule)
        print(f"Deleted monitoring schedule: {schedule}")
    except Exception as e:
        print(f"Schedule {schedule}: {e}")

# Delete the endpoint
try:
    sm_client.delete_endpoint(EndpointName="fraud-rf-production-ep")
    print("Deleted endpoint: fraud-rf-production-ep")
except Exception as e:
    print(f"Endpoint: {e}")

# Delete endpoint config and model
try:
    sm_client.delete_endpoint_config(EndpointConfigName="fraud-rf-production-config-dc")
    print("Deleted endpoint config.")
except Exception as e:
    print(f"Config: {e}")

print("\nAlso delete manually in the console:")
print("  - CloudWatch Alarm: fraud-dq-violation-alarm")
print("  - EventBridge Rule: fraud-monitor-retrain-trigger")
print("  - Lambda Function: fraud-retrain-trigger")
print("  - SNS Topic: fraudshield-monitoring-alerts (if created for this lecture)")
```

**Console cleanup checklist:**

1. **CloudWatch > Alarms:** Delete `fraud-dq-violation-alarm`
2. **EventBridge > Rules:** Delete `fraud-monitor-retrain-trigger`
3. **Lambda > Functions:** Delete `fraud-retrain-trigger`
4. **SageMaker > Endpoints:** Verify endpoint is deleted
5. **Billing:** Check for any remaining active resources

> "Monitoring schedules continue running (and billing) until explicitly deleted. Always verify."

**Teaching Note:** Walk around the room to verify every student has deleted their monitoring schedules and endpoint.

[PAUSE FOR Q&A]

---

## Post-Lecture Wrap-Up

**Duration:** 25 minutes

### Summary (5 minutes)

> "Today you built a complete monitoring stack: a data quality baseline from training data, data capture on a live endpoint, a scheduled monitoring job that detects feature drift, model quality monitoring with ground truth ingestion, and bias monitoring with SageMaker Clarify. You interpreted violation reports, understood K-S and Chi-Squared tests, and connected monitoring alerts to automated retraining through CloudWatch, EventBridge, and Lambda."

> "The closed-loop architecture you designed is the standard pattern for production ML systems. Monitoring detects problems. Alerts route to automation. Automation retrains and registers. Humans approve. The new model deploys. The cycle repeats indefinitely."

> "In Module 6, you will focus on the infrastructure beneath all of this: cost optimization with Spot Training, auto-scaling for production endpoints, and security architecture with VPC and KMS. The monitoring and inference patterns from Modules 4 and 5 operate on top of the architecture patterns in Module 6."

### Discussion Activity (20 minutes)

> "Consider this scenario: FraudShield's model quality monitor shows that precision dropped from 0.92 to 0.78 over the past week, but data quality monitoring shows no violations -- feature distributions have not changed."

Discussion questions:

1. What type of drift is this? (Concept drift -- the relationship between features and labels changed.)
2. Why would data quality monitoring miss it? (Data quality only checks input feature distributions, not the feature-label relationship.)
3. What should the team investigate? (New fraud patterns, changes in labeling criteria, external events like a fraud ring targeting a new attack vector.)
4. Should the automated retraining pipeline help? (Only if the new training data includes examples of the changed patterns. If the training data has not been updated, retraining on the same data produces the same model.)

Allow 10 minutes for small group discussion, then 10 minutes for groups to share.

---

## Instructor Notes -- Common Issues

| Issue | Resolution |
|-------|-----------|
| Data capture files not appearing in S3 | Files appear within 3-5 minutes. Verify capture is enabled on the endpoint (check endpoint details in console). Also ensure the endpoint has received at least one invocation after enabling capture. |
| Baseline job fails with data format error | Verify the training CSV has no header row (or set header=True in the dataset format). Column types must be consistent. |
| Monitoring schedule shows no executions | The schedule runs at the configured interval. For hourly, the first execution starts at the top of the next hour. Use a short cron interval for classroom demos. |
| Violation report is empty despite injected drift | Ensure enough drifted samples were captured. A monitoring window with only 5 samples may not produce statistically significant violations. Send at least 50-100 drifted requests. |
| Clarify baseline job takes too long | Clarify runs predictions on the entire dataset. For large datasets, this can take 20+ minutes. Use a smaller sample for classroom demos (1000 rows). |
| Lambda function cannot start pipeline | Verify the Lambda execution role has `sagemaker:StartPipelineExecution` permission. Also verify the pipeline name matches exactly. |
| Associates forget to delete monitoring schedules | Monitoring schedules bill for every execution. Walk around and verify deletion. Check under Inference > Model monitoring > Monitoring schedules. |
| CloudWatch metric not appearing | Model Monitor metrics appear only after at least one monitoring execution completes. Wait for the first scheduled execution or trigger a manual one. |
