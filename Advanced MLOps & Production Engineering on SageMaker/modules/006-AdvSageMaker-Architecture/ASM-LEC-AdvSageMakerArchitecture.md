# AdvSageMaker-Architecture Lecture - Instructor Guide

**Total Duration:** 180 Minutes (3 Stages)
**Consolidated Activities:** Managed Spot Training (ASM-CT-ManagedSpotTraining), Spot Instances for HPO (ASM-CT-SpotInstancesForHPO), Instance Right-sizing (ASM-CT-InstanceRightSizing), Auto-scaling Policies (ASM-CT-AutoScalingPolicies), Inference Recommender (ASM-CT-InferenceRecommender), CloudWatch Dashboards (ASM-CT-CloudWatchDashboards), VPC PrivateLink & KMS (ASM-CT-VPCPrivateLinkKMS)

| Block | Content | Minutes |
|-------|---------|---------|
| Stage 1 | Cost Optimization -- Spot Training and Instance Right-sizing | 45 |
| Break 1 | Stretch / Questions | 10 |
| Stage 2 | Production Inference -- Auto-scaling, Recommender, and Dashboards | 45 |
| Break 2 | Stretch / Questions | 10 |
| Stage 3 | Security Architecture -- VPC, PrivateLink, and KMS | 45 |
| Buffer | Open Q&A, Wrap-Up | 25 |

---

## Lecture Overview

**Unified Scenario -- FraudShield Risk Analytics (Advanced)**

FraudShield's fraud detection system is functionally complete: models train, deploy, serve predictions, and retrain automatically when drift is detected. But the CTO has three concerns before green-lighting the production launch.

First, cost: the data science team trains models weekly and runs hyperparameter tuning jobs monthly. Each training run on an `ml.m5.xlarge` on-demand instance costs a predictable amount, but the monthly bill is growing. The CFO wants a 60% reduction in training costs without sacrificing model quality.

Second, reliability: the production endpoint serves the payment gateway at peak volumes of 500 requests per second during holiday sales. A single `ml.m5.xlarge` instance cannot handle that load, but paying for 10 instances at 3 AM when traffic drops to 5 requests per second is wasteful. The endpoint needs to scale automatically.

Third, security: FraudShield processes financial transaction data subject to PCI-DSS compliance. The current architecture allows SageMaker to access S3 and other services over the public internet. The security team requires all traffic to stay within the VPC, all data at rest to be encrypted with customer-managed keys, and all API calls to route through private endpoints.

Today the team addresses all three: Managed Spot Training for cost reduction, auto-scaling and Inference Recommender for production reliability, and VPC with PrivateLink and KMS for security compliance.

**Console + SDK:** Cost optimization and security configurations begin in the console for visibility, transitioning to SDK for training job and auto-scaling setup. Security architecture is primarily console-driven because VPC and KMS are infrastructure configurations.

---

## Pre-Lecture Setup

### Instructor Checklist

- [ ] SageMaker Studio open with a notebook on an `ml.t3.medium` instance
- [ ] FraudShield training data in S3 (`s3://<bucket>/fraudshield/data/train.csv`)
- [ ] Script Mode `train.py` from the foundational modules available
- [ ] A deployed real-time endpoint for auto-scaling exercises (or deploy one at lecture start)
- [ ] IAM execution role with S3, SageMaker, CloudWatch, EC2 (VPC), KMS, and Application Auto Scaling permissions
- [ ] A VPC with at least two private subnets and a security group prepared (or use the default VPC)
- [ ] A KMS key created (or be prepared to create one during Stage 3)
- [ ] Screen sharing enabled, font increased for projector readability
- [ ] This instructor guide open in a second tab
- [ ] AWS Billing dashboard bookmarked for cost comparison demonstrations

### VPC Setup (if not already configured)

If no private subnets exist, create them before class:

1. Open the VPC console.
2. Select the default VPC (or create a new one).
3. Create two private subnets in different Availability Zones.
4. Create a NAT Gateway in a public subnet (needed for SageMaker to download container images).
5. Create a route table for the private subnets with a route to the NAT Gateway.

Note the subnet IDs and security group ID for Stage 3.

### Student Prerequisites

- [ ] Completed foundational SageMaker modules and Modules 4-5
- [ ] Completed readings: Managed Spot Training, Spot Instances for HPO, Instance Right-sizing, Auto-scaling Policies, Inference Recommender, CloudWatch Dashboards, VPC PrivateLink & KMS
- [ ] Studio notebook open on an `ml.t3.medium` instance
- [ ] Familiarity with SageMaker training jobs and endpoint deployment

---

## Stage 1: Cost Optimization -- Spot Training and Instance Right-sizing

**Duration:** 45 minutes
**Exit Criteria Addressed:**
- Configure a Managed Spot Training job with checkpointing (Required)
- Compare on-demand vs. Spot instance costs (Required)
- Analyze CloudWatch metrics for instance right-sizing (Required)
- Make instance right-sizing recommendations based on utilization data (Preferred)

### Instructor Opening (5 minutes -- talk, no code)

> "FraudShield trains models weekly and runs hyperparameter tuning monthly. The CFO sees the AWS bill and asks: 'Can we get the same models for less money?' The answer is yes -- Managed Spot Training uses spare EC2 capacity at up to 90% discount. The catch: AWS can interrupt your training job at any time if it needs the capacity back."

> "Spot interruptions sound alarming, but SageMaker handles them gracefully with checkpointing. If a Spot instance is reclaimed, SageMaker saves the training state, waits for a new Spot instance, and resumes from the checkpoint. You pay only for the compute time actually used."

---

### STEP 1 -- Configuring Managed Spot Training from the Console (12 minutes)

**Pacing: live demonstration.** All Associates follow along.

1. Open the SageMaker console. Navigate to **Training > Training jobs**. Click **Create training job**.
2. Fill in the details:
   - **Job name:** `fraud-rf-spot-training`
   - **IAM role:** Select the existing execution role.
   - **Algorithm:** Select the scikit-learn container (or **Your own algorithm container** with the container URI).

> "Everything up to here is identical to the foundational training job. The Spot configuration is in the resource settings."

3. **Resource configuration:**
   - **Instance type:** `ml.m5.xlarge`
   - **Instance count:** `1`
   - **Managed Spot Training:** Check **Enable**.

> "That single checkbox is the difference between on-demand and Spot. SageMaker handles the rest -- bidding for capacity, monitoring for interruptions, and resuming from checkpoints."

   - **Maximum runtime:** `3600` seconds (1 hour)
   - **Maximum wait time:** `7200` seconds (2 hours)

> "Maximum runtime is how long the training can take. Maximum wait time is how long SageMaker waits for Spot capacity. If no Spot instance is available within the wait time, the job fails. Set the wait time to at least 2x the expected runtime."

4. **Checkpoint configuration:**
   - **S3 URI:** `s3://<bucket>/fraudshield/checkpoints/`
   - **Local path:** `/opt/ml/checkpoints`

> "Checkpointing is the safety net. Your training script must save checkpoints to `/opt/ml/checkpoints` periodically. If the Spot instance is interrupted, SageMaker copies the latest checkpoint to S3. When a new instance starts, it copies the checkpoint back and your script resumes."

5. Configure input channels and output path as in the foundational training job.
6. Click **Create training job**.

> "The job may start immediately if Spot capacity is available, or it may wait. The status shows 'Starting' while waiting for capacity."

---

### STEP 2 -- Spot Training via SDK with Checkpointing (10 minutes)

**Pacing: live demonstration in notebook.**

```python
# STEP 2: Managed Spot Training with SDK
import sagemaker
from sagemaker.sklearn import SKLearn

session = sagemaker.Session()
role = sagemaker.get_execution_role()
bucket = session.default_bucket()

estimator = SKLearn(
    entry_point="train.py",
    framework_version="1.2-1",
    role=role,
    instance_count=1,
    instance_type="ml.m5.xlarge",
    sagemaker_session=session,
    base_job_name="fraud-rf-spot",
    
    # Spot Training configuration
    use_spot_instances=True,
    max_run=3600,
    max_wait=7200,
    checkpoint_s3_uri=f"s3://{bucket}/fraudshield/checkpoints/",
    checkpoint_local_path="/opt/ml/checkpoints",
)

estimator.fit({
    "train": f"s3://{bucket}/fraudshield/data/train/",
    "validation": f"s3://{bucket}/fraudshield/data/validation/",
}, wait=False)

print(f"Spot Training job started: {estimator.latest_training_job.name}")
```

> "Three parameters enable Spot: `use_spot_instances=True`, `max_run` for training duration, and `max_wait` for total wait including queue time. The `checkpoint_s3_uri` tells SageMaker where to save and restore state."

**Adding checkpoint logic to train.py:**

> "Your training script needs to save and load checkpoints. For scikit-learn with Random Forest, checkpointing is simple because the model trains in one shot. For iterative algorithms like gradient boosting or neural networks, save after each epoch or round."

```python
# Addition to train.py for checkpoint support
import os
import joblib

CHECKPOINT_DIR = "/opt/ml/checkpoints"

def save_checkpoint(model, epoch):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    path = os.path.join(CHECKPOINT_DIR, f"checkpoint-{epoch}.pkl")
    joblib.dump(model, path)
    print(f"Checkpoint saved: {path}")

def load_latest_checkpoint():
    if not os.path.exists(CHECKPOINT_DIR):
        return None, 0
    checkpoints = sorted(
        [f for f in os.listdir(CHECKPOINT_DIR) if f.startswith("checkpoint-")],
        key=lambda x: int(x.split("-")[1].split(".")[0]),
    )
    if not checkpoints:
        return None, 0
    latest = checkpoints[-1]
    epoch = int(latest.split("-")[1].split(".")[0])
    model = joblib.load(os.path.join(CHECKPOINT_DIR, latest))
    print(f"Resumed from checkpoint: {latest}")
    return model, epoch
```

> "For iterative training, the script checks for existing checkpoints at startup. If found, it resumes from that epoch. If not, it starts fresh. This pattern works for any framework."

---

### STEP 3 -- Comparing On-demand vs. Spot Costs (8 minutes)

**Pacing: console demonstration.**

1. Navigate to **Training > Training jobs**. Find the completed Spot training job (or a recent one).
2. Click the job name. Scroll to **Training job metrics**.
3. Show the **Managed Spot Training savings** section:
   - **Training duration:** (actual compute time)
   - **Billable duration:** (same as training duration for Spot)
   - **Savings:** (percentage saved compared to on-demand)

> "SageMaker calculates and displays the savings for every Spot training job. Typical savings range from 60-90% depending on instance type and region."

Display the cost comparison:

| Instance | On-demand (us-east-1) | Spot (typical) | Savings |
|----------|----------------------|-----------------|---------|
| ml.m5.xlarge | ~$0.23/hour | ~$0.07/hour | ~70% |
| ml.m5.2xlarge | ~$0.46/hour | ~$0.14/hour | ~70% |
| ml.p3.2xlarge (GPU) | ~$3.83/hour | ~$1.15/hour | ~70% |

> "For FraudShield's weekly training job that takes 30 minutes, the per-job savings are modest. But for monthly hyperparameter tuning that launches 50 parallel training jobs, the difference is substantial."

[PAUSE FOR Q&A - Ask: "When should you NOT use Spot Training?" (When the training job has a strict deadline and cannot tolerate waiting for Spot capacity. Also when the framework does not support checkpointing and the job takes many hours -- losing progress to an interruption would be costly. Short jobs under 5 minutes rarely benefit because the interruption risk overhead is not worth the discount.)]

---

### STEP 4 -- Analyzing CloudWatch Metrics for Instance Right-sizing (10 minutes)

**Pacing: console demonstration.**

> "Spot Training saves money on the per-hour cost. Instance right-sizing saves money by choosing the right instance in the first place. If your training job uses only 30% of the CPU on an `ml.m5.xlarge`, you are paying for capacity you do not use."

1. Open the CloudWatch console. Navigate to **Metrics > All metrics**.
2. Search for `/aws/sagemaker/TrainingJobs`. Select a recent training job.
3. Show the available metrics:
   - **CPUUtilization**
   - **MemoryUtilization**
   - **DiskUtilization**
   - **GPUUtilization** (if applicable)

4. Graph **CPUUtilization** for the training job. Set the period to 1 minute.

> "Look at the pattern. If CPU utilization peaks at 25% and averages 15%, this job is over-provisioned. An `ml.m5.large` (half the size) would handle it."

5. Graph **MemoryUtilization** on the same chart.

> "Memory is often the binding constraint for tabular data. If memory utilization hits 90%, you need the current instance size even if CPU is underutilized. If both CPU and memory are under 50%, downsize."

**Right-sizing decision framework:**

| Metric Pattern | Recommendation |
|---------------|----------------|
| CPU < 30%, Memory < 30% | Downsize to smaller instance |
| CPU > 80%, Memory < 50% | Consider compute-optimized (ml.c5) instead of general purpose (ml.m5) |
| CPU < 50%, Memory > 80% | Consider memory-optimized (ml.r5) |
| CPU > 80%, Memory > 80% | Current size is appropriate, or upsize if job is slow |
| GPU < 30% (if applicable) | Reconsider whether GPU is needed; switch to CPU instance |

> "Right-sizing is an iterative process. Train once, check metrics, adjust, train again. The goal is to find the smallest instance that completes the job in acceptable time without running out of resources."

```python
# STEP 4b: Retrieve CloudWatch metrics programmatically
import boto3
from datetime import datetime, timedelta

cw = boto3.client("cloudwatch")

response = cw.get_metric_statistics(
    Namespace="/aws/sagemaker/TrainingJobs",
    MetricName="CPUUtilization",
    Dimensions=[
        {"Name": "Host", "Value": "algo-1"},
    ],
    StartTime=datetime.utcnow() - timedelta(hours=2),
    EndTime=datetime.utcnow(),
    Period=60,
    Statistics=["Average", "Maximum"],
)

for datapoint in sorted(response["Datapoints"], key=lambda x: x["Timestamp"]):
    print(f"Time: {datapoint['Timestamp']} | Avg CPU: {datapoint['Average']:.1f}% | Max CPU: {datapoint['Maximum']:.1f}%")
```

> "Programmatic access to CloudWatch metrics enables automated right-sizing reports. In a mature organization, a weekly Lambda function checks all training job metrics and recommends instance changes."

[PAUSE FOR Q&A]

[PAUSE FOR BREAK - 10 MINS]

---

## Stage 2: Production Inference -- Auto-scaling, Recommender, and Dashboards

**Duration:** 45 minutes
**Exit Criteria Addressed:**
- Configure target tracking auto-scaling on a real-time endpoint (Required)
- Run an Inference Recommender job (Required)
- Build a CloudWatch Dashboard for production monitoring (Required)
- Set up CloudWatch Alarms for latency and error metrics (Preferred)

### Instructor Opening (3 minutes)

> "FraudShield's production endpoint needs to handle 500 requests per second during holiday peaks and 5 requests per second at 3 AM. A fixed instance count wastes money or drops requests. Auto-scaling solves this: SageMaker adds instances when load increases and removes them when load decreases, within bounds you define."

---

### STEP 5 -- Configuring Target Tracking Auto-scaling from the Console (12 minutes)

**Pacing: live demonstration.** All Associates follow along.

1. Open the SageMaker console. Navigate to **Inference > Endpoints**.
2. Click the production endpoint (e.g., `fraud-rf-production-ep`).
3. Scroll to the **Endpoint runtime settings** section. Click the variant name.
4. Click **Configure auto scaling** (or find auto-scaling settings).

> "Auto-scaling is configured per production variant. If you have multiple variants for A/B testing, each variant scales independently."

5. **Auto-scaling configuration:**
   - **Minimum instance count:** `1`
   - **Maximum instance count:** `4`

> "Minimum 1 ensures the endpoint is always available. Maximum 4 caps costs during unexpected traffic spikes. For FraudShield, the security team requires at least 1 instance; finance caps at 4 for budget control."

6. **Scaling policy:**
   - **Policy type:** Target tracking
   - **Target metric:** `SageMakerVariantInvocationsPerInstance`
   - **Target value:** `100`
   - **Scale-in cool down:** `300` seconds (5 minutes)
   - **Scale-out cool down:** `60` seconds (1 minute)

> "Target tracking means: 'Keep the number of invocations per instance around 100.' If each instance gets 150 invocations, SageMaker adds an instance. If each gets 50, SageMaker removes one. The cool-down periods prevent thrashing -- rapid scale-out and scale-in cycles."

> "Scale-out cool down is shorter than scale-in because adding capacity during a spike is urgent, but removing capacity can wait to verify the traffic drop is sustained."

7. Click **Save** (or **Create scaling policy**).

---

### STEP 6 -- Auto-scaling via SDK (8 minutes)

**Pacing: live demonstration in notebook.**

```python
# STEP 6: Configure auto-scaling via SDK
import boto3

aas_client = boto3.client("application-autoscaling")

resource_id = "endpoint/fraud-rf-production-ep/variant/AllTraffic"

# Register the scalable target
aas_client.register_scalable_target(
    ServiceNamespace="sagemaker",
    ResourceId=resource_id,
    ScalableDimension="sagemaker:variant:DesiredInstanceCount",
    MinCapacity=1,
    MaxCapacity=4,
)

# Create the target tracking scaling policy
aas_client.put_scaling_policy(
    PolicyName="fraud-endpoint-scaling-policy",
    ServiceNamespace="sagemaker",
    ResourceId=resource_id,
    ScalableDimension="sagemaker:variant:DesiredInstanceCount",
    PolicyType="TargetTrackingScaling",
    TargetTrackingScalingPolicyConfiguration={
        "TargetValue": 100.0,
        "PredefinedMetricSpecification": {
            "PredefinedMetricType": "SageMakerVariantInvocationsPerInstance",
        },
        "ScaleInCooldown": 300,
        "ScaleOutCooldown": 60,
    },
)

print("Auto-scaling policy configured.")
```

> "The SDK approach uses Application Auto Scaling, the same service that scales DynamoDB, ECS, and other AWS services. The `ResourceId` format is specific to SageMaker: `endpoint/<name>/variant/<variant-name>`."

[PAUSE FOR Q&A - Ask: "Why use InvocationsPerInstance instead of CPUUtilization as the scaling metric?" (InvocationsPerInstance directly reflects request load, which is what the endpoint serves. CPU utilization may not correlate linearly with request count -- a model that does heavy preprocessing might spike CPU on fewer requests, causing premature scaling.)]

---

### STEP 7 -- Running an Inference Recommender Default Job (10 minutes)

**Pacing: console demonstration followed by SDK.**

> "Auto-scaling tells SageMaker how to adjust capacity. Inference Recommender tells you which instance type to use in the first place. It benchmarks your model on multiple instance types and reports latency, throughput, and cost for each."

1. Navigate to **Inference > Inference recommender**. Click **Create job**.
2. **Job configuration:**
   - **Job type:** Default (quick benchmark across instance types)
   - **Model package:** Select the model from the Model Registry (or specify model data directly)
   - **Job name:** `fraud-rf-recommender`

> "The Default job tests your model on a curated set of instance types. It sends synthetic traffic, measures P50 and P99 latency, throughput, and cost per inference. The results rank instance types by cost-effectiveness."

3. Click **Create** and wait for the job to complete (5-15 minutes).

**Or via SDK:**

```python
# STEP 7: Inference Recommender via SDK
sm_client = boto3.client("sagemaker")

sm_client.create_inference_recommendations_job(
    JobName="fraud-rf-recommender",
    JobType="Default",
    RoleArn=role,
    InputConfig={
        "ModelPackageVersionArn": "<model-package-arn>",
        "JobDurationInSeconds": 7200,
    },
)

print("Inference Recommender job started.")
```

> "While the job runs, let's build the CloudWatch Dashboard."

4. After the job completes, review the results in the console:
   - Navigate to the completed job. Show the recommendations table.

| Instance Type | P50 Latency (ms) | P99 Latency (ms) | Max Throughput (TPS) | Cost per Hour | Cost per Inference |
|--------------|-------------------|-------------------|---------------------|---------------|-------------------|
| ml.m5.large | 12 | 45 | 150 | $0.12 | $0.000022 |
| ml.m5.xlarge | 8 | 25 | 300 | $0.23 | $0.000021 |
| ml.c5.xlarge | 6 | 18 | 400 | $0.21 | $0.000015 |

> "Inference Recommender might reveal that `ml.c5.xlarge` (compute-optimized) gives better latency at lower cost than `ml.m5.xlarge` (general purpose) for this model. The recommendation depends on whether your model is CPU-bound, memory-bound, or I/O-bound."

---

### STEP 8 -- Building a CloudWatch Dashboard (8 minutes)

**Pacing: console demonstration.**

1. Open the CloudWatch console. Navigate to **Dashboards**. Click **Create dashboard**.
   - **Name:** `FraudShield-Production`
2. Add widgets one by one:

**Widget 1: Invocations per Instance**

   - Click **Add widget > Line chart**.
   - Select metric: **SageMaker > Endpoint > InvocationsPerInstance**.
   - Filter by endpoint: `fraud-rf-production-ep`.

> "This is the scaling metric. You want to see it hovering around your target value (100). Sustained values above the target mean auto-scaling should be adding instances."

**Widget 2: Model Latency**

   - Click **Add widget > Line chart**.
   - Select metric: **SageMaker > Endpoint > ModelLatency** (microseconds).

> "ModelLatency measures the time the model container takes to process a request, excluding network overhead. Track both average and P99."

**Widget 3: Invocation Errors**

   - Click **Add widget > Number**.
   - Select metric: **SageMaker > Endpoint > Invocation4XXErrors** and **Invocation5XXErrors**.

> "4XX errors are client errors (wrong input format). 5XX errors are server errors (model crash, out of memory). Any sustained 5XX errors require immediate investigation."

**Widget 4: Instance Count**

   - Click **Add widget > Line chart**.
   - Use a custom metric or query for the current instance count from the endpoint.

3. Click **Save dashboard**.

> "This dashboard gives you a single view of FraudShield's production health. In the foundational modules, you checked CloudWatch Logs for individual errors. This dashboard aggregates metrics across all invocations."

---

### STEP 9 -- Setting Up CloudWatch Alarms (5 minutes)

**Pacing: console demonstration.**

1. From the dashboard, click a metric widget. Select **Create alarm**.
2. Configure a latency alarm:
   - **Metric:** ModelLatency
   - **Statistic:** p99
   - **Period:** 5 minutes
   - **Threshold:** Greater than 500,000 microseconds (500ms)
   - **Action:** Notify `fraudshield-monitoring-alerts` SNS topic

> "If P99 latency exceeds 500ms for 5 minutes, the team gets an alert. This could indicate the instance is overloaded (auto-scaling should kick in) or the model has a performance regression."

3. Configure an error rate alarm:
   - **Metric:** Invocation5XXErrors
   - **Threshold:** Greater than 10 in 5 minutes
   - **Action:** Same SNS topic

> "Two alarms form the minimum production monitoring: latency and errors. Add more as the system matures -- invocation count dropping to zero (endpoint down), memory utilization spikes, etc."

[PAUSE FOR Q&A]

[PAUSE FOR BREAK - 10 MINS]

---

## Stage 3: Security Architecture -- VPC, PrivateLink, and KMS

**Duration:** 45 minutes
**Exit Criteria Addressed:**
- Configure a training job to run in a private VPC subnet (Required)
- Create VPC Endpoints for S3 and SageMaker API (Required)
- Configure KMS encryption for training volumes and S3 artifacts (Required)
- Describe the full production security architecture (Preferred)

### Instructor Opening (5 minutes -- talk, no code)

> "FraudShield processes credit card transactions. PCI-DSS compliance requires that sensitive data never traverses the public internet, is encrypted at rest with keys the organization controls, and is accessible only from authorized network segments. SageMaker runs in AWS's managed infrastructure by default -- training jobs download data from S3 over the public internet, and API calls go through public endpoints."

> "Three AWS services lock down the architecture: VPC isolates network traffic, VPC Endpoints (PrivateLink) keep API calls private, and KMS encrypts data at rest with customer-managed keys. Today we configure all three for FraudShield."

---

### STEP 10 -- Configuring a Training Job in a Private VPC (12 minutes)

**Pacing: console demonstration.** All Associates follow along.

1. Open the SageMaker console. Navigate to **Training > Training jobs**. Click **Create training job**.
2. Fill in the standard details (job name, role, algorithm, input/output).
3. Scroll to **Network** section (or **VPC configuration**):

> "By default, training jobs run in SageMaker-managed infrastructure with internet access. Enabling VPC mode puts the training containers in your VPC's private subnets."

   - **VPC:** Select the VPC.
   - **Subnets:** Select two private subnets (different Availability Zones for resilience).
   - **Security groups:** Select a security group that allows outbound traffic to S3 and CloudWatch.

> "When you place a training job in a VPC, the containers lose direct internet access. They can only reach services through VPC Endpoints or a NAT Gateway. This is by design -- it prevents data exfiltration."

   - **Enable network isolation:** Leave unchecked for this exercise.

> "Network isolation goes further: it blocks ALL network access, including S3. The training container can only access data provided through input channels. Enable this for the strictest compliance requirements, but be aware that the container cannot download packages or access external resources."

4. Do not start the job yet -- we need VPC Endpoints first.

**Show the VPC diagram:**

```
VPC (10.0.0.0/16)
  |
  +-- Private Subnet A (10.0.1.0/24)
  |     |-- Training Job Container
  |     |-- Endpoint Instance
  |
  +-- Private Subnet B (10.0.2.0/24)
  |     |-- (failover)
  |
  +-- Public Subnet (10.0.0.0/24)
        |-- NAT Gateway (for internet access if needed)
```

> "Training containers and endpoint instances run in private subnets. They cannot be reached from the internet. The NAT Gateway provides outbound-only internet access for downloading container images during initial setup."

---

### STEP 11 -- Creating VPC Endpoints for S3 and SageMaker API (12 minutes)

**Pacing: console demonstration.**

> "VPC Endpoints replace public internet access with private connections to AWS services. There are two types: Gateway Endpoints (for S3 and DynamoDB) and Interface Endpoints (for everything else, using PrivateLink)."

**Gateway Endpoint for S3:**

1. Open the VPC console. Navigate to **Endpoints**. Click **Create endpoint**.
2. **Endpoint details:**
   - **Service category:** AWS services
   - **Service name:** Search for `com.amazonaws.<region>.s3` and select the Gateway type.
   - **VPC:** Select the VPC.
   - **Route tables:** Select the route table(s) associated with the private subnets.

> "A Gateway Endpoint adds a route in your subnet's route table that directs S3-bound traffic through the AWS private backbone instead of the internet. No data leaves the AWS network."

3. Click **Create endpoint**.

**Interface Endpoint for SageMaker API:**

1. Click **Create endpoint** again.
2. **Endpoint details:**
   - **Service category:** AWS services
   - **Service name:** Search for `com.amazonaws.<region>.sagemaker.api`.
   - **VPC:** Select the VPC.
   - **Subnets:** Select the private subnets.
   - **Security groups:** Select a security group that allows inbound HTTPS (port 443).
   - **Enable DNS name:** Check this option.

> "An Interface Endpoint creates an elastic network interface (ENI) in your subnet with a private IP address. When your code calls the SageMaker API, DNS resolves to this private IP instead of the public SageMaker endpoint. The call never leaves the VPC."

3. Click **Create endpoint**.

4. Repeat for `com.amazonaws.<region>.sagemaker.runtime` (for invoke_endpoint calls).

> "You need separate endpoints for the SageMaker API (create/delete/describe operations) and SageMaker Runtime (invoke_endpoint). In production, also create endpoints for CloudWatch Logs and ECR so that training containers can push logs and pull images without internet access."

**Complete list of VPC Endpoints for a locked-down SageMaker deployment:**

| Service | Endpoint Type | Purpose |
|---------|--------------|---------|
| S3 | Gateway | Training data, model artifacts, data capture |
| SageMaker API | Interface | Create/describe/delete operations |
| SageMaker Runtime | Interface | Invoke endpoint |
| CloudWatch Logs | Interface | Training and endpoint logs |
| ECR (api + dkr) | Interface | Pull container images |
| STS | Interface | AssumeRole for IAM |
| KMS | Interface | Encryption/decryption operations |

> "Each Interface Endpoint costs approximately $0.01/hour plus data transfer. For a production deployment with 6 Interface Endpoints, that is about $45/month -- a small price for PCI-DSS compliance."

[PAUSE FOR Q&A - Ask: "If you forget to create the CloudWatch Logs endpoint, what happens to your training job in a VPC with no NAT Gateway?" (The training job runs but cannot push logs. You lose all visibility into training progress and errors. The job may succeed, but you cannot debug failures. Always include CloudWatch Logs in your VPC Endpoint list.)]

---

### STEP 12 -- Configuring KMS Encryption (10 minutes)

**Pacing: console demonstration.**

> "VPC keeps data off the public internet while in transit inside AWS. KMS encrypts data at rest. By default, SageMaker encrypts training volumes and S3 artifacts with AWS-managed keys. PCI-DSS requires customer-managed keys so FraudShield controls key rotation and access policies."

**Creating a KMS Key (if not already done):**

1. Open the KMS console. Click **Create key**.
   - **Key type:** Symmetric
   - **Key usage:** Encrypt and decrypt
   - **Alias:** `fraudshield-sagemaker-key`
   - **Key administrators:** Add your IAM user or role
   - **Key users:** Add the SageMaker execution role

> "The key policy controls who can use the key. The SageMaker execution role must be a key user, or training jobs and endpoints will fail with AccessDeniedException."

2. Click **Create key**. Note the key ARN.

**Applying KMS to a training job:**

3. Return to the SageMaker training job creation page (or show the SDK approach).

```python
# STEP 12: Training job with KMS encryption
estimator = SKLearn(
    entry_point="train.py",
    framework_version="1.2-1",
    role=role,
    instance_count=1,
    instance_type="ml.m5.xlarge",
    sagemaker_session=session,
    base_job_name="fraud-rf-encrypted",
    
    # VPC configuration
    subnets=["subnet-xxxxxx", "subnet-yyyyyy"],
    security_group_ids=["sg-zzzzzz"],
    
    # KMS encryption
    output_kms_key="arn:aws:kms:<region>:<account>:key/<key-id>",
    volume_kms_key="arn:aws:kms:<region>:<account>:key/<key-id>",
)

estimator.fit({
    "train": f"s3://{bucket}/fraudshield/data/train/",
}, wait=False)
```

> "`output_kms_key` encrypts the model artifact written to S3. `volume_kms_key` encrypts the EBS volume attached to the training instance. Both use the same customer-managed key."

**Applying KMS to an endpoint:**

```python
# STEP 12b: Endpoint with KMS-encrypted volume
from sagemaker.model import Model

model = Model(
    image_uri="<sklearn-container-uri>",
    model_data=f"s3://{bucket}/fraudshield/output/model.tar.gz",
    role=role,
    sagemaker_session=session,
)

predictor = model.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.xlarge",
    endpoint_name="fraud-rf-encrypted-ep",
    kms_key="arn:aws:kms:<region>:<account>:key/<key-id>",
)
```

> "The `kms_key` parameter on deploy encrypts the endpoint's EBS volume. Combined with S3 server-side encryption using the same KMS key, all data at rest is encrypted with a key FraudShield controls."

---

### STEP 13 -- Full Production Security Architecture Walkthrough (6 minutes)

**Pacing: whiteboard diagram with discussion.**

Draw the complete production architecture:

```
Internet
    |
    X (blocked by VPC -- no public access)
    
VPC (10.0.0.0/16)
  |
  +-- Private Subnet A
  |     |-- SageMaker Training Job (encrypted EBS via KMS)
  |     |-- SageMaker Endpoint (encrypted EBS via KMS)
  |     |-- VPC Endpoint: SageMaker API (PrivateLink)
  |     |-- VPC Endpoint: SageMaker Runtime (PrivateLink)
  |     |-- VPC Endpoint: CloudWatch Logs (PrivateLink)
  |     |-- VPC Endpoint: KMS (PrivateLink)
  |
  +-- Private Subnet B
  |     |-- (failover instances)
  |     |-- VPC Endpoint: ECR (PrivateLink)
  |     |-- VPC Endpoint: STS (PrivateLink)
  |
  +-- S3 Gateway Endpoint (route table entry)
  |     |-- Training data (SSE-KMS encrypted)
  |     |-- Model artifacts (SSE-KMS encrypted)
  |     |-- Data capture logs (SSE-KMS encrypted)
  |
  +-- IAM
        |-- Execution role with least-privilege policies
        |-- S3 bucket policy restricting access to VPC Endpoint
        |-- KMS key policy restricting to authorized roles
```

> "Four layers of security: VPC isolates the network, PrivateLink keeps API calls private, KMS encrypts data at rest, and IAM controls access at every level. This architecture satisfies PCI-DSS, HIPAA, and SOC 2 requirements for most ML workloads."

> "The S3 bucket policy deserves special mention. You can add a condition that restricts S3 access to requests originating from your VPC Endpoint. Even if someone has valid IAM credentials, they cannot access the data from outside the VPC."

```json
{
  "Condition": {
    "StringEquals": {
      "aws:sourceVpce": "vpce-xxxxxxxxx"
    }
  }
}
```

> "This condition on the S3 bucket policy means: allow access only through the VPC Gateway Endpoint. Requests from the internet, even with valid credentials, are denied."

[PAUSE FOR Q&A]

---

### STEP 14 -- Mandatory Cleanup (5 minutes)

**Pacing: live demonstration. EVERY student must complete this step.**

> "We need to clean up all resources created today: endpoints, auto-scaling policies, VPC Endpoints, CloudWatch Dashboards, and KMS key usage."

```python
# STEP 14: Cleanup
sm_client = boto3.client("sagemaker")
aas_client = boto3.client("application-autoscaling")

# Remove auto-scaling policy
try:
    aas_client.deregister_scalable_target(
        ServiceNamespace="sagemaker",
        ResourceId="endpoint/fraud-rf-production-ep/variant/AllTraffic",
        ScalableDimension="sagemaker:variant:DesiredInstanceCount",
    )
    print("Auto-scaling policy removed.")
except Exception as e:
    print(f"Auto-scaling: {e}")

# Delete endpoints
endpoints_to_delete = [
    "fraud-rf-production-ep",
    "fraud-rf-encrypted-ep",
]

for ep in endpoints_to_delete:
    try:
        sm_client.delete_endpoint(EndpointName=ep)
        print(f"Deleted endpoint: {ep}")
    except Exception as e:
        print(f"Endpoint {ep}: {e}")

# Delete endpoint configs and models (check console for auto-generated names)
print("\nAlso delete manually in the console:")
print("  - Endpoint configurations for the above endpoints")
print("  - Model objects created during this lecture")
print("  - CloudWatch Dashboard: FraudShield-Production")
print("  - CloudWatch Alarms: latency and error alarms")
print("  - Inference Recommender job results (auto-expire, no action needed)")
```

**Console cleanup checklist:**

1. **SageMaker > Endpoints:** Delete all endpoints created today
2. **SageMaker > Endpoint configurations:** Delete associated configs
3. **SageMaker > Models:** Delete models created today
4. **SageMaker > Training jobs:** Spot training job (already terminated, no action)
5. **CloudWatch > Dashboards:** Delete `FraudShield-Production`
6. **CloudWatch > Alarms:** Delete latency and error alarms
7. **Application Auto Scaling:** Deregistered via SDK above
8. **VPC Endpoints:** Delete Interface Endpoints created today (they incur hourly charges)
9. **KMS Key:** Disable (do not delete -- key deletion has a waiting period and may be needed for decrypting existing artifacts)
10. **Billing:** Verify no unexpected active resources

> "VPC Interface Endpoints bill hourly. Do not forget to delete them. The S3 Gateway Endpoint is free and can remain."

**Teaching Note:** Walk around the room to verify every student has deleted their endpoints and VPC Interface Endpoints. VPC Endpoints are easy to forget because they are in a different console.

[PAUSE FOR Q&A]

---

## Post-Lecture Wrap-Up

**Duration:** 25 minutes

### Summary (5 minutes)

> "Today you addressed the three pillars of production ML infrastructure: cost, reliability, and security. Managed Spot Training reduces training costs by up to 90% with checkpointing as the safety net. Instance right-sizing ensures you are not over-provisioning. Auto-scaling adapts endpoint capacity to real traffic patterns. Inference Recommender benchmarks instance types to find the cost-performance sweet spot. CloudWatch Dashboards and Alarms give you visibility into production health."

> "On the security side, VPC isolates your ML workloads from the public internet, VPC Endpoints keep API calls private via PrivateLink, and KMS encrypts all data at rest with keys you control. Together, these three services satisfy the compliance requirements that gate production launches in regulated industries."

> "This module completes the Advanced MLOps curriculum. Over three modules, you progressed from advanced inference patterns (Serverless, Async, Batch, Multi-Model) to production monitoring (Data Quality, Model Quality, Bias, automated retraining) to infrastructure architecture (cost optimization, auto-scaling, security). Every component connects to the FraudShield system you have been building since the foundational SageMaker skill."

### Discussion Activity (20 minutes)

> "The CTO asks for a one-page architecture review of FraudShield's production ML system. Working in pairs, create a diagram and brief description that covers:"

1. How models are trained (Spot Training, VPC, KMS encryption)
2. How models are deployed (inference pattern selection, auto-scaling)
3. How models are monitored (Data Quality, Model Quality, Bias monitoring)
4. How retraining is triggered (CloudWatch, EventBridge, Lambda, Pipeline)
5. How security is enforced (VPC, PrivateLink, KMS, IAM)

Allow 12 minutes for design, then 8 minutes for 2 groups to present.

> "This exercise synthesizes everything from Modules 4, 5, and 6. Your architecture review is the deliverable that demonstrates you can design, not just implement, a production ML system."

---

## Instructor Notes -- Common Issues

| Issue | Resolution |
|-------|-----------|
| Spot Training job stuck in "Starting" | Spot capacity may not be available. Wait for `max_wait` duration. If it fails, rerun with on-demand for the classroom. |
| Checkpointing fails silently | Verify `checkpoint_local_path` in the estimator matches the path in `train.py`. SageMaker does not error if the paths do not match -- it simply has nothing to checkpoint. |
| Auto-scaling does not scale out during testing | Target tracking responds to sustained load, not single requests. Send continuous traffic for at least 3 minutes above the target to trigger scaling. |
| Inference Recommender job takes too long | Default jobs can take 15-45 minutes. Start the job early and review results later. Show pre-computed results if needed. |
| VPC training job fails with download error | The training container cannot reach ECR to pull the image. Create an ECR VPC Endpoint or ensure the NAT Gateway is configured correctly. |
| KMS AccessDeniedException | The SageMaker execution role must be listed as a key user in the KMS key policy. Add it via the KMS console. |
| VPC Endpoint creation fails | Verify the VPC has DNS resolution and DNS hostnames enabled. Interface Endpoints require both settings. |
| Associates forget VPC Endpoint cleanup | VPC Endpoints bill ~$0.01/hour each. With 6 endpoints, that is $43/month if forgotten. Check the VPC Endpoints page explicitly during cleanup. |
| CloudWatch metrics not appearing for training job | Metrics are published only after the training job starts processing. If the job is in "Starting" or "Downloading" phase, metrics do not appear yet. |
| S3 bucket policy blocks access after adding VPC condition | If you add a VPC Endpoint condition to the bucket policy, ensure the condition uses the correct VPC Endpoint ID. A typo locks everyone out. Always test access before applying restrictive policies. |
