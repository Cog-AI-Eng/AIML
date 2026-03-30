# Week 3 Monday -- Advanced SageMaker Architecture, Cost Optimization, and Security

**Total Duration:** 185 Minutes (3 Stages)
**Consolidated Activities:**
- SM Cost Optimization: Managed Spot Training, Spot Instances for HPO, Instance Right-Sizing, Auto-Scaling Policies, Inference Recommender, CloudWatch Dashboards
- SM Security & Governance: VPC/PrivateLink/KMS, IAM Least-Privilege, Full ML Lifecycle Review

| Block | Content | Minutes |
|-------|---------|---------|
| Stage 1 | Cost Optimization for Training: Spot Instances, Checkpointing, HPO + Spot, Instance Right-Sizing | 55 |
| Break 1 | Stretch / Questions | 5 |
| Stage 2 | Cost Optimization for Inference: Auto-Scaling, Inference Recommender, CloudWatch, Cost Comparison | 55 |
| Break 2 | Stretch / Questions | 5 |
| Stage 3 | Security and Governance: VPC, PrivateLink, KMS, IAM, Full Lifecycle Review | 55 |
| Buffer | Open Q&A, Summary, Tuesday Preview | 10 |

---

## Lecture Overview

**Unified Scenario -- FraudShield Risk Analytics**

On Friday you deployed models across four inference patterns and set up monitoring. But someone has to pay the AWS bill. Today we learn how to slash training costs by up to 90% with Spot instances, right-size inference endpoints, set up auto-scaling, and lock down the environment with VPC isolation and KMS encryption. This is the operations and security layer that makes ML production-ready.

Associates continue as ML engineers at FraudShield. Friday's inference patterns and monitoring infrastructure are in place. Now we answer the operations questions:

1. **"How do we reduce training costs without sacrificing quality?"** (Managed Spot Training, checkpointing, spot HPO)
2. **"How do we right-size and auto-scale inference endpoints?"** (Instance families, auto-scaling policies, Inference Recommender)
3. **"How do we monitor costs and performance in real time?"** (CloudWatch dashboards, cost comparison across patterns)
4. **"How do we lock down the ML environment for production?"** (VPC isolation, PrivateLink, KMS encryption, IAM least-privilege)

Each stage builds on the previous: reduce costs for training, reduce costs for inference, then secure the entire pipeline.

---

## Pre-Lecture Setup

### Instructor Checklist

- [ ] Friday's inference patterns and monitoring concepts reviewed
- [ ] SageMaker execution role ARN ready
- [ ] Companion lecture notebook (`W3-Monday-notebook.ipynb`) open and tested
- [ ] AWS account with SageMaker access verified
- [ ] Budget verified and active
- [ ] S3 bucket and prefix configured for checkpoint output
- [ ] KMS key created (or plan to create during lecture) for encryption demonstrations
- [ ] This instructor guide open in a second tab

### Student Prerequisites

- [ ] Completed readings: Managed Spot Training CT, Spot Instances for HPO CT, Instance Right-Sizing CT, Auto-Scaling Policies CT, Inference Recommender CT, CloudWatch Dashboards CT, VPC/PrivateLink/KMS CT
- [ ] Friday's notebook completed (inference patterns deployed and cleaned up, monitoring concepts understood)
- [ ] AWS credentials configured, SageMaker SDK installed
- [ ] Familiarity with deployment patterns from Friday and training workflows from Thursday

---

# STAGE 1 -- Cost Optimization for Training (55 min)

> **Goal:** Understand how Managed Spot Training can reduce training costs by up to 90%, configure checkpointing for fault tolerance, combine spot instances with HPO, and choose the right instance type for each workload.

**Exit Criteria Addressed:**
- Configure a SageMaker training job with Managed Spot Training and explain the cost savings mechanism (Required)
- Set up S3-based checkpointing to handle spot interruptions gracefully (Required)
- Combine spot instances with hyperparameter tuning for maximum cost efficiency (Required)
- Select the appropriate instance family (general, compute, GPU, memory) for a given ML workload (Required)

### Instructor Opening (3 minutes -- talk, no code)

> "Friday you deployed models across real-time, serverless, async, and batch patterns. You set up monitoring to catch drift. That is the operational side of deployment. But there is another operational dimension we have not addressed: cost. A single ml.p3.2xlarge instance costs $3.83 per hour. If you run HPO with 20 jobs, that is $76 per tuning run. What if I told you that you could cut that to $7.60? That is what Managed Spot Training does."

---

## STEP 1 -- What Are Spot Instances? (8 minutes)

**Pacing: conceptual with notebook markdown. Establish the foundational concept.**

> "Spot instances are unused EC2 capacity that AWS sells at a steep discount -- up to 90% off on-demand pricing. The trade-off: AWS can reclaim the instance with a two-minute warning when it needs the capacity back. For web servers, that is risky. For ML training, it is manageable -- because training jobs can checkpoint and resume."

Key concepts:
- On-demand instances: guaranteed availability, full price
- Spot instances: interruptible, up to 90% discount
- SageMaker Managed Spot Training: SageMaker handles the spot lifecycle automatically
- You configure: `use_spot_instances=True`, `max_wait`, `max_run`

| Parameter | Purpose | Typical Value |
|-----------|---------|---------------|
| `use_spot_instances` | Enable spot training | `True` |
| `max_run` | Maximum seconds the training job can run | `1800` (30 min) |
| `max_wait` | Maximum seconds to wait for spot capacity (includes `max_run`) | `3600` (1 hr) |
| `checkpoint_s3_uri` | S3 path for saving checkpoints | `s3://<bucket>/checkpoints/` |

> "`max_wait` must be greater than or equal to `max_run`. The difference (`max_wait - max_run`) is the maximum time SageMaker will wait for spot capacity to become available. If spot capacity is not available within `max_wait`, the job fails."

**Discussion Prompt:** "Why must `max_wait` be at least as large as `max_run`?" (The job needs enough total time to both wait for capacity and run the training. If `max_wait` equals `max_run`, there is zero tolerance for waiting -- the job must start immediately or fail.)

---

## STEP 2 -- Managed Spot Training with XGBoost (12 minutes)

**Pacing: live code in notebook.**

> "Let us configure a spot training job for our FraudShield XGBoost model. We add three parameters to the Estimator: `use_spot_instances`, `max_wait`, `max_run`, and a checkpoint configuration."

Walk through the Estimator configuration:
- `use_spot_instances=True`
- `max_run=1800` (30 minutes -- more than enough for our small dataset)
- `max_wait=3600` (1 hour total -- allows 30 minutes of waiting for capacity)
- `checkpoint_s3_uri` pointing to S3

Run the training job and highlight the output:

> "Look at the training job output. SageMaker reports the managed spot training savings. It shows you the training time, the billable time, and the percentage saved. For our small job, the savings percentage may vary, but for large jobs running hours on GPU instances, this routinely saves 60-90%."

After the job completes, show the savings calculation:

```
Training seconds:        120
Billable seconds:        36   (spot discount applied)
Managed Spot Savings:    70%
```

> "The savings scale with job duration and instance cost. A 4-hour training run on an ml.p3.8xlarge at $14.68/hour saves you roughly $40 per run at a 70% discount. Multiply that across weekly retraining and HPO experiments."

---

## STEP 3 -- Checkpointing for Fault Tolerance (10 minutes)

**Pacing: conceptual explanation then live code.**

> "Spot instances can be interrupted. Without checkpointing, the entire training run starts over. With checkpointing, the training resumes from the last saved checkpoint. This is the insurance policy that makes spot training practical."

How checkpointing works:
1. Training framework periodically saves model state to a local directory (`/opt/ml/checkpoints/`)
2. SageMaker syncs this directory to S3 automatically
3. If the instance is interrupted, SageMaker provisions a new spot instance
4. SageMaker downloads the latest checkpoint from S3 to the new instance
5. Training resumes from where it left off

Key points:
- XGBoost built-in algorithm handles checkpointing automatically
- For custom frameworks (PyTorch, TensorFlow), you must implement checkpoint save/load in your training script
- Checkpoint frequency is a trade-off: too frequent wastes I/O, too infrequent loses more progress on interruption

> "For XGBoost, checkpointing is automatic -- the built-in algorithm saves after each boosting round. For deep learning frameworks, you write the checkpoint logic yourself. We will not cover custom checkpointing today, but the principle is the same: save state periodically, resume from the latest save."

Show the checkpoint files in S3 after the training job completes.

---

## STEP 4 -- Spot Instances for Hyperparameter Optimization (10 minutes)

**Pacing: conceptual with code configuration.**

> "Thursday you ran HPO with 10-20 training jobs. Each job was an on-demand instance. Now imagine running those same jobs with spot instances. Each individual job saves 60-90%, and because HPO jobs are independent of each other, a spot interruption on one job does not affect the others. This is where spot savings compound."

Key configuration:
- The Estimator passed to `HyperparameterTuner` has `use_spot_instances=True`
- Each tuning trial runs as a separate spot training job
- If one trial is interrupted, it restarts independently
- The tuner itself is not affected by individual trial interruptions

Show the conceptual configuration in the notebook:

```python
estimator = Estimator(
    ...,
    use_spot_instances=True,
    max_wait=3600,
    max_run=1800,
    checkpoint_s3_uri=checkpoint_s3_uri,
)

tuner = HyperparameterTuner(
    estimator=estimator,
    ...,
    max_jobs=20,
    max_parallel_jobs=4,
)
```

> "With 20 HPO trials on ml.p3.2xlarge, on-demand would cost roughly $76. With spot at 70% savings, that drops to about $23. For large-scale HPO with hundreds of trials, spot is practically mandatory."

**Discussion Prompt:** "What happens if spot capacity is unavailable for an extended period during HPO?" (Individual trials fail if they exceed `max_wait`. The tuner marks those trials as failed and continues with the remaining trials. You may get fewer completed trials than planned, but the best result from completed trials is still valid.)

---

## STEP 5 -- Instance Right-Sizing (12 minutes)

**Pacing: conceptual with reference tables in notebook markdown.**

> "Spot savings are powerful, but they compound when combined with right-sizing. If you are training on an ml.p3.8xlarge when an ml.m5.xlarge would suffice, you are wasting money even with spot discounts. The right instance family depends on the workload."

Present the instance family guide:

| Family | Optimized For | Use When | Example Instance | On-Demand Cost (approx) |
|--------|--------------|----------|-----------------|------------------------|
| **ml.m5** | General purpose | Tabular data, XGBoost, small models, preprocessing | ml.m5.xlarge | ~$0.27/hr |
| **ml.c5** | Compute | CPU-intensive feature engineering, ensemble models | ml.c5.2xlarge | ~$0.47/hr |
| **ml.p3** | GPU (training) | Deep learning training, CNNs, transformers, large models | ml.p3.2xlarge | ~$3.83/hr |
| **ml.g4dn** | GPU (inference) | GPU inference at lower cost than p3, small to medium models | ml.g4dn.xlarge | ~$0.74/hr |
| **ml.r5** | Memory | Large datasets that must fit in memory, embeddings, NLP | ml.r5.xlarge | ~$0.36/hr |

> "Notice the 14x cost difference between ml.m5.xlarge and ml.p3.2xlarge. If your workload is tabular data with XGBoost, the GPU is literally idle -- you are paying 14x more for hardware you are not using."

Decision framework for choosing an instance type:

```
Is the model a deep learning model (CNN, RNN, Transformer)?
  YES --> Is this training or inference?
            Training --> ml.p3 (or ml.p4d for very large models)
            Inference --> ml.g4dn (cheaper GPU, optimized for inference)
  NO  --> Does the data fit in memory on a general-purpose instance?
            NO  --> ml.r5 (memory-optimized)
            YES --> Is the workload CPU-bound (heavy preprocessing)?
                      YES --> ml.c5 (compute-optimized)
                      NO  --> ml.m5 (general purpose)
```

> "For FraudShield's XGBoost model, ml.m5.xlarge is the right choice. XGBoost does not use GPUs. The dataset fits in memory. An ml.p3.2xlarge would work but waste GPU resources."

**Discussion Prompt:** "FraudShield is adding a deep learning fraud detection model alongside XGBoost. It uses a transformer architecture trained on transaction sequences. Which instance would you choose for training? For inference?" (Training: ml.p3.2xlarge or ml.p3.8xlarge depending on model size. Inference: ml.g4dn.xlarge -- cheaper GPU optimized for inference workloads.)

---

## Stage 1 Summary (3 minutes)

> "Three levers to reduce training costs: Spot instances save 60-90% on compute. Checkpointing makes spot practical by enabling resume on interruption. Right-sizing ensures you are not paying for hardware you do not use. Combined, these can cut your training bill by an order of magnitude."

[PAUSE FOR BREAK - 5 MINS]

---

# STAGE 2 -- Cost Optimization for Inference (55 min)

> **Goal:** Configure auto-scaling policies for SageMaker endpoints, use Inference Recommender to benchmark instance selection, build CloudWatch dashboards for operational visibility, and compare costs across inference patterns.

**Exit Criteria Addressed:**
- Configure target tracking and step scaling policies for SageMaker endpoints (Required)
- Use Inference Recommender to benchmark model performance on different instance types (Required)
- Create a CloudWatch dashboard with key endpoint metrics (Required)
- Compare cost profiles across real-time, serverless, async, and batch inference patterns (Required)

### Instructor Opening (3 minutes -- talk, no code)

> "Stage 1 reduced training costs. But training is a one-time expense per model version. Inference is a continuous expense -- your endpoint runs 24/7. A single ml.m5.xlarge endpoint costs about $200/month. If traffic doubles, you need two instances. If traffic drops at night, one instance sits idle. Auto-scaling matches capacity to demand so you pay for what you use."

---

## STEP 6 -- Auto-Scaling Concepts (8 minutes)

**Pacing: conceptual with notebook markdown.**

SageMaker endpoint auto-scaling uses the Application Auto Scaling service. Two policy types:

| Policy Type | How It Works | Best For |
|-------------|-------------|----------|
| **Target Tracking** | Maintains a metric at a target value (like a thermostat) | Most workloads -- simple, effective |
| **Step Scaling** | Adds/removes instances in steps based on CloudWatch alarm thresholds | Fine-grained control, complex traffic patterns |

Target tracking key metric: `SageMakerVariantInvocationsPerInstance`
- This is the average number of invocations per instance per minute
- SageMaker scales out when the metric exceeds the target, scales in when it drops below

> "Think of target tracking like a thermostat. You set the temperature (target invocations per instance) and the system adjusts the heating (instance count) to maintain it. Step scaling is like manual HVAC controls -- you define exactly what happens at each temperature threshold."

Configuration parameters:
- `MinCapacity`: minimum number of instances (never scale below this)
- `MaxCapacity`: maximum number of instances (budget protection)
- `TargetValue`: the target invocations per instance per minute
- `ScaleInCooldown`: seconds to wait before scaling in again (prevents flapping)
- `ScaleOutCooldown`: seconds to wait before scaling out again

---

## STEP 7 -- Configure Target Tracking Auto-Scaling (10 minutes)

**Pacing: live code in notebook.**

> "Let us configure target tracking scaling for a FraudShield endpoint. We register the endpoint as a scalable target, then create a scaling policy."

Walk through the code:
1. Register the scalable target with Application Auto Scaling
2. Create a target tracking scaling policy
3. Explain each parameter

> "We set the target to 100 invocations per instance per minute. If average invocations exceed 100, SageMaker adds instances. If they drop below 100, SageMaker removes instances (down to MinCapacity). The cooldown periods prevent the system from scaling too aggressively."

After running the code:

> "The auto-scaling policy is now attached to the endpoint variant. In production, you would load test the endpoint to find the optimal target value. Too low means you over-provision (wasteful). Too high means you under-provision (latency spikes)."

---

## STEP 8 -- Step Scaling Alternative (7 minutes)

**Pacing: conceptual with code configuration.**

> "Step scaling gives you more granular control. Instead of a single target, you define thresholds and the corresponding scaling action at each threshold."

Example step scaling configuration:

```
0-50 invocations/min   --> 1 instance (baseline)
50-150 invocations/min  --> add 1 instance
150-300 invocations/min --> add 2 instances
300+ invocations/min    --> add 3 instances
```

> "Step scaling requires a CloudWatch alarm as the trigger. You create the alarm, then define the step adjustments. It is more work to configure than target tracking, but it gives you precise control over scaling behavior."

Show the code configuration but note:

> "For most SageMaker workloads, target tracking is sufficient and simpler. Use step scaling when you have well-understood traffic patterns with distinct tiers, or when target tracking's scaling behavior is too aggressive or too conservative for your needs."

---

## STEP 9 -- Inference Recommender (10 minutes)

**Pacing: conceptual with code structure.**

> "You know you need auto-scaling. But on which instance type? Inference Recommender answers this question empirically. It benchmarks your model on multiple instance types and reports latency, throughput, and cost for each."

How Inference Recommender works:
1. You provide a model artifact (in Model Registry or as a `model.tar.gz`)
2. You specify a sample payload
3. SageMaker deploys the model on multiple instance types
4. It runs benchmark invocations on each
5. It returns a ranked list of recommendations with latency, throughput, and cost

Two modes:
- **Default job**: SageMaker picks instance types to benchmark (broader exploration)
- **Advanced job**: You specify the instance types and traffic pattern

> "Inference Recommender takes 30-45 minutes for a default job. We will show the API structure and interpret sample results conceptually rather than waiting for a full run."

Show the API call structure and a conceptual results table:

| Instance Type | Invocations/min | Model Latency (ms) | Cost/hr | Cost/1M invocations |
|---------------|----------------|--------------------|---------|--------------------|
| ml.m5.xlarge | 600 | 12 | $0.134 | $3.72 |
| ml.m5.xlarge | 1200 | 8 | $0.269 | $3.74 |
| ml.c5.large | 800 | 9 | $0.119 | $2.48 |
| ml.g4dn.xlarge | 2000 | 4 | $0.736 | $6.13 |

> "For our XGBoost model, ml.c5.large offers the best cost per invocation. The GPU instance (ml.g4dn) has the lowest latency but the highest cost per invocation -- because XGBoost does not use the GPU. Inference Recommender surfaces these trade-offs with real data from your model."

**Discussion Prompt:** "When would you choose the GPU instance despite higher cost per invocation?" (When latency is the primary constraint -- 4ms vs 12ms matters for real-time fraud scoring at checkout. Cost efficiency and latency are different optimization targets.)

---

## STEP 10 -- CloudWatch Dashboards (10 minutes)

**Pacing: live code in notebook.**

> "Auto-scaling and right-sizing are great, but you need visibility into what is happening. CloudWatch dashboards give you real-time metrics for your SageMaker endpoints."

Key metrics for SageMaker endpoints:

| Metric | What It Measures | Why It Matters |
|--------|-----------------|---------------|
| `Invocations` | Total invocations per period | Traffic volume |
| `ModelLatency` | Time the model takes to respond (ms) | Model performance |
| `OverheadLatency` | SageMaker overhead beyond model latency (ms) | Infrastructure health |
| `CPUUtilization` | CPU usage percentage | Right-sizing signal |
| `MemoryUtilization` | Memory usage percentage | Right-sizing signal |
| `InvocationsPerInstance` | Invocations per instance per minute | Auto-scaling signal |

Walk through the code that creates a CloudWatch dashboard with these metrics.

> "A well-designed dashboard answers three questions at a glance: Is the endpoint healthy? Is it right-sized? Is auto-scaling working? You should create one as part of every production deployment."

---

## STEP 11 -- Cost Comparison Across Inference Patterns (7 minutes)

**Pacing: conceptual, notebook markdown table.**

> "Friday you learned five inference patterns. Now let us compare them on cost. The right pattern can save you thousands per month."

| Pattern | Billing Model | Idle Cost | Estimated Monthly Cost (500 TPS avg) | Best For |
|---------|--------------|-----------|--------------------------------------|----------|
| **Real-Time** | Per instance-hour | Full instance cost | ~$200-400 (1-2 ml.m5.xlarge) | Steady, latency-sensitive traffic |
| **Serverless** | Per ms of compute | $0 (scale to zero) | ~$50-150 (depends on invocation duration) | Bursty traffic, dev/test |
| **Async** | Per instance-hour | Full instance cost (unless custom scale-to-zero) | ~$200 (1 ml.m5.xlarge) | Large payloads, tolerates delay |
| **Batch Transform** | Per instance-hour (job duration only) | $0 (ephemeral) | ~$5-20 per job | Bulk scoring, scheduled runs |
| **Multi-Model** | Per instance-hour (shared) | Full instance cost (shared across models) | ~$100-200 (1 instance, many models) | Many similar models |

> "The key insight: match the billing model to the traffic pattern. Steady traffic with low latency requirements? Real-time with auto-scaling. Bursty traffic with idle periods? Serverless. Bulk scoring? Batch transform. Many models with moderate traffic each? Multi-model. The wrong pattern wastes money; the right pattern optimizes it."

**Discussion Prompt:** "FraudShield has steady traffic during business hours (8 AM - 8 PM) and near-zero traffic overnight. What combination of patterns would minimize cost?" (Real-time with aggressive auto-scaling, or a scheduled switch between real-time and serverless. Discuss the operational complexity of switching patterns vs the cost savings.)

[PAUSE FOR BREAK - 5 MINS]

---

# STAGE 3 -- Security and Governance (55 min)

> **Goal:** Configure VPC isolation for SageMaker workloads, understand PrivateLink, encrypt data with KMS at rest and in transit, review IAM least-privilege patterns, and tie the entire curriculum together in a full ML lifecycle review.

**Exit Criteria Addressed:**
- Configure VPC subnets and security groups for SageMaker training jobs and endpoints (Required)
- Explain how PrivateLink keeps SageMaker API traffic off the public internet (Required)
- Configure KMS encryption for S3 data, training job output, and endpoint storage (Required)
- Apply IAM least-privilege principles to SageMaker roles and policies (Required)
- Map the full ML lifecycle (Prepare through Monitor) to SageMaker services covered in the curriculum (Required)

### Instructor Opening (3 minutes -- talk, no code)

> "Cost optimization handles the 'how much does it cost' question. Security handles the 'who can access it and how is it protected' question. For FraudShield, we are processing financial transaction data. Regulators -- PCI DSS, SOC 2, GDPR -- require encryption at rest and in transit, network isolation, and audit trails. A model that is cheap but insecure is a compliance violation waiting to happen."

---

## STEP 12 -- VPC Configuration for SageMaker (12 minutes)

**Pacing: conceptual with code parameters.**

> "By default, SageMaker training jobs and endpoints run in an AWS-managed VPC. They can access the internet, S3, and other AWS services directly. For regulated workloads, this is not acceptable. You need the workloads inside your own VPC, in private subnets, with no internet access."

Why VPC isolation matters:
- Data does not traverse the public internet
- Network traffic is controlled by security groups and NACLs
- Meets compliance requirements for network segmentation
- Limits blast radius if credentials are compromised

VPC configuration for training jobs:

```python
estimator = Estimator(
    ...,
    subnets=["subnet-abc123", "subnet-def456"],
    security_group_ids=["sg-789xyz"],
)
```

VPC configuration for endpoints:

```python
model.deploy(
    ...,
    vpc_config={
        "Subnets": ["subnet-abc123", "subnet-def456"],
        "SecurityGroupIds": ["sg-789xyz"],
    },
)
```

Key considerations:
- Subnets must be private (no internet gateway route)
- Security groups control inbound/outbound traffic
- SageMaker needs access to S3, ECR, and CloudWatch -- use VPC endpoints (PrivateLink)
- Without VPC endpoints, jobs in a private subnet cannot pull containers or write logs

> "The most common mistake is putting SageMaker in a VPC without VPC endpoints. The training job starts, tries to pull the container image from ECR, cannot reach ECR because there is no internet access, and times out. Always configure VPC endpoints for S3, ECR, CloudWatch, and SageMaker API."

---

## STEP 13 -- PrivateLink (8 minutes)

**Pacing: conceptual, notebook markdown.**

> "PrivateLink creates a private connection between your VPC and AWS services. Instead of traffic going: VPC -> Internet Gateway -> AWS Service, it goes: VPC -> VPC Endpoint -> AWS Service. The traffic never touches the public internet."

VPC endpoints needed for SageMaker in a private subnet:

| Service | Endpoint Type | Purpose |
|---------|--------------|---------|
| S3 | Gateway endpoint | Read/write training data, model artifacts, checkpoints |
| ECR (api + dkr) | Interface endpoint | Pull container images for training and inference |
| CloudWatch Logs | Interface endpoint | Write training logs and monitoring logs |
| SageMaker API | Interface endpoint | API calls (create training job, deploy endpoint) |
| SageMaker Runtime | Interface endpoint | Invoke endpoint (inference requests) |
| STS | Interface endpoint | Assume IAM roles |
| KMS | Interface endpoint | Encrypt/decrypt with customer-managed keys |

> "Gateway endpoints (S3) are free. Interface endpoints cost about $0.01/hour per AZ. For a production deployment with 6-7 interface endpoints across 2 AZs, that is roughly $100/month. This is the cost of network security."

**Discussion Prompt:** "Is the $100/month for VPC endpoints worth it for FraudShield?" (Absolutely. The alternative is processing financial data over the public internet, which is a PCI DSS violation. The cost is trivial compared to the compliance risk.)

---

## STEP 14 -- KMS Encryption at Rest and in Transit (12 minutes)

**Pacing: conceptual then live code.**

> "Encryption at rest protects data stored in S3, EBS volumes, and SageMaker storage. Encryption in transit protects data moving between services. Both are required for PCI DSS and SOC 2 compliance."

**At rest -- what gets encrypted:**

| Storage Location | Default Encryption | Customer-Managed KMS |
|-----------------|-------------------|---------------------|
| S3 (training data, model artifacts) | SSE-S3 (AES-256) | SSE-KMS with custom key |
| EBS volumes (training instances) | AWS-managed key | Customer-managed KMS key |
| SageMaker storage (notebook, endpoint) | AWS-managed key | Customer-managed KMS key |

> "Default encryption uses AWS-managed keys. You get encryption, but you do not control key rotation, key policies, or access logging. Customer-managed KMS keys give you full control -- you define who can use the key, you set rotation schedules, and CloudTrail logs every key usage."

**In transit:**
- All SageMaker API calls use TLS 1.2
- S3 transfers use HTTPS by default
- Inter-container traffic (distributed training) can be encrypted with `encrypt_inter_container_traffic=True`
- This adds latency (~5-10%) but ensures data is encrypted between training instances

Show the code that configures KMS encryption:

```python
kms_key_id = "arn:aws:kms:us-east-1:123456789012:key/abc-def-123"

estimator = Estimator(
    ...,
    output_kms_key=kms_key_id,
    volume_kms_key=kms_key_id,
    encrypt_inter_container_traffic=True,
)
```

```python
model.deploy(
    ...,
    kms_key=kms_key_id,
)
```

> "The `output_kms_key` encrypts the model artifact written to S3. The `volume_kms_key` encrypts the EBS volume attached to the training instance. `encrypt_inter_container_traffic` encrypts data moving between instances in distributed training."

---

## STEP 15 -- IAM Least-Privilege Review (8 minutes)

**Pacing: conceptual, notebook markdown table.**

> "The SageMaker execution role defines what SageMaker can do on your behalf. In development, we often use a broad role with full SageMaker and S3 access. In production, least-privilege is mandatory."

Key SageMaker IAM permissions and what they allow:

| Permission | What It Allows | Least-Privilege Guidance |
|------------|---------------|------------------------|
| `sagemaker:CreateTrainingJob` | Start training jobs | Restrict to specific instance types and VPC configs |
| `sagemaker:CreateEndpoint` | Deploy endpoints | Restrict to specific endpoint config names |
| `sagemaker:InvokeEndpoint` | Call deployed models | Restrict to specific endpoint ARNs |
| `s3:GetObject` | Read from S3 | Restrict to specific bucket and prefix |
| `s3:PutObject` | Write to S3 | Restrict to specific output paths |
| `kms:Decrypt` / `kms:GenerateDataKey` | Use KMS keys | Restrict to specific key ARNs |
| `ecr:GetDownloadUrlForLayer` | Pull container images | Restrict to specific repositories |
| `logs:CreateLogGroup` / `PutLogEvents` | Write CloudWatch logs | Restrict to specific log groups |
| `iam:PassRole` | Pass the execution role to SageMaker | Restrict to specific role ARNs |

> "The most dangerous permission is `iam:PassRole`. If a user can pass any role to SageMaker, they can effectively escalate their own privileges. Always restrict `iam:PassRole` to specific role ARNs."

Production IAM pattern:
- Separate roles for training, deployment, and monitoring
- Condition keys to restrict instance types (prevent accidental ml.p4d.24xlarge usage)
- Resource-level permissions (specific S3 paths, specific endpoints)
- Service control policies (SCPs) at the organization level as guardrails

**Discussion Prompt:** "What could go wrong if the SageMaker execution role has `s3:*` on all buckets?" (The training job could read data from any bucket in the account, including buckets belonging to other teams. A bug or malicious code in the training script could exfiltrate data or overwrite critical files.)

---

## STEP 16 -- Full ML Lifecycle Review (12 minutes)

**Pacing: conceptual, notebook markdown. Walk through the full table.**

> "We have spent two and a half weeks building an ML pipeline from end to end. Let us map every stage of the ML lifecycle to the SageMaker services and concepts we have covered."

| Lifecycle Stage | What Happens | SageMaker Service / Feature | When We Covered It |
|----------------|-------------|----------------------------|-------------------|
| **Prepare** | Collect, clean, transform data | Data Wrangler, Feature Store, Processing Jobs | W2 Tuesday |
| **Build** | Explore data, select features, choose algorithm | Studio Notebooks, Canvas (no-code), Autopilot (AutoML) | W2 Tuesday |
| **Train** | Train model on data | Estimator, Built-in Algorithms, Managed Spot Training | W1 Friday, W2 Thursday, W3 Monday |
| **Tune** | Optimize hyperparameters | HyperparameterTuner (Bayesian, Random, Grid, Hyperband) | W2 Thursday |
| **Evaluate** | Assess model performance | Processing Jobs, Clarify (bias, explainability) | W2 Monday, W2 Wednesday |
| **Register** | Version and catalog models | Model Registry, Model Groups, Approval Status | W2 Wednesday |
| **Approve** | Human or automated approval gate | Model Registry Approval Status, CI/CD integration | W2 Wednesday |
| **Deploy** | Serve model for inference | Real-Time, Serverless, Async, Batch Transform, Multi-Model | W2 Monday, W2 Friday |
| **Optimize** | Reduce cost, right-size, auto-scale | Spot Training, Instance Right-Sizing, Auto-Scaling, Inference Recommender | W3 Monday |
| **Secure** | Encrypt, isolate, restrict access | VPC, PrivateLink, KMS, IAM Least-Privilege | W3 Monday |
| **Monitor** | Detect drift, track performance | Model Monitor, Data Quality, Bias Drift, EventBridge Automation | W2 Friday |
| **Retrain** | Update model with new data | Pipelines, EventBridge triggers, Spot Training | W2 Monday, W2 Friday, W3 Monday |

> "This is the complete cycle. Every stage connects to the next. The model you trained in Week 1 was deployed in Week 2, monitored in Week 2 Friday, and today you learned how to optimize and secure it. Tomorrow we shift from custom-trained models to foundation models and Amazon Bedrock -- a different paradigm, but the same lifecycle principles apply."

**Discussion Prompt:** "Which lifecycle stage is most often skipped in practice? Why?" (Monitoring. Teams deploy and move on to the next project. The model degrades silently. This is why automating monitoring and alerting is critical.)

---

## Cleanup (3 minutes)

**Pacing: live code. Mandatory.**

> "Delete any resources created during this session. Check for any endpoints, training jobs, or auto-scaling policies."

Run the cleanup cell. Verify no endpoints remain in the console.

---

## Wrap-up & Q&A Buffer (10 minutes)

### Summary (5 minutes)

> "Today you learned the operations and security layer that makes ML production-ready. Stage 1 showed you how to cut training costs by up to 90% with Managed Spot Training, checkpoint for fault tolerance, combine spot with HPO for compound savings, and right-size instances to avoid paying for idle hardware. Stage 2 showed you how to auto-scale inference endpoints with target tracking and step scaling, use Inference Recommender to benchmark instance selection, build CloudWatch dashboards for operational visibility, and compare cost profiles across inference patterns. Stage 3 locked down the environment: VPC isolation keeps traffic off the public internet, PrivateLink provides private connectivity to AWS services, KMS encryption protects data at rest and in transit, and IAM least-privilege restricts who can do what. Finally, we mapped the full ML lifecycle from Prepare to Monitor, connecting every concept from the past two and a half weeks."

### Tuesday Preview (2 minutes)

> "Tomorrow we shift paradigms. Instead of training our own models from scratch, we explore Amazon Bedrock and foundation models. The question changes from 'how do we train a model' to 'how do we leverage a pre-trained foundation model for our tasks.' Read the Bedrock CTs before Tuesday."

### Open Q&A (3 minutes)

---

## Instructor Notes -- Common Issues

| Issue | Resolution |
|-------|-----------|
| Spot training job fails immediately | Check that `max_wait >= max_run`. If spot capacity is unavailable, increase `max_wait` or try a different instance type. |
| No managed spot savings shown in output | The savings line only appears after the job completes. For very short jobs, savings may be minimal or the display may not appear. |
| Checkpoint files not in S3 | Verify `checkpoint_s3_uri` is set correctly. XGBoost built-in algorithm checkpoints automatically; custom scripts need explicit save logic. |
| Auto-scaling policy not triggering | Scaling requires sustained metric changes (not a single spike). Cooldown periods delay scaling actions. Load test with sustained traffic. |
| CloudWatch dashboard shows no data | Metrics take 1-2 minutes to populate. Ensure the endpoint name in the dashboard matches the deployed endpoint exactly. |
| Inference Recommender job takes too long | Default jobs take 30-45 minutes. Use the notebook's conceptual walkthrough and sample results instead of waiting. |
| VPC configuration causes training job timeout | Missing VPC endpoints. Add gateway endpoint for S3 and interface endpoints for ECR, CloudWatch, and SageMaker. |
| KMS key access denied | The SageMaker execution role needs `kms:Decrypt` and `kms:GenerateDataKey` permissions on the KMS key ARN. Update the key policy or IAM role. |
| Student deploys endpoint without VPC in security stage | This is fine for learning -- VPC config is shown conceptually. In production, enforce VPC via IAM condition keys. |
| `encrypt_inter_container_traffic` slows training | Expected -- encryption adds ~5-10% overhead. Explain this is the trade-off for data-in-transit security in distributed training. |
