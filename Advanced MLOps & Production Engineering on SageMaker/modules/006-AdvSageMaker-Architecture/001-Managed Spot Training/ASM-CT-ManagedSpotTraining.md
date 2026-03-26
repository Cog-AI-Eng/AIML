# Managed Spot Training

**Estimated Time:** 10 Minutes

## Introduction

SageMaker training jobs run on EC2 instances billed by the second. For long-running jobs (hours of XGBoost tuning, deep learning training), compute costs can add up quickly. AWS offers **Spot Instances** -- spare EC2 capacity available at up to 90% discount compared to on-demand pricing. The trade-off: AWS can reclaim (interrupt) a Spot Instance with two minutes' notice when demand for that capacity increases.

SageMaker **Managed Spot Training** integrates Spot Instances directly into training jobs with built-in checkpointing and automatic retry, removing the operational complexity of managing Spot interruptions yourself. This reading covers how Managed Spot Training works, how to enable it, and how checkpointing protects against data loss during interruptions.

## Core Concepts

### How Managed Spot Training works

1. When you create a training job with Spot enabled, SageMaker requests Spot capacity instead of on-demand instances.
2. If Spot capacity is available, the job starts immediately at the discounted price.
3. If Spot capacity is not available, SageMaker waits up to the **maximum wait time** you configure.
4. If a Spot interruption occurs during training, SageMaker:
   - Saves the current state to the **checkpoint S3 location** (if checkpointing is configured).
   - Waits for new Spot capacity.
   - Resumes training from the last checkpoint.
5. SageMaker tracks both **billable time** (actual training compute) and **total elapsed time** (including wait and interruption periods).

### Enabling Managed Spot Training in the console

1. Navigate to **SageMaker > Training > Training jobs > Create training job**.
2. Under **Resource configuration**:
   - **Enable managed spot training:** Check the box.
   - **Maximum wait time:** Set the maximum total time (in seconds) the job is allowed to run, including waiting for capacity and interruption periods. Must be greater than the estimated training time.
   - **Maximum run time:** The maximum training compute time allowed.
3. Under **Checkpoint configuration:**
   - **S3 URI:** Specify where checkpoints should be stored (e.g., `s3://bucket/checkpoints/`).
   - **Local path:** The local path inside the training container where checkpoints are written (default: `/opt/ml/checkpoints/`).

### Checkpointing

Checkpointing is the mechanism that makes Spot Training practical. Without it, a Spot interruption after 3 hours of training would lose all progress.

**For built-in algorithms (XGBoost, K-Means, etc.):** SageMaker handles checkpointing automatically. The algorithm saves model state at regular intervals.

**For Script Mode training:** Your training script must implement checkpointing explicitly:
- At regular intervals during training, save model weights, optimizer state, and current epoch/step to the checkpoint directory (`/opt/ml/checkpoints/`).
- At training start, check if a checkpoint exists and resume from it rather than starting from scratch.

SageMaker automatically syncs the local checkpoint directory to the S3 checkpoint location, so checkpoints survive instance termination.

### Cost savings calculation

SageMaker reports the cost savings of Managed Spot Training in the training job details:

- **Billable seconds:** The actual compute time used for training.
- **Training elapsed seconds:** Total wall-clock time including waits and interruptions.
- **Savings percentage:** Typically 60-90% depending on instance type and region. The console shows this directly on the training job detail page.

### When to use Managed Spot Training

| Scenario | Spot recommended | Why |
| :--- | :--- | :--- |
| Long training jobs (> 1 hour) | Yes | High potential savings |
| HPO jobs with many trials | Yes | Many parallel short jobs tolerate interruptions well |
| Time-critical training | No | Wait times and interruptions add unpredictable delays |
| Training jobs under 10 minutes | Marginal | Savings are small and checkpoint overhead may exceed benefit |
| Deep learning with expensive GPUs | Yes | GPU Spot discounts can save hundreds of dollars per job |

### Limitations

- **No guaranteed capacity:** Spot instances may not be available in your chosen instance type and region. Set a reasonable `MaxWaitTimeInSeconds` to fail gracefully if capacity never becomes available.
- **Interruption overhead:** Resuming from a checkpoint adds time (downloading checkpoint, reinitializing). Frequent interruptions can make total elapsed time much longer than on-demand.
- **Not all algorithms checkpoint equally:** Custom scripts must implement checkpoint/resume logic. Without it, interruptions lose all progress.

## Connecting to Practice

Managed Spot Training is the first of several cost optimization patterns in this module. The next topic, *Spot Instances for HPO*, shows how to combine Spot with HPO jobs for maximum cost reduction across many trials. The module assignment will require you to run a training job with Managed Spot Training enabled, demonstrate checkpoint recovery, and calculate the cost savings.

## Further Learning & Resources

**Documentation and reading**

- **[Managed Spot Training](https://docs.aws.amazon.com/sagemaker/latest/dg/model-managed-spot-training.html)** - *Docs*: Complete reference for Spot Training configuration, checkpointing, and cost reporting.
- **[Spot Instance Pricing](https://aws.amazon.com/ec2/spot/pricing/)** - *Docs*: Current Spot pricing by instance type and region.

**Interactive practice**

- **[Spot Training Example](https://github.com/aws/amazon-sagemaker-examples/tree/main/sagemaker-fundamentals/managed-spot-training)** - *Interactive*: Sample notebook demonstrating Managed Spot Training with checkpoint configuration.
