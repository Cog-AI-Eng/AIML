# Spot Instances for HPO

**Estimated Time:** 10 Minutes

## Introduction

HPO jobs are inherently expensive: they launch dozens or hundreds of individual training jobs to explore the hyperparameter space. Each trial is a separate training job with its own instance. Combining HPO with Managed Spot Training multiplies the savings -- every trial runs at a 60-90% discount. Since HPO trials are independent and many are exploratory (testing unpromising configurations), the cost of an interrupted trial is low: the optimizer simply learns that the trial was inconclusive and moves on.

This reading covers how to configure Spot Instances within HPO jobs, how interruptions affect optimization strategies, and best practices for maximizing cost efficiency.

## Core Concepts

### Enabling Spot in an HPO job

When creating an HPO job, Spot Training is configured in the **training job definition** (the base configuration that each trial uses):

1. In the console: **SageMaker > Training > Hyperparameter tuning jobs > Create**.
2. In the training job definition section, enable **Managed Spot Training** and set the maximum wait time and checkpoint configuration (same as standalone Spot Training).
3. All trials launched by the HPO job will use Spot instances.

### Impact on optimization strategies

Spot interruptions interact differently with each HPO strategy:

**Random Search:** No impact. Trials are independent; an interrupted trial is simply a failed data point. The optimizer generates a new random configuration for the replacement trial.

**Bayesian Optimization:** Minimal impact. Interrupted trials that complete partially may still provide useful metric data (if checkpointed and resumed). Trials that fail entirely are excluded from the Bayesian model. Bayesian optimization is robust to missing trials because it maintains uncertainty estimates.

**Hyperband:** Moderate impact. Hyperband's successive halving depends on comparing all trials in a bracket. If a trial is interrupted during an early round, it may be eliminated from the bracket even if it would have been competitive after resuming.

**Grid Search:** No impact. Each grid point is independent. Interrupted trials are retried.

### Cost calculation for HPO with Spot

An HPO job with 50 trials, each training for 30 minutes on `ml.m5.xlarge`:

| Configuration | Estimated Cost |
| :--- | :--- |
| On-demand | 50 trials x 0.5 hours x $0.23/hr = $5.75 |
| Spot (~70% savings) | 50 trials x 0.5 hours x $0.07/hr = $1.75 |

For larger HPO jobs with GPU instances (e.g., 100 trials on `ml.p3.2xlarge` at $3.06/hr on-demand vs. ~$0.92/hr Spot), the savings run into hundreds of dollars per HPO run.

### Best practices

- **Set generous `MaxWaitTimeInSeconds`:** HPO trials are typically not time-critical. Set the wait time to 2-3x the expected training time per trial to tolerate capacity delays.
- **Enable checkpointing:** Even for short trials, checkpointing ensures partial progress is preserved across interruptions.
- **Limit `MaxParallelTrainingJobs`:** More parallel trials means more simultaneous Spot requests, which can exceed available Spot capacity in a region. Keep parallel trials moderate (4-10) to improve capacity availability.
- **Use multiple instance types:** If your algorithm supports it, configure the HPO job to use a set of equivalent instance types (e.g., `ml.m5.xlarge` and `ml.m5a.xlarge`). SageMaker selects whichever has Spot availability.

### Monitoring HPO Spot savings

After the HPO job completes, check the individual trial details in the console. Each trial shows billable time vs. elapsed time, and the training job detail page shows the Spot savings percentage. Aggregate across all trials to calculate total HPO savings.

## Connecting to Practice

Combining Spot with HPO is the most impactful cost optimization for the training phase. The next topic, *Instance Right-sizing*, covers choosing the optimal instance type for different workload profiles. The module assignment will require you to run an HPO job with Spot enabled and compare the total cost against the same job run on-demand.

## Further Learning & Resources

**Documentation and reading**

- **[Managed Spot Training with HPO](https://docs.aws.amazon.com/sagemaker/latest/dg/model-managed-spot-training.html)** - *Docs*: Configuration reference for Spot Training in HPO jobs.
- **[HPO Best Practices](https://docs.aws.amazon.com/sagemaker/latest/dg/automatic-model-tuning-considerations.html)** - *Docs*: Guidance on trial counts, parallelism, and cost management.

**Interactive practice**

- **[HPO with Spot Training Example](https://github.com/aws/amazon-sagemaker-examples/tree/main/hyperparameter_tuning)** - *Interactive*: Sample notebooks demonstrating HPO configurations including Spot instances.
