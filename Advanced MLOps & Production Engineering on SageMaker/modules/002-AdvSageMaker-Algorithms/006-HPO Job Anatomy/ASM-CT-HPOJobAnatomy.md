# HPO Job Anatomy

**Estimated Time:** 10 Minutes

## Introduction

In the foundational SageMaker skill, you configured hyperparameters manually when creating training jobs -- setting `max_depth`, `learning_rate`, and `num_round` based on intuition or rough guidelines. In practice, finding the optimal combination of hyperparameters is a search problem: the space of possible configurations is enormous, and small changes can have significant effects on model performance.

SageMaker **Hyperparameter Tuning Jobs** (commonly called HPO jobs) automate this search. You define a range for each hyperparameter and an objective metric, and SageMaker systematically launches multiple training jobs with different configurations, evaluates the results, and identifies the best-performing combination. This reading covers the anatomy of an HPO job -- its components, lifecycle, and how it coordinates multiple training jobs.

## Core Concepts

### Components of an HPO job

An HPO job consists of four parts:

1. **Training job definition:** The base configuration for each trial -- algorithm, input data, instance type, and output location. Every trial uses this same base config but with different hyperparameter values.
2. **Hyperparameter ranges:** The search space. For each tunable hyperparameter, you define a range and type:
   - **ContinuousParameterRange:** Float values (e.g., `learning_rate` from 0.01 to 0.3).
   - **IntegerParameterRange:** Integer values (e.g., `max_depth` from 3 to 10).
   - **CategoricalParameterRange:** Discrete choices (e.g., `optimizer` from `["sgd", "adam"]`).
3. **Objective metric:** The metric that HPO optimizes. You specify the metric name (as it appears in the training job's CloudWatch logs) and whether to minimize or maximize it. For example, `validation:auc` with `Maximize`.
4. **Resource limits:** Maximum number of training jobs to launch (`MaxNumberOfTrainingJobs`) and maximum parallel jobs (`MaxParallelTrainingJobs`). These control cost and time.

### Lifecycle of an HPO job

1. **Job creation:** You submit the HPO job with all four components. SageMaker validates the configuration.
2. **Trial scheduling:** SageMaker selects hyperparameter values for the first batch of trials (up to `MaxParallelTrainingJobs`) and launches them as independent Training Jobs.
3. **Trial execution:** Each Training Job runs with its assigned hyperparameters. During training, it emits the objective metric to CloudWatch logs.
4. **Metric collection:** SageMaker monitors the logs and records the objective metric value for each completed trial.
5. **Next-batch selection:** Based on completed trial results, SageMaker selects hyperparameter values for the next batch using the optimization strategy (covered in the next topic). Better-performing regions of the search space receive more exploration.
6. **Termination:** The job completes when `MaxNumberOfTrainingJobs` trials have finished, or an early stopping condition is met.
7. **Results:** SageMaker ranks all trials by objective metric and identifies the best trial.

### Creating an HPO job in the console

1. Navigate to **SageMaker > Training > Hyperparameter tuning jobs > Create hyperparameter tuning job**.
2. **Job name:** Enter a descriptive name (e.g., `xgboost-fraud-hpo`).
3. **Optimization strategy:** Select Bayesian (default) or Random (covered in the next topic).
4. **Objective metric:** Enter the metric name and direction (e.g., `validation:f1`, `Maximize`).
5. **Resource limits:** Set `MaxNumberOfTrainingJobs` (e.g., 20) and `MaxParallelTrainingJobs` (e.g., 4).
6. **Training job definition:** Configure the base training job (algorithm, instance type, input channels, output path).
7. **Hyperparameter ranges:** For each parameter you want to tune, set the range type and bounds. Fixed hyperparameters (not tuned) go in the static hyperparameters section.
8. Click **Create hyperparameter tuning job**.

### Monitoring an HPO job

While the job runs, you can monitor it in the console:

- **SageMaker > Training > Hyperparameter tuning jobs > [your job]:** Shows overall status, total trials completed, best trial so far, and elapsed time.
- **Training jobs tab:** Lists every individual trial with its hyperparameter values, objective metric, status, and training time.
- **Best training job tab:** Shows the hyperparameter values and metric for the top-performing trial.

You can also view training logs for any individual trial by clicking through to its CloudWatch log stream.

### Cost management

Each trial in an HPO job is a separate Training Job with its own instance. Cost = (number of trials) x (instance cost per trial) x (training time per trial). To manage costs:

- **Start small:** Begin with `MaxNumberOfTrainingJobs` = 10-20 and `MaxParallelTrainingJobs` = 2-4. You can always run another HPO job with refined ranges.
- **Use `ml.m5.xlarge`:** Stays within Free Tier limits and is sufficient for most tabular models.
- **Enable early stopping:** SageMaker can automatically stop trials that are clearly underperforming, saving compute. Set `EarlyStoppingType` to `Auto` in the job configuration.
- **Narrow ranges:** Wide ranges waste trials on unpromising configurations. If you know `max_depth` should be between 4 and 8, do not set the range to 1-20.

### HPO job artifacts

When the HPO job completes:

- The **best training job** has its model artifact in S3, ready for deployment. You can deploy it directly from the console by navigating to the best trial's details and clicking **Create model**.
- The **hyperparameter values** from the best trial are your tuned configuration. Use these values in your production training script for reproducibility.
- The **full trial history** is available in the console and via the `DescribeHyperParameterTuningJob` API, enabling analysis of which hyperparameter ranges were most impactful.

## Connecting to Practice

This reading gives you the structural understanding of HPO jobs. The next topic, *Optimization Strategies*, covers the algorithms SageMaker uses to select hyperparameter values between trials (Bayesian, Random, Grid, Hyperband). The topic after that, *Parameter Ranges in Pipelines*, shows how to embed HPO as a step in a SageMaker Pipeline. In the module assignment, you will run an HPO job to tune an XGBoost model and compare the best trial's metrics against your manual and Autopilot baselines.

## Further Learning & Resources

**Documentation and reading**

- **[SageMaker Hyperparameter Tuning](https://docs.aws.amazon.com/sagemaker/latest/dg/automatic-model-tuning.html)** - *Docs*: Complete reference for HPO job configuration, metric definitions, and best practices.
- **[HPO Best Practices](https://docs.aws.amazon.com/sagemaker/latest/dg/automatic-model-tuning-considerations.html)** - *Docs*: AWS guidance on selecting ranges, managing costs, and interpreting results.

**Interactive practice**

- **[Hyperparameter Tuning with XGBoost](https://github.com/aws/amazon-sagemaker-examples/tree/main/hyperparameter_tuning/xgboost_random_log)** - *Interactive*: Sample notebook demonstrating HPO with XGBoost on SageMaker.
