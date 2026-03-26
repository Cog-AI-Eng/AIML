# Parameter Ranges in Pipelines

**Estimated Time:** 10 Minutes

## Introduction

The previous two topics covered how to run HPO jobs from the console as standalone operations. In a production MLOps workflow, hyperparameter tuning is not a one-time manual step -- it is a recurring stage in an automated pipeline that runs whenever new data arrives or model performance degrades. SageMaker Pipelines supports a dedicated **Tuning Step** that embeds an HPO job as a node in your pipeline DAG, making tuning a first-class citizen of your CI/CD process.

This reading covers how to configure hyperparameter ranges within a Pipeline, how the Tuning Step interacts with other Pipeline steps, and how to extract the best model from a completed tuning step for downstream registration and deployment.

## Core Concepts

### The Tuning Step in SageMaker Pipelines

A SageMaker Pipeline is a DAG of steps. The `TuningStep` wraps an HPO job, giving you the same capabilities (strategy selection, parallel trials, warm start) but orchestrated automatically within a pipeline execution.

The Tuning Step definition in the SDK requires:

1. **A tuner object:** Configured with the estimator, hyperparameter ranges, objective metric, and strategy -- identical to a standalone HPO job.
2. **Input channels:** S3 paths for training and validation data. These can be outputs from a preceding Processing Step (e.g., Data Wrangler export or custom preprocessing).
3. **Step name:** A unique identifier within the pipeline.

When the pipeline reaches the Tuning Step, SageMaker launches the HPO job, waits for all trials to complete, and makes the best model artifact available for subsequent steps.

### Connecting Tuning to downstream steps

The most common pipeline pattern after tuning:

1. **Processing Step:** Preprocesses raw data, splits into train/validation/test, writes to S3.
2. **Tuning Step:** Takes the processed train and validation data, runs HPO, produces the best model.
3. **Evaluation Step:** Loads the best model and runs it against the held-out test set. Computes metrics.
4. **Condition Step:** Checks whether the test metrics meet a threshold (e.g., AUC > 0.85). If yes, proceed; if no, fail the pipeline.
5. **Register Step:** Registers the best model in Model Registry with metadata (metrics, hyperparameters, data lineage).
6. **Deploy Step (optional):** Deploys the registered model to a real-time endpoint.

The key integration point is between the Tuning Step and the Evaluation Step. The Tuning Step exposes a `get_top_model_s3_uri()` method that returns the S3 path of the best trial's model artifact. The Evaluation Step uses this path to load the model.

### Parameterizing ranges with Pipeline Parameters

SageMaker Pipelines supports **Pipeline Parameters** -- variables that you set at execution time without modifying the pipeline definition. You can parameterize hyperparameter ranges so the same pipeline can be re-executed with different search configurations:

- Use a `ParameterString` to pass the optimization strategy (`Bayesian`, `Random`).
- Use `ParameterInteger` for `MaxNumberOfTrainingJobs` and `MaxParallelTrainingJobs`.
- Hyperparameter range bounds themselves can also be parameterized, though this is less common.

This parameterization is valuable when running A/B experiments: execute the same pipeline with different HPO budgets to compare cost vs. accuracy trade-offs.

### Viewing Tuning Steps in the console

After a pipeline execution that includes a Tuning Step:

1. Navigate to **SageMaker > Pipelines > [your pipeline] > [execution ID]**.
2. The DAG view shows the Tuning Step as a node. Click it.
3. The step detail panel links to the underlying HPO job, where you can see individual trials, metrics, and the best configuration.
4. Downstream steps show their inputs traced back to the Tuning Step, providing full lineage.

### Cost governance in pipeline tuning

When HPO runs inside a pipeline, it is triggered automatically -- potentially on a schedule or by an S3 event. Without guardrails, this can generate unexpected costs. Best practices:

- **Set conservative `MaxNumberOfTrainingJobs`** in the tuner (e.g., 20-30 rather than 100).
- **Use `ml.m5.xlarge`** instances and avoid GPU instances unless the algorithm requires them.
- **Add a Condition Step before tuning** that checks whether the existing model's performance has degraded below a threshold. Only proceed to tuning if retraining is actually needed.
- **Tag HPO jobs** with cost-tracking tags so you can identify pipeline-triggered HPO costs in AWS Cost Explorer.

### Warm start across pipeline executions

When a pipeline runs repeatedly (e.g., weekly retraining), each execution's Tuning Step can warm-start from the previous execution's best trial. This is configured by storing the previous HPO job ARN as a Pipeline Parameter and passing it to the tuner's `WarmStartConfig`.

This pattern significantly reduces the number of trials needed in subsequent runs because the search starts from the best-known configuration rather than from scratch.

## Connecting to Practice

This topic completes the Algorithms module by connecting HPO to the broader MLOps pipeline. You now have a full picture: select an algorithm (XGBoost, K-Means, RCF, BlazingText, DeepAR), tune it with HPO (using the right strategy), and embed the entire workflow in an automated Pipeline. The module lecture will walk through building a Pipeline with Processing, Tuning, and Evaluation steps. The assignment will require you to build a Pipeline that includes a Tuning Step with parameterized ranges and a Condition Step that gates deployment on metric thresholds.

## Further Learning & Resources

**Documentation and reading**

- **[Pipeline Tuning Step](https://docs.aws.amazon.com/sagemaker/latest/dg/build-and-manage-steps.html#step-type-tuning)** - *Docs*: Reference for configuring Tuning Steps within SageMaker Pipelines.
- **[Pipeline Parameters](https://docs.aws.amazon.com/sagemaker/latest/dg/build-and-manage-parameters.html)** - *Docs*: Guide to parameterizing pipeline steps for flexible execution.

**Interactive practice**

- **[SageMaker Pipelines with HPO](https://github.com/aws/amazon-sagemaker-examples/tree/main/sagemaker-pipelines/tabular/tuning-step)** - *Interactive*: Sample notebook demonstrating a Pipeline with a Tuning Step for XGBoost.
