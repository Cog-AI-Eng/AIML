# Experiments & Trials

**Estimated Time:** 10 Minutes

## Introduction

Every training job you launch on SageMaker produces metrics, hyperparameters, and model artifacts. Without a structured system for organizing these outputs, it becomes impossible to answer basic questions: "Which training configuration produced the best F1 score?" "What hyperparameters did we use for the model currently in production?" "How does last week's model compare to today's?"

SageMaker **Experiments** is the managed experiment tracking service that organizes training runs into a hierarchy: **Experiments** contain **Runs** (formerly called Trials), and each Run records the parameters, metrics, and artifacts from a single training job. This structure lets you compare runs side by side, reproduce past results, and maintain an audit trail of every model you have built.

## Core Concepts

### The Experiment hierarchy

- **Experiment:** A named container for related Runs. Typically one Experiment per ML project or prediction task (e.g., `fraud-detection-xgboost`).
- **Run (Trial):** A single training execution within an Experiment. Each Run captures:
  - **Parameters:** Hyperparameters and configuration values (e.g., `max_depth=6`, `learning_rate=0.1`).
  - **Metrics:** Training and validation metrics logged during training (e.g., `train:auc`, `validation:f1`).
  - **Artifacts:** References to input data, model artifacts, and output files in S3.
  - **Metadata:** Timestamps, instance type, training job ARN, and custom tags.

### Creating Experiments in the console

1. Navigate to **SageMaker > Experiments** in the console sidebar.
2. Click **Create experiment**.
3. Enter an experiment name and optional description.
4. The experiment appears in the list. It is empty until you associate Runs with it.

### Associating training jobs with Experiments

When creating a training job (either from the console or SDK), you can associate it with an Experiment by setting the experiment name and run name in the job configuration. In the console: under **Training job > Advanced settings**, look for the **Experiment configuration** section. Enter the experiment name and a run name.

Each training job associated with an experiment becomes a Run. SageMaker automatically captures the job's hyperparameters and metrics into the Run record.

### Comparing Runs

The primary value of Experiments is comparison. In the console:

1. Navigate to **SageMaker > Experiments > [your experiment]**.
2. Select multiple Runs by checking their boxes.
3. Click **Compare**. SageMaker shows a side-by-side table of parameters and metrics.
4. You can sort by any metric column to identify the best-performing run.
5. The comparison view also supports metric charts: plot a metric (e.g., validation loss) across training epochs for selected Runs to visualize convergence behavior.

### Integration with HPO jobs

HPO jobs automatically create Experiment Runs. When you launch an HPO job, SageMaker creates one Run per trial, all under the same Experiment. This means the HPO leaderboard from the previous module is also available as an Experiment comparison in the Experiments console. The benefit: you can compare HPO trials alongside manual training jobs in the same Experiment.

### Custom metric logging

SageMaker captures metrics that appear in training job CloudWatch logs using a regex pattern you define in the metric definitions. For built-in algorithms, metric definitions are pre-configured. For Script Mode, you need to:

1. Print metric values to stdout in a consistent format during training (e.g., `print(f"validation:f1={f1_score}")`).
2. Define a regex pattern in the estimator's `metric_definitions` parameter that matches the format (e.g., `{"Name": "validation:f1", "Regex": "validation:f1=([0-9\\.]+)"}`).

SageMaker parses the training logs and records matching values as time-series metrics in the Run.

### Experiment analytics

Beyond the console comparison view, you can query Experiment data programmatically using the `sagemaker.analytics.ExperimentAnalytics` class in the SDK. This returns a pandas DataFrame of all Runs with their parameters and metrics, enabling custom analysis, visualization, and reporting in a Jupyter notebook.

## Connecting to Practice

Experiments provide the organizational layer for everything you have built so far: training jobs, HPO trials, and model evaluations are all trackable as Runs. The next topic, *Lineage Tracking Entities*, goes deeper into the data-level provenance system that tracks which data, code, and artifacts produced each model. The module lecture will have you set up an Experiment, run multiple training configurations, and use the console comparison view to select the best model. The assignment will require you to demonstrate Experiment tracking integrated with an HPO job.

## Further Learning & Resources

**Documentation and reading**

- **[SageMaker Experiments](https://docs.aws.amazon.com/sagemaker/latest/dg/experiments.html)** - *Docs*: Complete reference for Experiments, Runs, metric tracking, and analytics.

**Interactive practice**

- **[Experiment Tracking Workshop](https://github.com/aws/amazon-sagemaker-examples/tree/main/sagemaker-experiments)** - *Interactive*: Sample notebooks demonstrating Experiment creation, Run logging, and comparison analytics.
