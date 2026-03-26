# Model Monitor Baselines

**Estimated Time:** 10 Minutes

## Introduction

A model that performs well at deployment does not stay that way forever. Input data distributions shift, customer behavior changes, and the patterns the model learned become stale. Without monitoring, you discover degradation only when business metrics drop -- often weeks or months after the model started producing poor predictions.

SageMaker **Model Monitor** detects data quality and model quality issues in production endpoints by comparing live inference data against a **baseline** -- a statistical profile of the data the model was trained on. This reading covers how baselines work, how to create them, and how they anchor the monitoring pipeline you will build across this module.

## Core Concepts

### What is a baseline?

A baseline is a statistical summary of your training data. It captures the distributions, ranges, and relationships that the model expects to see at inference time. Model Monitor uses the baseline as the reference point: if incoming data deviates significantly from the baseline statistics, Model Monitor flags a violation.

A baseline includes:

- **Schema:** Column names, data types, and whether each feature is required or optional.
- **Statistics:** Per-feature statistics: mean, standard deviation, min, max, median, unique values, and missing value percentages.
- **Constraints:** Rules derived from the statistics. For example: "Feature `age` must be between 18 and 120" or "Feature `product_category` must be one of ['electronics', 'clothing', 'food']."

### Creating a baseline with a Baselining Job

SageMaker provides a dedicated job type for baseline generation:

1. Navigate to **SageMaker > Inference > Model monitoring > Create monitoring schedule** (the console walks you through baseline creation as part of schedule setup).
2. Alternatively, you can run a standalone **Processing Job** using the Model Monitor container:
   - **Container:** SageMaker's pre-built Model Monitor container (available as a built-in image per region).
   - **Input:** Your training dataset in S3 (CSV or JSON Lines).
   - **Output:** The baseline statistics file (`statistics.json`) and constraints file (`constraints.json`) written to S3.

The baselining job analyzes every feature in the training data and produces the statistics and constraints files. These files are the reference for all subsequent monitoring.

### Baseline outputs

**`statistics.json`** contains computed statistics for each feature:

| Statistic | Description |
| :--- | :--- |
| `mean` | Average value (numeric features) |
| `stddev` | Standard deviation (numeric features) |
| `min` / `max` | Range boundaries |
| `num_present` / `num_missing` | Count of present and missing values |
| `distinct_count` | Number of unique values (categorical features) |
| `distribution` | Histogram or frequency table |

**`constraints.json`** contains rules like:

- Feature `amount` has type `Fractional` and is `non_nullable`.
- Feature `category` has an allowed set of values: `["A", "B", "C"]`.
- Feature `age` has a minimum of 18 and maximum of 95.

You can edit the constraints file to add, remove, or adjust rules before attaching it to a monitoring schedule. For example, if the training data had ages up to 95 but you expect the model to handle ages up to 120, you would increase the maximum constraint.

### Baseline for different monitoring types

Model Monitor supports four monitoring types, each requiring its own baseline:

| Monitoring Type | Baseline Content | What It Detects |
| :--- | :--- | :--- |
| **Data Quality** | Statistics and constraints of input features | Feature distribution drift, missing values, out-of-range values |
| **Model Quality** | Performance metrics (accuracy, F1, etc.) from a labeled evaluation set | Prediction accuracy degradation |
| **Bias Drift** | Bias metrics from SageMaker Clarify | Changes in model fairness across protected groups |
| **Feature Attribution Drift** | SHAP values from SageMaker Clarify | Changes in which features drive predictions |

This reading focuses on the baseline concept and Data Quality baselines. The remaining monitoring types are covered in subsequent topics.

### Updating baselines

Baselines should be updated when you retrain your model on new data. The new training data establishes the new expected distributions. Best practice:

1. Run a new baselining job as part of your retraining Pipeline (include it as a Processing Step after the training data is prepared).
2. Store the baseline files in S3 alongside the model artifacts (e.g., `s3://bucket/models/v2/baseline/`).
3. Update the monitoring schedule to reference the new baseline files.
4. Register the baseline version in Model Registry metadata for auditability.

## Connecting to Practice

Baselines are the foundation for all four monitoring types. The next topic, *Data Quality Monitoring*, shows how to configure automated monitoring schedules that compare live endpoint data against your baseline. The module lecture will walk through creating a baseline from training data and inspecting the statistics and constraints files. The assignment will require you to create a baseline and customize constraints for a deployed model.

## Further Learning & Resources

**Documentation and reading**

- **[Model Monitor Baseline](https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor-create-baseline.html)** - *Docs*: Step-by-step guide for creating baselines with the Model Monitor container.
- **[Monitor Data Quality](https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor-data-quality.html)** - *Docs*: Reference for data quality baseline statistics and constraint formats.

**Interactive practice**

- **[Model Monitor Workshop](https://catalog.workshops.aws/sagemaker-model-monitor/en-US)** - *Interactive*: Hands-on lab covering baseline creation, monitoring schedules, and violation analysis.
