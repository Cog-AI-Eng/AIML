# Model Quality Monitoring

**Estimated Time:** 10 Minutes

## Introduction

Data Quality Monitoring detects changes in the *input* data. Model Quality Monitoring goes a step further: it detects changes in the *prediction accuracy* of the model. If the model's F1 score was 0.92 at deployment but has degraded to 0.78 in production, Model Quality Monitoring catches this -- but only if you provide ground truth labels to compare against.

This distinction is critical: data drift can occur without affecting model quality (if the drift is in features the model does not rely on), and model quality can degrade without data drift (if the underlying relationship between features and target has changed). Model Quality Monitoring addresses the second case.

## Core Concepts

### The ground truth requirement

Model Quality Monitoring compares predictions against actual outcomes. This means you need a mechanism to collect ground truth labels for production predictions:

- **Delayed labels:** In fraud detection, you discover the true outcome (fraudulent or not) days or weeks after the prediction. You must collect and store these labels, matched to the original prediction by request ID.
- **Immediate labels:** In recommendation systems, the user's click (or non-click) is the ground truth, available immediately after the prediction.
- **Sampled labels:** In some cases, ground truth is expensive to obtain (e.g., manual review). You label a sample and extrapolate.

Ground truth data must be uploaded to S3 in a format that Model Monitor can join with the captured predictions. The typical format is a CSV or JSON Lines file with columns for the prediction request ID and the true label.

### Setting up Model Quality Monitoring

1. **Enable data capture** on the endpoint (same as Data Quality Monitoring -- capture input and output).
2. **Create a model quality baseline:** Run a baseline job using a labeled evaluation dataset. This produces baseline metrics (e.g., accuracy = 0.92, F1 = 0.89, AUC = 0.95) and establishes the performance the model achieved at training time.
3. **Create a monitoring schedule:**
   - Navigate to **SageMaker > Inference > Model monitoring > Create monitoring schedule**.
   - **Monitoring type:** Select **Model quality**.
   - **Problem type:** Classification or regression.
   - **Ground truth S3 path:** The S3 location where ground truth labels are stored. Model Monitor reads new labels at each scheduled execution.
   - **Baseline:** The S3 path to the baseline metrics from step 2.
   - Configure schedule frequency, instance type, and output path.

### Baseline metrics

The model quality baseline records performance metrics computed on the evaluation dataset at training time:

**Classification metrics:**
- Accuracy, Precision, Recall, F1
- AUC-ROC
- Per-class metrics (for multi-class)

**Regression metrics:**
- RMSE, MAE, R-squared
- Mean and standard deviation of residuals

These baseline values serve as the reference. Model Quality Monitoring computes the same metrics on production data (using ground truth) and flags violations when any metric degrades beyond a configurable threshold.

### Monitoring execution flow

Each scheduled execution:

1. Reads captured predictions from S3 (from data capture).
2. Reads ground truth labels from S3 (from your label collection pipeline).
3. Joins predictions with labels by request ID.
4. Computes quality metrics on the joined data.
5. Compares computed metrics against the baseline.
6. Writes a violation report if any metric has degraded.
7. Emits CloudWatch metrics for alerting.

### Handling delayed labels

When labels arrive days or weeks after predictions, you need to account for the lag:

- **Monitoring schedule frequency:** Set to match your label availability cadence. If labels arrive weekly, schedule monitoring weekly.
- **Partial monitoring:** Model Monitor can compute metrics on whatever labels are available at execution time. If only 60% of predictions have labels, the metrics are computed on the labeled subset.
- **Backfill:** When late-arriving labels fill in, the next monitoring execution includes them automatically (as long as they are in the ground truth S3 path).

### When model quality degrades

Model Quality violations are the strongest signal that your model needs retraining. When violations are detected:

1. Check Data Quality violations for the same period. If data drift is also detected, the model degradation is likely caused by distribution shift -- retrain on recent data.
2. If no data drift is detected, the relationship between features and target may have changed (concept drift). Investigate whether the business context has shifted.
3. Use the monitoring metrics to determine the severity. A small degradation (F1 from 0.92 to 0.89) may be acceptable. A large degradation (F1 from 0.92 to 0.70) requires immediate action.

## Connecting to Practice

Model Quality Monitoring is the "outcome-aware" complement to Data Quality Monitoring. The next topic, *Bias & Attribution Drift*, covers monitoring for fairness and explainability changes. The module assignment will require you to set up a model quality monitoring schedule with simulated ground truth labels and demonstrate the metric comparison workflow.

## Further Learning & Resources

**Documentation and reading**

- **[Model Quality Monitoring](https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor-model-quality.html)** - *Docs*: Complete reference for model quality baseline creation, ground truth integration, and metrics.
- **[Ground Truth Merge](https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor-model-quality-merge.html)** - *Docs*: Guide to formatting and merging ground truth labels with captured predictions.

**Interactive practice**

- **[Model Quality Monitor Example](https://github.com/aws/amazon-sagemaker-examples/tree/main/sagemaker_model_monitor/model_quality)** - *Interactive*: Sample notebook demonstrating model quality monitoring with classification metrics.
