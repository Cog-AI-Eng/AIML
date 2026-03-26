# Data Quality Monitoring

**Estimated Time:** 10 Minutes

## Introduction

The baseline you created in the previous topic defines what the model's input data *should* look like. Data Quality Monitoring continuously compares live inference requests against that baseline to detect drift, missing values, schema violations, and out-of-range values. When violations are detected, Model Monitor generates a report and can trigger alerts through CloudWatch Alarms and SNS notifications.

This is the first and most commonly deployed monitoring type because data issues are the most frequent cause of model degradation. A model trained on data where `price` ranges from $5 to $500 will produce unreliable predictions if production traffic suddenly includes prices of $50,000.

## Core Concepts

### How Data Quality Monitoring works

1. **Data capture:** SageMaker captures a sample of inference requests and responses from your endpoint and writes them to S3 in JSON Lines format. You enable data capture in the endpoint configuration.
2. **Monitoring schedule:** A recurring Processing Job (hourly, daily, or custom) that reads the captured data, computes statistics, and compares them against the baseline.
3. **Violation report:** If any feature violates the baseline constraints, Model Monitor writes a violation report to S3.
4. **Alerting:** CloudWatch metrics are emitted for each monitoring execution, including violation counts. You can create CloudWatch Alarms to notify your team via SNS when violations exceed a threshold.

### Enabling data capture

Data capture must be enabled on the endpoint before monitoring can begin:

1. In the **Endpoint configuration** (at creation time or via update):
   - **Enable data capture:** Toggle on.
   - **Capture content type:** The content type of captured data (e.g., `text/csv`).
   - **Initial sampling percentage:** The fraction of requests to capture (e.g., 100% for low-traffic endpoints, 10-20% for high-traffic). Higher percentages provide more data for monitoring but increase S3 storage costs.
   - **S3 destination:** The S3 prefix where captured data is stored (e.g., `s3://bucket/data-capture/`).
   - **Capture mode:** `Input` (requests only), `Output` (responses only), or `InputAndOutput` (both).

### Creating a monitoring schedule

1. Navigate to **SageMaker > Inference > Model monitoring > Create monitoring schedule**.
2. **Schedule name:** Enter a descriptive name (e.g., `fraud-model-data-quality`).
3. **Monitoring type:** Select **Data quality**.
4. **Endpoint:** Select the endpoint to monitor.
5. **Baseline:** Specify the S3 paths to the `statistics.json` and `constraints.json` files from the baseline job.
6. **Output:** S3 path for monitoring reports.
7. **Schedule:** Hourly (recommended for production), daily, or a custom cron expression.
8. **Instance type:** `ml.m5.xlarge` is sufficient for most monitoring jobs.
9. Click **Create schedule**.

### Interpreting monitoring reports

Each execution of the monitoring schedule produces:

- **`constraint_violations.json`:** Lists every constraint that was violated, with details:
  - Feature name
  - Violation type (e.g., `data_type_check`, `completeness_check`, `baseline_drift_check`)
  - Description (e.g., "Feature `price` has a mean of 12,500, baseline mean was 250, exceeding the threshold of 3 standard deviations")

- **`statistics.json`:** The computed statistics for the current monitoring period, in the same format as the baseline. Useful for comparing current vs. baseline distributions.

### Common violation types

| Violation Type | What It Means | Typical Cause |
| :--- | :--- | :--- |
| `data_type_check` | A feature's data type does not match the baseline | Upstream pipeline change, data corruption |
| `completeness_check` | Missing value percentage exceeds the baseline threshold | Data source outage, ETL failure |
| `baseline_drift_check` | Feature distribution has shifted significantly from baseline | Natural data drift, population change |
| `extra_column_check` | A column exists in live data that was not in the baseline | Schema evolution in the upstream system |
| `missing_column_check` | A baseline column is missing from the live data | Upstream schema change, feature pipeline failure |

### CloudWatch integration

Model Monitor emits CloudWatch metrics after each execution:

- **Violations count:** Total number of constraint violations. Create an alarm if this exceeds 0 (or a tolerance threshold).
- **Execution status:** Success or failure of the monitoring job itself.
- **Processing time:** Duration of the monitoring execution.

To set up alerting: **CloudWatch > Alarms > Create alarm**, select the Model Monitor metric namespace, choose the violations metric, set a threshold, and attach an SNS topic for email/Slack notification.

### Cost management

Each monitoring execution is a Processing Job with its own instance. Costs:

- **Instance cost:** Per monitoring job execution (typically 5-15 minutes on `ml.m5.xlarge`).
- **S3 storage:** Data capture files accumulate over time. Set S3 lifecycle policies to archive or delete old capture data.
- **Frequency vs. cost:** Hourly monitoring = 720 jobs/month. Daily monitoring = 30 jobs/month. Choose frequency based on how quickly you need to detect issues vs. cost tolerance.

## Connecting to Practice

Data Quality Monitoring is the most fundamental monitoring type. The next topic, *Model Quality Monitoring*, covers monitoring the model's prediction accuracy over time (which requires ground truth labels). The module assignment will require you to enable data capture, create a monitoring schedule, inject synthetic drift, and demonstrate the violation detection workflow.

## Further Learning & Resources

**Documentation and reading**

- **[Data Quality Monitoring](https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor-data-quality.html)** - *Docs*: Complete reference for data quality monitoring configuration and violation types.
- **[Data Capture](https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor-data-capture.html)** - *Docs*: Guide to enabling and configuring data capture on endpoints.

**Interactive practice**

- **[Model Monitor Data Quality Lab](https://github.com/aws/amazon-sagemaker-examples/tree/main/sagemaker_model_monitor)** - *Interactive*: Sample notebook demonstrating end-to-end data quality monitoring with violation analysis.
