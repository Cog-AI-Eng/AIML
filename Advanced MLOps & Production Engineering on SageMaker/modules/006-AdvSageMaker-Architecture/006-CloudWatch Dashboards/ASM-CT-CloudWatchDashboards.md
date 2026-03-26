# CloudWatch Dashboards

**Estimated Time:** 10 Minutes

## Introduction

SageMaker emits dozens of CloudWatch metrics for every training job, endpoint, and monitoring schedule. By default, these metrics are scattered across different CloudWatch namespaces and require navigating through multiple console pages to view. CloudWatch Dashboards let you assemble the most important metrics into a single visual display, providing real-time operational visibility into your ML platform.

This reading covers the SageMaker-specific CloudWatch metrics, how to build dashboards for training and inference workloads, and best practices for alerting.

## Core Concepts

### SageMaker CloudWatch metric namespaces

| Namespace | Metrics | Source |
| :--- | :--- | :--- |
| `AWS/SageMaker` | Endpoint invocations, latency, errors | Real-time endpoints |
| `aws/sagemaker/Endpoints` | `CPUUtilization`, `MemoryUtilization`, `DiskUtilization` | Endpoint instances |
| `aws/sagemaker/TrainingJobs` | `CPUUtilization`, `MemoryUtilization`, `GPUUtilization` | Training job instances |
| `aws/sagemaker/ModelMonitor` | Violation counts, execution status | Model Monitor schedules |

### Key endpoint metrics

| Metric | Description | Alert Threshold |
| :--- | :--- | :--- |
| `Invocations` | Total request count | N/A (informational) |
| `InvocationsPerInstance` | Requests per instance per minute | Scale-out trigger (e.g., > 100) |
| `ModelLatency` | Time for the model to process one request (ms) | SLA threshold (e.g., > 200ms) |
| `OverheadLatency` | SageMaker overhead (serialization, routing) | > 50ms indicates infrastructure issues |
| `Invocation4XXErrors` | Client errors (bad requests) | > 1% of total invocations |
| `Invocation5XXErrors` | Server errors (model failures) | Any occurrence |
| `CPUUtilization` | Instance CPU usage | > 80% sustained |
| `MemoryUtilization` | Instance memory usage | > 90% (OOM risk) |

### Building a SageMaker dashboard

1. Navigate to **CloudWatch > Dashboards > Create dashboard**.
2. **Name:** Use a descriptive name (e.g., `fraud-model-production`).
3. **Add widgets:** Click **Add widget** and choose the visualization type (Line graph, Number, Gauge, etc.).
4. **Select metrics:** Browse to the appropriate namespace and select metrics. For a production endpoint dashboard, add:
   - **Line graph:** `Invocations` over time (traffic pattern)
   - **Line graph:** `ModelLatency` p50 and p99 (latency trend)
   - **Number:** Current `Invocation5XXErrors` (instant alert)
   - **Line graph:** `CPUUtilization` and `MemoryUtilization` per instance (resource usage)
   - **Number:** Instance count from auto-scaling (current capacity)
   - **Line graph:** Model Monitor violation count (drift detection)

### Training job dashboard

For teams running many training jobs and HPO trials:

- **Line graph:** `CPUUtilization` and `MemoryUtilization` during training (helps with instance right-sizing)
- **Number:** Active training jobs count
- **Line graph:** HPO best objective metric over trial number (convergence visualization)
- **Number:** Spot savings percentage for recent training jobs

### Alerting with CloudWatch Alarms

Dashboards provide visibility; alarms provide action. For each critical metric, create a CloudWatch Alarm:

1. **CloudWatch > Alarms > Create alarm**.
2. Select the metric (e.g., `Invocation5XXErrors`).
3. Set the condition (e.g., "greater than 0 for 1 consecutive period of 5 minutes").
4. Set the action: publish to an SNS topic for email/Slack notification, or trigger a Lambda function for automated remediation.

### Dashboard best practices

- **One dashboard per production model:** Each deployed model gets its own dashboard with endpoint metrics, monitoring status, and recent Pipeline executions.
- **Include business context:** Add a text widget with the model name, version, deployment date, and owner. This helps on-call engineers identify what they are looking at.
- **Use anomaly detection bands:** CloudWatch supports anomaly detection that automatically computes expected metric ranges. Enable this on `ModelLatency` to detect unusual latency without setting static thresholds.
- **Share with stakeholders:** CloudWatch dashboards can be shared as public URLs (for internal use) or embedded in wikis. Non-technical stakeholders appreciate a single screen showing model health.

## Connecting to Practice

CloudWatch Dashboards provide the operational visibility that ties together all the SageMaker services covered in this skill unit. The final topic, *VPC, PrivateLink & KMS*, covers network security and encryption. The module assignment will require you to build a dashboard for a production endpoint that includes invocation metrics, latency, resource utilization, and Model Monitor violation status.

## Further Learning & Resources

**Documentation and reading**

- **[Monitor SageMaker with CloudWatch](https://docs.aws.amazon.com/sagemaker/latest/dg/monitoring-cloudwatch.html)** - *Docs*: Reference for all SageMaker CloudWatch metrics and dimensions.
- **[CloudWatch Dashboards](https://docs.aws.amazon.com/AmazonCloudWatch/latest/monitoring/CloudWatch_Dashboards.html)** - *Docs*: Guide to creating and sharing CloudWatch dashboards.

**Interactive practice**

- **[SageMaker Monitoring Dashboard Lab](https://catalog.workshops.aws/sagemaker-cost-optimization/en-US)** - *Interactive*: Workshop section covering CloudWatch dashboard creation for SageMaker workloads.
