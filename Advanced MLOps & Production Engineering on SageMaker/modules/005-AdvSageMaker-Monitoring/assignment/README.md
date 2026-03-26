# Advanced SageMaker Monitoring Lab

## Scenario
FraudShield Risk Analytics has production endpoints serving fraud predictions around the clock. The risk management team has raised concerns: how will you know if incoming transaction data starts drifting from the patterns the model was trained on? What if model accuracy silently degrades after a seasonal shift in spending behavior? Your job is to instrument the FraudShield inference pipeline with comprehensive monitoring so the team gets alerted before bad predictions reach customers.

In this lab you will enable data capture on a live endpoint, establish statistical baselines from training data, schedule continuous data quality and model quality monitors, and wire up CloudWatch alarms with SNS notifications. When you are finished, FraudShield will have an automated early-warning system that detects data drift, schema violations, and model performance degradation without any human having to manually inspect predictions.

---

## Learning Objectives
By completing this lab you will demonstrate the ability to:
1. Enable and configure data capture on a real-time SageMaker endpoint
2. Generate a data quality baseline from training data with statistics and constraints
3. Create and schedule a recurring data quality monitoring job
4. Configure model quality monitoring with ground truth integration
5. Set up CloudWatch alarms and SNS notifications for monitoring violations
6. Clean up all monitoring resources, schedules, and alarms

---

## Prerequisites
- AWS account with SageMaker, CloudWatch, and SNS permissions
- A deployed real-time FraudShield endpoint (from Module 4 or freshly created)
- The original FraudShield training dataset stored in S3
- An S3 bucket named `sagemaker-fraudshield-<account-id>` in us-east-1
- A valid email address for SNS notification testing

---

## Milestones

| # | Guide | Estimated Time | What You Build |
|---|-------|---------------|----------------|
| 1 | [Enable Data Capture](console_guides/01_enable_data_capture.md) | 20 min | Data capture configuration on a live endpoint |
| 2 | [Create Data Quality Baseline](console_guides/02_create_data_quality_baseline.md) | 25 min | Baseline statistics and constraints from training data |
| 3 | [Configure Monitoring Schedule](console_guides/03_configure_monitoring_schedule.md) | 25 min | An hourly data quality monitoring schedule |
| 4 | [Configure Model Quality Monitor](console_guides/04_configure_model_quality_monitor.md) | 25 min | Model quality baseline and monitoring with ground truth |
| 5 | [Setup CloudWatch Alerts](console_guides/05_setup_cloudwatch_alerts.md) | 20 min | CloudWatch alarm with SNS email notification |
| 6 | [Cleanup](console_guides/06_cleanup.md) | 15 min | All monitoring resources deleted |
| SDK | [SDK Monitoring Lab](notebooks/sdk_monitoring_lab.ipynb) | 45 min | Deploy with DataCaptureConfig, create baselines, configure monitoring schedules, and analyze violations using the SageMaker Python SDK |

**Total estimated time:** ~175 minutes (console guides ~130 min + SDK notebook ~45 min)

---

## Presentation Deliverables
1. Screenshot of the endpoint detail page showing data capture enabled with capture percentage and S3 path
2. Screenshot of the baseline job output in S3 showing statistics.json and constraints.json
3. Screenshot of a completed monitoring execution with the violation report visible
4. Screenshot of the model quality monitoring configuration showing ground truth S3 path
5. Screenshot of the CloudWatch alarm in ALARM or OK state with SNS action configured
6. Screenshot confirming all monitoring schedules, baselines, and alarms have been deleted

---

## Important Reminders
- **Free Tier:** Use ml.m5.xlarge or smaller. No GPU instances.
- **Region Consistency:** Stay in us-east-1.
- **Cleanup Is Mandatory:** Always complete the cleanup guide.
- **Do Not Skip Steps:** Each guide builds on the previous one.
