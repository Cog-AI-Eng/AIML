# Monitor & Pipeline Automation

**Estimated Time:** 10 Minutes

## Introduction

The monitoring types covered in this module -- Data Quality, Model Quality, Bias Drift, and Feature Attribution -- detect problems. But detection alone does not fix anything. The final piece of a production monitoring system is **automated response**: when monitoring detects drift or degradation, the system should trigger a retraining pipeline, redeploy the updated model, and update the monitoring baselines -- without human intervention.

This reading covers the architectural patterns that connect Model Monitor violations to SageMaker Pipelines, enabling closed-loop automation from detection to remediation.

## Core Concepts

### The closed-loop MLOps pattern

1. **Monitor:** Model Monitor detects a violation (data drift, model quality degradation, bias shift).
2. **Alert:** CloudWatch Alarm triggers based on violation metrics.
3. **Trigger:** The alarm invokes an action: start a Pipeline execution, notify a human, or both.
4. **Retrain:** The Pipeline processes fresh data, retrains the model, evaluates it, and registers a new version.
5. **Approve:** Manual or automatic approval in Model Registry.
6. **Deploy:** The approved model is deployed to the endpoint, replacing the degraded model.
7. **Rebaseline:** New baselines are computed from the retraining data and attached to the monitoring schedules.
8. **Resume monitoring:** The cycle continues with the updated baselines.

### Connecting CloudWatch Alarms to Pipelines

Model Monitor emits CloudWatch metrics after each execution. To trigger a Pipeline from an alarm:

**Option A: EventBridge Rule**
1. Create a CloudWatch Alarm on the Model Monitor violation count metric (e.g., alarm when violations > 0 for 2 consecutive periods).
2. Create an Amazon EventBridge rule that matches the alarm state change event.
3. Set the EventBridge target to a Lambda function that calls `sagemaker:StartPipelineExecution` via the SDK.
4. The Lambda function passes relevant parameters to the Pipeline (e.g., the date range of drifted data, the model name, the monitoring report S3 path).

**Option B: SNS to Lambda**
1. Create a CloudWatch Alarm on the violation metric.
2. Set the alarm action to publish to an SNS topic.
3. Subscribe a Lambda function to the SNS topic.
4. The Lambda function starts the Pipeline execution.

Both patterns achieve the same result. EventBridge is preferred for complex routing (e.g., different alarms trigger different pipelines), while SNS is simpler for a single alarm-to-pipeline connection.

### Pipeline design for automated retraining

The retraining Pipeline should include:

1. **Data Processing Step:** Pull the latest data from Feature Store or S3. Apply the same transformations as the original training pipeline.
2. **Training Step or Tuning Step:** Train (or re-tune) the model on the fresh data.
3. **Evaluation Step:** Evaluate the new model on a held-out test set.
4. **Baseline Step:** Run a Model Monitor baselining job on the new training data to generate updated statistics and constraints.
5. **Condition Step:** Check whether the new model's metrics exceed the current production model's metrics. If yes, proceed; if no, fail the pipeline (the new data does not produce a better model, indicating the issue is not solvable by retraining).
6. **Register Step:** Register the new model version in Model Registry with status `PendingManualApproval`.
7. **Deploy Step (optional, for automatic deployment):** If your governance policy allows, deploy the approved model to the endpoint automatically.
8. **Update Monitoring Step:** Update the monitoring schedules with the new baseline files.

### Human-in-the-loop vs. fully automated

The approval step (step 6) is the governance gate:

- **Manual approval:** The model is registered as `PendingManualApproval`. A human reviews the evaluation metrics, SHAP analysis, and bias report before clicking **Approve** in the Model Registry console. Only then is the model deployed.
- **Automatic approval:** The Condition Step in the Pipeline checks metrics against thresholds. If all thresholds pass, the Pipeline automatically sets the model status to `Approved` and deploys it.

Most organizations start with manual approval and transition to automatic approval as they gain confidence in their testing and monitoring systems. High-risk models (financial, healthcare) typically retain manual approval permanently.

### Preventing retraining loops

A risk with automated retraining is a feedback loop: the model degrades, triggers retraining, the new model is slightly better but still degraded, triggers monitoring again, and so on. Safeguards:

- **Cooldown period:** After a Pipeline execution, suppress additional alarm triggers for a configurable period (e.g., 24 hours) to give the new model time to stabilize.
- **Minimum improvement threshold:** The Condition Step should require the new model to exceed the *baseline* metrics (from the original training), not just the degraded metrics.
- **Maximum retrain frequency:** Limit Pipeline executions to at most once per day or week, depending on your use case.

### Console monitoring dashboard

To monitor the health of the entire closed-loop system:

1. **SageMaker > Inference > Model monitoring:** View all active schedules, recent violations, and execution history.
2. **SageMaker > Pipelines:** View Pipeline executions triggered by monitoring alerts, with step-level status.
3. **SageMaker > Governance > Model registry:** View model versions, approval status, and deployment history.
4. **CloudWatch > Dashboards:** Create a custom dashboard that combines Model Monitor violation counts, Pipeline execution status, and endpoint invocation metrics on a single screen.

## Connecting to Practice

This topic completes the Monitoring module by connecting detection to remediation. You now have a full monitoring system: baselines define expected behavior, four monitoring types detect different kinds of degradation, statistical tests quantify drift, and Pipeline automation closes the loop. The module lecture will demonstrate a full closed-loop workflow: deploy a model, inject drift, detect violations, trigger retraining, and redeploy. The assignment will require you to build an EventBridge-to-Lambda-to-Pipeline connection that triggers retraining from a monitoring alarm.

## Further Learning & Resources

**Documentation and reading**

- **[SageMaker Pipelines with Model Monitor](https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor-autopilot.html)** - *Docs*: Guide to integrating Model Monitor with Pipeline automation.
- **[EventBridge Rules for SageMaker](https://docs.aws.amazon.com/sagemaker/latest/dg/automating-sagemaker-with-eventbridge.html)** - *Docs*: Reference for SageMaker events and EventBridge integration patterns.

**Interactive practice**

- **[MLOps End-to-End Workshop](https://catalog.workshops.aws/sagemaker-mlops/en-US)** - *Interactive*: Comprehensive lab covering the full MLOps lifecycle from training through monitoring and automated retraining.
