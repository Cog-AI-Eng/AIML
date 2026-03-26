# Guide 5: Explore Model Monitor and EventBridge

This guide is a conceptual walkthrough -- you will navigate the console sections where Model Monitor and EventBridge are configured, understand what each service does in the MLOps lifecycle, and discuss how they connect to the pipeline you built. You will NOT create monitoring schedules or EventBridge rules in this lab.

---

## Steps

### Step 1 -- Understand Where Model Monitor Fits

Model Monitor sits between your live endpoint and your retraining pipeline:

```
Training Pipeline
       │
       ▼
Model Registry (Approved)
       │
       ▼
Deployed Endpoint (InService) ←── Predictions ←── Client
       │
       ▼
Model Monitor (captures inputs/outputs)
       │
       ▼
Drift Detected? ──Yes──> EventBridge ──> Retrain Pipeline
       │
       No
       │
       ▼
Continue serving
```

### Step 2 -- Explore Model Monitor in the Console

1. In the **SageMaker console**, go to **Inference** -> **Model monitoring** (or look for "Monitoring" in the left navigation).
2. This section is where you would configure:
   - **Data Capture:** Enable on an endpoint to record prediction inputs and outputs to S3
   - **Baseline:** Compute statistics and constraints from your training data
   - **Monitoring Schedules:** Periodic jobs that compare live data against the baseline
   - **Alerts:** Violations when data drifts beyond the baseline constraints
3. Note: Since you do not have a live endpoint right now (you cleaned it up in Module 3), this section will be empty. That is expected.

### Step 3 -- Understand the Four Types of Monitoring

SageMaker Model Monitor supports four types:

| Type | What It Checks | Example |
|------|---------------|---------|
| **Data Quality** | Input feature distributions vs. training baseline | Mean transaction amount shifted from $500 to $5000 |
| **Model Quality** | Prediction accuracy vs. ground truth (requires labels) | F1 score dropped from 0.91 to 0.72 |
| **Bias** | Fairness metrics across demographic groups | Model predicts differently for different customer segments |
| **Feature Attribution** | Feature importance drift via SHAP values | A previously important feature is now ignored |

For the FraudShield scenario, **Data Quality** monitoring would catch shifts in transaction patterns (e.g., a new type of fraud that produces different feature distributions).

### Step 4 -- Explore EventBridge

1. Navigate to **Amazon EventBridge** (search for "EventBridge" in the top search bar).
2. Click **Rules** in the left navigation.
3. EventBridge acts as the glue between monitoring and automation. Key patterns for MLOps:

| Event Source | Event | Target Action |
|-------------|-------|---------------|
| Model Monitor | Drift violation detected | Trigger retraining pipeline |
| Model Registry | New model version registered | Send notification via SNS |
| Model Registry | Model approved | Trigger deployment pipeline |
| SageMaker Pipeline | Execution completed | Send summary to Slack/email |

### Step 5 -- Walk Through a Hypothetical EventBridge Rule

If you were creating a rule (you will NOT create one, just understand the concept):

1. **Event pattern:** Match events from SageMaker where:
   - Source: `aws.sagemaker`
   - Detail type: `SageMaker Model Package State Change`
   - Detail: `ModelApprovalStatus` = `Approved`

2. **Target:** Start a pipeline execution or send an SNS notification

3. The rule would look like:

```json
{
  "source": ["aws.sagemaker"],
  "detail-type": ["SageMaker Model Package State Change"],
  "detail": {
    "ModelApprovalStatus": ["Approved"]
  }
}
```

This means: "Whenever someone approves a model version in the Registry, automatically trigger the next step (e.g., deploy it)."

### Step 6 -- Explore AWS CodePipeline (Brief Overview)

1. Navigate to **AWS CodePipeline** (search for "CodePipeline" in the top search bar).
2. CodePipeline is AWS's CI/CD orchestration service. For MLOps, it can:
   - Watch a Git repository for code changes
   - Trigger a SageMaker Pipeline execution when code is pushed
   - Run CodeBuild to package and deploy models
3. This creates the full MLOps automation: code change -> build -> train -> evaluate -> register -> approve -> deploy.
4. Browse the service page briefly -- you will not create a pipeline here, but understand that this is how teams achieve **Level 2 MLOps maturity** (fully automated training AND deployment).

### Step 7 -- Map the Complete MLOps Architecture

Document how all the services connect:

```
Code Change (Git)
    │
    ▼
CodePipeline ──> CodeBuild ──> SageMaker Pipeline
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
              PreprocessData   TrainModel     EvaluateModel
                                                    │
                                                    ▼
                                              CheckQuality
                                                    │
                                            ┌───────┴───────┐
                                            ▼               ▼
                                      RegisterModel      (skip)
                                            │
                                            ▼
                                    Model Registry
                                            │
                                    EventBridge (on Approved)
                                            │
                                            ▼
                                    Deploy to Endpoint
                                            │
                                            ▼
                                    Model Monitor
                                            │
                                    Drift Detected?
                                            │
                                    EventBridge (retrigger)
                                            │
                                            ▼
                                    SageMaker Pipeline (retrain)
```

---

## Presentation Checkpoint

Be prepared to show:
- The **Model Monitor** section in the console (even if empty) and explain what would be configured there
- The **EventBridge** section and explain how rules connect events to actions
- Your MLOps architecture diagram showing how all services connect
- Explain: What is the difference between Data Quality monitoring and Model Quality monitoring? (Data Quality checks input feature distributions against a baseline. Model Quality checks prediction accuracy against actual outcomes. Data Quality can detect problems before they affect predictions; Model Quality catches degradation after it happens.)
- Explain: How would you set up automatic retraining when data drift is detected? (Model Monitor detects drift -> publishes an event to EventBridge -> EventBridge rule triggers a SageMaker Pipeline execution -> pipeline retrains, evaluates, and registers a new model version)
- Explain: What are the three levels of MLOps maturity? (Level 0: everything is manual. Level 1: automated training pipeline, manual deployment. Level 2: fully automated training, evaluation, and deployment with monitoring-driven retraining.)

---

## Key Concepts

- **Model Monitor:** Continuously validates that live data and predictions match training expectations. Detects four types of drift: data quality, model quality, bias, and feature attribution.
- **EventBridge:** AWS's serverless event bus. Connects events from any AWS service to any target action. The "nervous system" of MLOps automation.
- **MLOps Maturity Levels:** A framework for assessing automation maturity. Most teams start at Level 0 (manual) and progressively automate toward Level 2 (fully automated with monitoring feedback loops).
- **Feedback Loop:** The complete cycle from monitoring to retraining: Monitor -> Detect drift -> Retrain -> Evaluate -> Register -> Deploy -> Monitor again. This loop keeps models fresh as data evolves.
