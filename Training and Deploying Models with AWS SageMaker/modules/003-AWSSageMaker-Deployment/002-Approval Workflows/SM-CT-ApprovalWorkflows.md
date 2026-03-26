# Approval Workflows

**Estimated Time:** 10 Minutes

## Introduction

In the *Model Registry & Versioning* reading you registered a model version and set its approval status to `PendingManualApproval`. That status is a gate: nothing downstream can deploy the model until someone changes it to `Approved`. But who makes that decision? Based on what criteria? And how do teams avoid bottlenecks where models sit pending for days because the right person did not notice?

Approval workflows answer these questions by defining the rules, roles, and automation around the transition from "trained" to "deployable." Think of it as the quality control station at the end of a manufacturing line. The product (your model) arrives with an inspection report (evaluation metrics). A quality inspector (a team lead, an automated check, or both) examines the report and either stamps it for shipping or sends it back for rework.

In the AIML Evaluation module you learned to evaluate models using precision, recall, F1, AUC-ROC, and confusion matrices. Those metrics are not just academic exercises -- they are the evidence your approval workflow consumes. A well-designed workflow specifies exactly which metrics must meet which thresholds before a model moves to production. This reading shows you how to design that workflow using the tools SageMaker and AWS provide.

## Core Concepts

### The approval lifecycle

Every model version in the Registry moves through a simple state machine:

```
PendingManualApproval  -->  Approved  -->  (Deployed)
         |
         v
      Rejected  -->  (Retrain / Investigate)
```

**PendingManualApproval** is the default starting state. It signals that the model has been registered but has not been reviewed. Deployment tools that read from the Registry can be configured to only deploy `Approved` versions, so a pending model cannot accidentally reach production.

**Approved** means the model has passed quality checks and is cleared for deployment. This status change can happen manually (a human clicks a button in the console) or automatically (a pipeline step checks metrics and updates the status programmatically).

**Rejected** means the model failed quality checks. It stays in the Registry as a record (you do not delete it), but it is marked as unsuitable for deployment. The team investigates what went wrong -- data quality issues, a regression in metrics, an overfitting problem -- and retrains.

### Manual approval in the console

The simplest approval workflow is a human reviewing metrics and clicking a button. Here is the console walkthrough:

1. **Navigate to the Model Registry.** In the SageMaker sidebar, click **Governance > Model registry**.
2. **Open the model package group** (e.g., `fraud-detection-rf`).
3. **Click the pending version.** You see its details page with the artifact S3 path, inference specification, and any metrics that were attached during registration.
4. **Review the metrics.** If the version includes model quality metrics (accuracy, F1, precision, recall), examine them. Compare against the previous approved version:
   - Is F1 higher or lower?
   - Has precision dropped below an acceptable threshold?
   - Does the confusion matrix show a new pattern of false positives or false negatives?

   These are the same evaluation questions you practiced in the AIML Evaluation module. The difference is that here the answers determine a deployment decision, not just a notebook observation.

5. **Update the status.** Click **Update status** on the version details page. Select **Approved** if the metrics meet your criteria, or **Rejected** if they do not. Optionally add an approval description explaining the rationale (e.g., "F1 improved from 0.87 to 0.91 on holdout set; precision stable at 0.93").
6. **Click Update.** The version status changes immediately and is visible to anyone with Registry access.

> **Tip:** Always add an approval description. Three months from now, when someone asks "why was version 4 approved over version 3?" the description is the audit trail. This is the ML equivalent of a meaningful Git commit message.

### Designing approval criteria

A manual click is only as good as the criteria behind it. Before approving any model, define clear thresholds. Here is a framework:

| Criterion | Example Threshold | Rationale |
| :--- | :--- | :--- |
| Primary metric meets minimum | F1 >= 0.85 | Below this, the model is not useful for the business case |
| Primary metric does not regress | F1(new) >= F1(current production) | A new model should not be worse than what is already serving |
| Secondary metrics are stable | Precision >= 0.80, Recall >= 0.75 | Guards against optimizing one metric at the expense of another |
| No data quality issues | Training data passes validation checks | Prevents models trained on corrupted or stale data |
| Bias checks pass (if applicable) | Disparate impact ratio within acceptable range | Ensures fairness across protected groups |

Write these thresholds down as a team agreement before you start training. When a model version arrives for review, you check it against the list. This removes ambiguity and makes approvals consistent regardless of who is reviewing.

### Comparing versions in the console

The Registry makes comparison straightforward:

1. **Open the group** and note the metrics for the currently approved version (e.g., version 2: F1 = 0.87, Precision = 0.92).
2. **Open the pending version** (e.g., version 3: F1 = 0.91, Precision = 0.93).
3. Compare each metric against your thresholds and against the current production version.
4. If all criteria pass, approve version 3. If any criterion fails, reject it with a description explaining which check failed.

For teams managing many models, you can also use **SageMaker Studio** to view Registry contents. Inside Studio, the Model Registry panel provides a visual interface for browsing groups, comparing versions, and updating statuses without switching to the SageMaker console.

### Automated approval with EventBridge

Manual approval works for small teams, but it creates a bottleneck when training runs happen frequently. AWS **EventBridge** lets you automate parts of the workflow by reacting to Registry events.

When a model version is created or its status changes, SageMaker emits an event to EventBridge. You can create EventBridge rules that trigger actions based on these events:

**Notification workflow:** When a new version is registered with `PendingManualApproval` status, EventBridge sends a notification (via SNS email, Slack webhook, or similar) to the reviewer. This eliminates the "nobody noticed it was pending" problem.

Here is how to set up a basic notification in the console:

1. **Open EventBridge.** Search "EventBridge" in the console search bar.
2. **Create a rule.** Click **Rules > Create rule**.
3. **Event pattern:** Select **SageMaker** as the service, and filter for **Model Package State Change** events.
4. **Target:** Select **SNS topic** and point it to a topic that emails your review team.
5. **Create the rule.**

Now, every time a model version is registered or its status changes, the team gets an email. The reviewer opens the console, examines the metrics, and approves or rejects.

**Automated gate workflow (advanced):** For teams with mature MLOps practices, the EventBridge target can be a **Lambda function** that automatically checks metrics against thresholds and updates the approval status programmatically. If all thresholds pass, the Lambda sets the status to `Approved`. If any fail, it sets `Rejected` and includes the failing criteria in the description. This fully automates the approval decision for straightforward cases while still allowing manual override for edge cases.

This automated pattern is a preview of the MLOps concepts you will explore in Module 4. For this module, focus on understanding the manual workflow and the EventBridge notification setup.

### Approval workflow patterns

Teams typically evolve through three maturity levels:

| Level | Workflow | When to Use |
| :--- | :--- | :--- |
| **Manual** | Human reviews metrics in console, clicks Approve/Reject | Small teams, early-stage projects, learning environments |
| **Notified** | EventBridge sends alert when version is pending; human still approves | Medium teams, regular retraining cadence |
| **Automated** | Lambda checks metrics against thresholds; auto-approves or flags for human review | Mature MLOps, frequent retraining, CI/CD pipelines |

For this curriculum, you will practice the manual and notified patterns. The automated pattern is covered conceptually here and will appear again in the *MLOps & CI/CD Principles* topic in Module 4.

### SDK equivalents

Updating approval status programmatically:

```python
import boto3

sm_client = boto3.client("sagemaker")

sm_client.update_model_package(
    ModelPackageArn="arn:aws:sagemaker:us-east-1:123456789012:model-package/fraud-detection-rf/3",
    ModelApprovalStatus="Approved",
    ApprovalDescription="F1 improved to 0.91; all thresholds met.",
)
```

Querying for the latest approved version:

```python
response = sm_client.list_model_packages(
    ModelPackageGroupName="fraud-detection-rf",
    ModelApprovalStatus="Approved",
    SortBy="CreationTime",
    SortOrder="Descending",
    MaxResults=1,
)
latest_approved = response["ModelPackageSummaryList"][0]
```

This query is how deployment pipelines find the right model to deploy: "give me the most recent approved version." The Registry becomes the single source of truth for what is production-ready.

## Connecting to Practice

This reading gives you the framework for designing approval workflows. In the *Approval Workflows Video*, you will see a live demonstration of manual approval and EventBridge notification setup. In the next readings, *Real-time Inference Endpoints* and *Invoking Endpoints*, you will deploy an approved model version and send it predictions. And in the module assignment, you will build a workflow that registers, approves, and deploys a model end to end.

The most useful thing you can do right now is write down three to five approval criteria for a model you have trained (or plan to train). What metric must exceed what threshold? Must it beat the current version? Are there fairness or data quality checks? Having these criteria defined before you reach the deployment step is what separates ad-hoc experimentation from governed ML.

---

## Further Learning & Resources

**Documentation and reading**

- **[Model Registry Approval Workflows](https://docs.aws.amazon.com/sagemaker/latest/dg/model-registry-approve.html)** - *Docs*: The official guide covering approval status management, including programmatic updates and cross-account approval patterns.
- **[Amazon EventBridge with SageMaker](https://docs.aws.amazon.com/sagemaker/latest/dg/automating-sagemaker-with-eventbridge.html)** - *Docs*: How SageMaker emits events to EventBridge, with event pattern examples for model registry state changes.

**Interactive practice**

- **[AWS Hands-On: EventBridge Getting Started](https://aws.amazon.com/getting-started/hands-on/amazon-eventbridge-getting-started/)** - *Interactive*: A free guided lab teaching EventBridge rule creation and target configuration in the console.
- **[SageMaker MLOps Workshop](https://catalog.workshops.aws/sagemaker-mlops/en-US)** - *Interactive*: A workshop-style environment covering end-to-end MLOps workflows including model registry and approval automation.
