# Guide 2: Approve a Model Version

In a real MLOps workflow, model versions go through an approval process before deployment. A data scientist trains the model, an ML lead reviews the metrics, and only after approval can the model be deployed to a production endpoint. In this guide, you will manually approve the model version you registered and understand the approval workflow.

---

## Steps

### Step 1 -- Review the Model Before Approving

1. In the **SageMaker console**, go to **Governance** -> **Model registry**.
2. Click on `fraud-detection-rf`.
3. Click on **Version 1**.
4. Before approving, review the details as if you were an ML lead evaluating this model:
   - **Container image:** Is it the correct framework and version?
   - **Model data URL:** Does the S3 path point to a valid artifact from a known training job?
   - **Tags:** Can you trace this back to the specific training job?
   - **Description:** Does it explain what this version contains?

In a production workflow, you would also review:
- Training metrics (accuracy, F1 score, AUC-ROC)
- Comparison against the current production model
- Test results on held-out data
- Data quality and drift reports

### Step 2 -- Update the Approval Status

1. On the model version details page, click **Update status** (or look for an **Edit** or **Update model package** option).
2. Change the status from **PendingManualApproval** to **Approved**.
3. Add an approval description: `Approved for initial deployment. Baseline model for FraudShield fraud detection.`
4. Click **Update** or **Save**.

### Step 3 -- Verify the Status Change

1. Return to the `fraud-detection-rf` group listing.
2. Version 1 should now show **Approved** (green) instead of PendingManualApproval (orange).
3. Click on the version again and confirm the status and description are updated.

### Step 4 -- Understand the Approval States

SageMaker supports three approval states:

| Status | Meaning | When to Use |
|--------|---------|-------------|
| **PendingManualApproval** | Default. Awaiting review. | Automatically set on registration |
| **Approved** | Cleared for deployment. | After human review confirms quality |
| **Rejected** | Does not meet quality standards. | When metrics are below threshold or issues found |

In production, the transition from PendingManualApproval to Approved would typically involve:
1. Automated metric checks (e.g., "F1 must be >= 0.85")
2. Human review and sign-off
3. Optionally, automated notifications via EventBridge and SNS

### Step 5 -- Understand the Approval Workflow Concept

Consider how this maps to a team workflow:

```
Data Scientist trains model
        │
        ▼
Model registered → PendingManualApproval
        │
        ▼
ML Lead reviews metrics and artifacts
        │
   ┌────┴────┐
   ▼         ▼
Approved   Rejected
   │
   ▼
Ready for deployment (Module 3, Guide 3)
```

The approval gate prevents untested models from reaching production. Even in automated pipelines (Module 4), a ConditionStep can programmatically check metrics before registration, and a human can still manually approve/reject in the console.

---

## Presentation Checkpoint

Be prepared to show:
- Version 1 with **Approved** status in the Model Registry
- The approval description you added
- Explain: What are the three approval states and when would you use each? (PendingManualApproval for new registrations, Approved after review, Rejected when quality is insufficient)
- Explain: In a production team, who would approve models and what would they check? (ML leads or senior engineers would verify metrics meet thresholds, check for regression against the current production model, and review data quality)
- Explain: How could this approval process be partially automated? (EventBridge can trigger notifications when a model is registered; a Lambda function can check metrics programmatically; but final approval can still require human sign-off)

---

## Key Concepts

- **Approval Workflow:** A governance process that prevents models from reaching production without review. Implemented via the three approval states in the Model Registry.
- **Quality Gate:** A threshold or set of criteria that a model must meet before being approved (e.g., F1 >= 0.85, no regression > 5% from baseline).
- **Separation of Concerns:** The person who trains the model is not necessarily the person who approves it. This separation reduces the risk of deploying flawed models.
