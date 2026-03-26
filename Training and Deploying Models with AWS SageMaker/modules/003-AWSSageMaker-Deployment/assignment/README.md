# AWSSageMaker-Deployment Lab

## Scenario

The FraudShield team has a trained model artifact in S3 from Module 2. Now you need to take it from artifact to live predictions: register it in the Model Registry with version tracking, approve it through a quality gate, deploy it to a real-time endpoint, invoke it with prediction requests, and clean everything up properly.

This lab continues from Module 2 -- you will use the model artifact and S3 bucket from your training job.

---

## Learning Objectives

By completing this lab you will demonstrate the ability to:

1. Create a Model Package Group and register a model version in the Model Registry
2. Update a model version's approval status from PendingManualApproval to Approved
3. Deploy a model using the three-object pattern (Model, Endpoint Configuration, Endpoint)
4. Invoke a live endpoint with prediction requests and interpret the response
5. Clean up all deployment resources in the correct order to prevent billing

---

## Prerequisites

- Completed Module 2 lab (trained model artifact in S3)
- The S3 URI of your `model.tar.gz` from Module 2
- The ECR image URI of the training container (visible on your training job's details page)
- Studio Domain still active from Module 1

---

## Milestones

| # | Guide | Estimated Time | What You Do |
|---|-------|---------------|-------------|
| 1 | [Register a Model in the Model Registry](console_guides/01_register_model.md) | 20 min | Create a package group and register a version |
| 2 | [Approve a Model Version](console_guides/02_approve_model_version.md) | 10 min | Update approval status, understand the workflow |
| 3 | [Deploy Using the Three-Object Pattern](console_guides/03_deploy_three_object_pattern.md) | 25 min | Create Model, Endpoint Config, Endpoint |
| 4 | [Invoke the Endpoint](console_guides/04_invoke_endpoint.md) | 20 min | Send predictions from a Studio notebook |
| 5 | [Clean Up in the Correct Order](console_guides/05_cleanup.md) | 15 min | Delete endpoint, config, model; verify billing |

**Total estimated time:** ~90 minutes

---

## Presentation Deliverables

When presenting your work, be prepared to show and explain:

1. Your Model Package Group and registered model version in the Model Registry
2. The approval status change from PendingManualApproval to Approved
3. All three deployment objects (Model, Endpoint Configuration, Endpoint) and how they reference each other
4. A successful prediction request and response from your live endpoint
5. Evidence that all resources have been cleaned up (no InService endpoints)
6. The correct cleanup order and why it matters

---

## Important Reminders

- **Free Tier:** Use `ml.m5.xlarge` for the endpoint. Do not select GPU instances.
- **Endpoint Charges:** Your endpoint will incur charges (~$0.23/hour) from the moment it reaches InService until you delete it. Complete the cleanup guide promptly.
- **Artifact from Module 2:** You need the S3 URI of your `model.tar.gz` and the container image URI. Both are available on your training job's details page in the SageMaker console.
