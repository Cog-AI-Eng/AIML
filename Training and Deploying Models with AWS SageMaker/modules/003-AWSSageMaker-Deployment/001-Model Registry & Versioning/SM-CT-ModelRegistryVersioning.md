# Model Registry & Versioning

**Estimated Time:** 10 Minutes

## Introduction

In Module 2 you trained models and produced artifacts -- `model.tar.gz` files sitting in S3. You know how to find them, download them, and inspect them. But a team that produces dozens or hundreds of training runs faces a new problem: which artifact is the "good" one? Which version is currently deployed? Which one was trained on last week's data with the updated feature engineering?

S3 stores files, but it does not answer those questions. You could name your training jobs carefully, keep a spreadsheet, or tag S3 objects -- but none of that scales, and none of it integrates with SageMaker's deployment tools. The **SageMaker Model Registry** solves this by providing a versioned catalog of model artifacts with metadata, approval status, and a direct connection to deployment.

If you recall the *ML Lifecycle & Reproducibility* reading from the AIML Foundations module, you learned that Git provides version control for code -- every commit captures a snapshot that you can trace, compare, and roll back to. The Model Registry does the same thing for models. Where Git tracks code changes across commits, the Model Registry tracks model versions across training runs. Where Git branches let you manage parallel development, the Registry's approval workflows let you manage the progression from "experimental" to "production-approved."

This reading walks you through the Model Registry in the SageMaker console: creating a model package group, registering a model version, and managing versions with metadata and tags.

## Core Concepts

### Model Registry structure

The Registry organizes models into two levels:

**Model Package Groups** are named collections that hold all versions of a particular model. Think of a group as a Git repository: it represents one logical model (e.g., "fraud-detection-classifier" or "demand-forecaster") regardless of how many times you retrain it. The group name is the stable identifier that deployment pipelines reference.

**Model Packages** (versions) are individual entries within a group. Each package points to a specific `model.tar.gz` artifact in S3, records metadata (framework, instance type, training job name), and carries an **approval status** that gates deployment. Think of a package as a Git commit: it captures a specific snapshot of the model at a point in time.

| Concept | Git Equivalent | Purpose |
| :--- | :--- | :--- |
| Model Package Group | Repository | Named container for all versions of one model |
| Model Package (Version) | Commit | Specific snapshot with artifact, metadata, and status |
| Approval Status | Code review / merge approval | Gates whether a version can be deployed |

### Finding the Model Registry in the console

1. **Navigate to SageMaker.** Sign in to the AWS Console and open SageMaker.
2. **Open the Registry.** In the left sidebar, click **Governance > Model registry** (or **Home > Models > Model registry** depending on your console version).
3. **The Registry page** shows a list of all Model Package Groups in your account. If this is your first time, the list will be empty. Each group displays its name, creation date, and the number of versions registered.

### Creating a Model Package Group

A group must exist before you can register model versions into it. Here is the console walkthrough:

1. On the **Model registry** page, click **Create model package group**.
2. **Group name:** Enter a descriptive, stable name (e.g., `fraud-detection-rf`). This name will be referenced by deployment pipelines and approval workflows, so choose something meaningful and consistent. Avoid timestamps or experiment IDs in the group name -- those belong on individual versions.
3. **Description:** Add a brief description of what this model does (e.g., "Random forest classifier for transaction fraud detection, trained on tabular transaction data").
4. **Tags:** Add key-value pairs for organizational metadata. Useful tags include:
   - `project`: `fraud-detection`
   - `team`: `data-science`
   - `framework`: `scikit-learn`

   Tags help you filter and search groups when you have many models registered. They also support the tag-based IAM access control pattern you learned about in the *IAM & Least-Privilege Practices* reading.
5. Click **Create model package group**. The group appears in the Registry list immediately.

### Registering a model version

Once you have a group, you can register a trained model artifact as a new version. Here is the console walkthrough:

1. **Open the group.** On the Model registry page, click the group name (e.g., `fraud-detection-rf`). You see a list of registered versions (empty initially).
2. **Click Register model version** (or **Create model package** depending on console version).
3. **Inference specification section:**
   - **Container image:** The ECR URI of the inference container that will serve this model. For scikit-learn Script Mode, use the same managed container image you used for training (e.g., the scikit-learn 1.2-1 image). SageMaker uses this container to load and serve your model at deployment time.
   - **Model data URL:** The S3 URI of the `model.tar.gz` artifact (e.g., `s3://my-bucket/training-output/rf-classifier-2026-03-22/output/model.tar.gz`). This is the artifact you located in the *Model Artifacts & S3 Storage* reading.
   - **Supported instance types:** Select the instance types this model can be deployed on (e.g., `ml.m5.xlarge`). This metadata helps downstream deployment tools choose appropriate resources.
4. **Model metrics (optional):** You can attach evaluation metrics to the version:
   - **Model quality metrics:** Accuracy, precision, recall, F1 -- the metrics you learned about in the AIML Evaluation module.
   - **Bias and explainability reports:** If you ran SageMaker Clarify evaluations.

   These metrics are stored as metadata on the version and are visible in the console, making it easy to compare versions side by side.
5. **Approval status:** Set the initial status:
   - **PendingManualApproval** -- the version is registered but not yet cleared for deployment. This is the recommended default for any model that will serve live traffic.
   - **Approved** -- the version is immediately deployable. Use this for development/testing scenarios where you want to skip the approval step.
   - **Rejected** -- marks the version as unsuitable. Useful for documenting versions that failed quality checks.
6. **Tags:** Add version-specific tags (e.g., `training-job`: `rf-v2-feature-scaling-2026-03-22`, `dataset-version`: `v3`).
7. Click **Create model package**. The version appears in the group's version list with a version number (auto-incremented: 1, 2, 3, ...).

### Viewing and comparing versions

After registering multiple versions, the group page becomes a version history:

1. **Open the group** from the Model registry page.
2. **Version list:** Each row shows the version number, approval status, creation date, and the S3 artifact path. Click any version to see its full details.
3. **Version details page:** Shows the inference specification, model metrics, tags, and approval status. This is where you compare a new version's metrics against a previous one to decide whether to approve it.

> **Tip:** When comparing versions, look at the model metrics side by side. If version 3 has higher F1 than version 2 but lower precision, the approval decision depends on your use case. The Registry does not make this decision for you -- it gives you the data to make it yourself. This connects to the evaluation trade-offs you studied in the AIML Evaluation module (*Metrics: Precision, Recall, F1* and *AUC-ROC & Confusion Matrix*).

### Updating approval status

Approval status is the gate between training and deployment. To update it in the console:

1. Open the version details page.
2. Click **Update status** (or the edit icon next to the approval status).
3. Select the new status (**Approved**, **Rejected**, or **PendingManualApproval**).
4. Click **Update**.

The next topic, *Approval Workflows*, covers how teams formalize this process with automated checks and human review gates. For now, the key concept is that the Registry enforces a clear signal: a model is either approved for deployment or it is not.

### SDK equivalents

The SDK provides programmatic access to all Registry operations:

```python
import boto3

sm_client = boto3.client("sagemaker")

sm_client.create_model_package_group(
    ModelPackageGroupName="fraud-detection-rf",
    ModelPackageGroupDescription="RF classifier for fraud detection",
    Tags=[{"Key": "project", "Value": "fraud-detection"}],
)

sm_client.create_model_package(
    ModelPackageGroupName="fraud-detection-rf",
    InferenceSpecification={
        "Containers": [{
            "Image": "123456789012.dkr.ecr.us-east-1.amazonaws.com/sklearn:1.2-1",
            "ModelDataUrl": "s3://my-bucket/output/rf-classifier/output/model.tar.gz",
        }],
        "SupportedContentTypes": ["text/csv"],
        "SupportedResponseMIMETypes": ["text/csv"],
        "SupportedRealtimeInferenceInstanceTypes": ["ml.m5.xlarge"],
    },
    ModelApprovalStatus="PendingManualApproval",
)
```

After training, you can also register directly from the Estimator:

```python
estimator.register(
    model_package_group_name="fraud-detection-rf",
    content_types=["text/csv"],
    response_types=["text/csv"],
    inference_instances=["ml.m5.xlarge"],
    approval_status="PendingManualApproval",
)
```

This is more concise because the Estimator already knows the artifact location and container image from the training job it just ran.

## Connecting to Practice

This reading introduces the Model Registry as the bridge between training and deployment. In the *Model Registry & Versioning Video*, you will see a live walkthrough of creating a group and registering versions. The next reading, *Approval Workflows*, explains how teams build formal review processes around the approval status. And in the module assignment, you will register models and manage versions as part of a deployment workflow.

The most useful thing you can do right now is open the SageMaker console, navigate to **Governance > Model registry**, and create a Model Package Group. If you have a completed training job from Module 2, register its artifact as version 1 with `PendingManualApproval` status. Seeing the version appear in the Registry makes the concept concrete.

---

## Further Learning & Resources

**Documentation and reading**

- **[SageMaker Model Registry](https://docs.aws.amazon.com/sagemaker/latest/dg/model-registry.html)** - *Docs*: The official guide covering group creation, version registration, approval workflows, and cross-account registry sharing.
- **[Model Registry and MLOps](https://docs.aws.amazon.com/sagemaker/latest/dg/model-registry-deploy.html)** - *Docs*: How the Model Registry integrates with SageMaker Pipelines and deployment automation for production workflows.

**Interactive practice**

- **[AWS Hands-On: Register and Deploy Models](https://aws.amazon.com/getting-started/hands-on/machine-learning-tutorial-mlops-automate-data-processing-train-model/)** - *Interactive*: A free guided lab covering model registration and deployment in the console.
- **[SageMaker Examples - Model Registry](https://github.com/aws/amazon-sagemaker-examples/tree/main/sagemaker-pipelines)** - *Interactive*: Runnable notebooks demonstrating Model Registry operations within automated pipeline workflows.
