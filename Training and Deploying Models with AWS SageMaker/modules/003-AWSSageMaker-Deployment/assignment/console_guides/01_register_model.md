# Guide 1: Register a Model in the Model Registry

The Model Registry is SageMaker's governance layer for tracking model versions. Instead of managing artifacts by raw S3 paths, you register them as versioned **model packages** inside a **model package group**. This gives you version numbering, approval workflows, metadata tracking, and a single place to find all versions of a model.

---

## Steps

### Step 1 -- Gather Information from Module 2

Before starting, collect two pieces of information from your Module 2 training job:

1. Go to **SageMaker** -> **Training** -> **Training jobs**.
2. Click on your successful training job.
3. Note these values:
   - **Model data URL:** The S3 URI to `model.tar.gz` (found in Output data configuration)
   - **Training image:** The ECR container image URI (found in Algorithm specification / Container)

Write these down -- you will need them in the registration form.

### Step 2 -- Create a Model Package Group

1. In the **SageMaker console**, go to **Governance** -> **Model registry** in the left navigation.
2. Click **Create model package group**.
3. Configure:
   - **Name:** `fraud-detection-rf`
   - **Description:** `FraudShield Random Forest classifier for transaction fraud detection`
4. Optionally add tags:
   - **Key:** `project`, **Value:** `fraudshield`
   - **Key:** `algorithm`, **Value:** `random-forest`
5. Click **Create model package group**.

### Step 3 -- Register a Model Version

1. Click on the `fraud-detection-rf` group you just created.
2. Click **Register model** (or **Create model package**).
3. Fill in the registration form:

**Inference specification:**
   - **Container image:** Paste the **Training image** URI from Step 1 (e.g., `683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3`)
   - **Model data URL:** Paste the S3 URI to your `model.tar.gz` from Step 1
   - **Supported content types:** Add `text/csv` and `application/json`
   - **Supported response MIME types:** Add `text/csv` and `application/json`

**Model approval status:**
   - Select **PendingManualApproval** (this is the default and correct starting state)

**Description:**
   - Enter: `Version 1 - baseline RF model trained on initial FraudShield dataset`

4. Optionally add tags:
   - **Key:** `training-job`, **Value:** `<your-training-job-name>`
5. Click **Create model package**.

### Step 4 -- Verify the Registration

1. You should now see **Version 1** listed under the `fraud-detection-rf` group.
2. Click on the version to see its details:
   - **Version number:** 1
   - **Status:** PendingManualApproval (orange)
   - **Inference specification** with the container and artifact details you provided
3. Note how the Registry ties together the container image, the S3 artifact, and metadata -- everything needed to deploy this exact model version later.

---

## Presentation Checkpoint

Be prepared to show:
- The `fraud-detection-rf` Model Package Group in the Registry
- Version 1 with its status showing **PendingManualApproval**
- The inference specification showing the correct container image and model data URL
- Explain: What is the difference between a Model Package Group and a Model Package? (The group is a container for all versions of a model -- like a Git repository. A model package is a specific version -- like a Git tag or release.)
- Explain: Why does the model start in PendingManualApproval status? (This enforces a quality gate -- someone must review the model before it can be deployed to production. This prevents untested models from reaching customers.)

---

## Key Concepts

- **Model Package Group:** A logical container for all versions of a model. Similar to a project or repository.
- **Model Package (Version):** A specific, registered version of a model with its artifact, container, and metadata. Immutable once created.
- **Inference Specification:** The pairing of a container image (the runtime) and a model artifact (the weights/parameters). This is everything SageMaker needs to serve predictions.
- **PendingManualApproval:** The default status for new model versions. Enforces human review before deployment.

---

## AIML Connection

The *ML Lifecycle & Reproducibility* reading emphasized version control for models, not just code. The Model Registry is SageMaker's answer to this: every registered version captures the exact artifact, container, and training lineage, making it possible to reproduce or roll back any deployment.
