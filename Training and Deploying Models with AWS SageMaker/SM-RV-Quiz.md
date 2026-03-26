# AWS SageMaker Review Quiz

**Activity Name:** SM-RV-Quiz
**Display Name:** AWS SageMaker Review Quiz
**Duration:** 30 min
**Total Questions:** 30
**Question Types:** Multiple Choice (22), True/False (5), Matching (3)
**Difficulty Distribution:** Beginner (9), Intermediate (13), Advanced (8)

---

## Questions

---

### Question 1 -- MCQ | Module 1 | Beginner | Conceptual

**Which AWS service provides the persistent storage layer that backs SageMaker Studio notebooks, enabling users to resume work after stopping and restarting a kernel?**

A) Amazon EFS (Elastic File System)
B) Amazon DynamoDB
C) Amazon RDS
D) Amazon S3

**Correct Answer: A**

**Rationale:** SageMaker Studio domains provision an Amazon EFS volume that stores user notebooks, files, and Git repos. This EFS-backed storage persists even when compute instances are stopped. Amazon S3 (D) is used for data and model artifact storage, not for live notebook persistence. RDS (C) and DynamoDB (B) are database services unrelated to notebook file systems.

---

### Question 2 -- MCQ | Module 1 | Intermediate | Scenario-Based

**A team lead wants to onboard three personas to SageMaker: a data engineer who prefers JupyterLab, a citizen data scientist who has no coding experience, and an MLOps engineer who needs full IDE control. Which combination of SageMaker interfaces best serves each persona?**

A) SageMaker Studio for all three users
B) Studio for the data engineer, Studio Classic for the citizen data scientist, Canvas for the MLOps engineer
C) Studio Classic for the data engineer, Canvas for the citizen data scientist, Studio for the MLOps engineer
D) Canvas for all three users

**Correct Answer: C**

**Rationale:** Studio Classic provides the familiar JupyterLab interface preferred by data engineers. Canvas offers a no-code, visual interface ideal for citizen data scientists. Studio provides the newest, full-featured IDE with integrated MLOps tools for the MLOps engineer. Assigning all users to Studio (A) or Canvas (D) ignores persona-specific needs. Option B incorrectly puts the no-code user in Studio Classic, which requires coding.

---

### Question 3 -- MCQ | Module 1 | Beginner | Conceptual

**Which of the following correctly lists the five stages of the SageMaker ML Lifecycle in order?**

A) Build, Prepare, Deploy, Train & Tune, Monitor
B) Prepare, Build, Train & Tune, Deploy, Monitor
C) Train & Tune, Prepare, Build, Monitor, Deploy
D) Prepare, Train & Tune, Build, Deploy, Monitor

**Correct Answer: B**

**Rationale:** The SageMaker ML Lifecycle follows the sequence Prepare (data ingestion and labeling), Build (feature engineering and algorithm selection), Train & Tune (model training and hyperparameter tuning), Deploy (endpoint creation), and Monitor (drift detection and retraining triggers). The other options scramble this order, which would violate the data-first principle where preparation must precede model building.

---

### Question 4 -- MCQ | Module 1 | Intermediate | Scenario-Based

**A developer deploys a text summarization model from SageMaker JumpStart through the console. After deployment completes, she navigates to the SageMaker console and notices three new resources were created automatically. What are these three resources?**

A) A Training Job, a Processing Job, and an Endpoint
B) An S3 Bucket, a CloudWatch Alarm, and an Endpoint
C) A Lambda Function, an API Gateway, and an Endpoint
D) A Model, an Endpoint Configuration, and an Endpoint

**Correct Answer: D**

**Rationale:** JumpStart follows SageMaker's three-object deployment pattern: it creates a Model (container image + artifact reference), an Endpoint Configuration (instance type and count), and an Endpoint (the live inference resource). Training Jobs (A) are created during model training, not deployment. S3 Buckets (B) and Lambda Functions (C) are not automatically created by JumpStart deployments.

---

### Question 5 -- T/F | Module 1 | Beginner

**True or False: A SageMaker Execution Role is an IAM User that SageMaker logs into to access AWS resources like S3 and CloudWatch on your behalf.**

**Correct Answer: False**

**Rationale:** A SageMaker Execution Role is an IAM **Role**, not an IAM User. SageMaker assumes this role via `sts:AssumeRole` using a trust policy that allows `sagemaker.amazonaws.com` as the principal. IAM Users have permanent credentials (access keys), whereas IAM Roles provide temporary credentials that are automatically rotated, making roles the preferred mechanism for service-to-service access.

---

### Question 6 -- MCQ | Module 1 | Advanced | Scenario-Based

**An instructor asks you to create a custom IAM policy for a SageMaker Execution Role that can only read training data from a specific S3 bucket called `fraudshield-data-123` and write model artifacts to `fraudshield-models-123`. Which policy statement best implements least-privilege?**

A) Allow `s3:GetObject` on `arn:aws:s3:::fraudshield-data-123/*` and `s3:PutObject` on `arn:aws:s3:::fraudshield-models-123/*`, plus `s3:ListBucket` on both bucket ARNs
B) Allow `s3:*` on `arn:aws:s3:::*`
C) Allow `s3:GetObject` and `s3:PutObject` on `arn:aws:s3:::fraudshield-*`
D) Allow `s3:GetObject` on `arn:aws:s3:::fraudshield-data-123/*` and `s3:PutObject` on `arn:aws:s3:::fraudshield-models-123/*`

**Correct Answer: A**

**Rationale:** Least-privilege means granting only the permissions needed. Option A correctly scopes `GetObject` to the data bucket's objects, `PutObject` to the models bucket's objects, and adds `ListBucket` on the bucket ARNs themselves (required for SageMaker to enumerate objects during training). Option B grants full S3 access everywhere. Option D is close but missing `ListBucket`, which causes `Access Denied` errors when SageMaker tries to list the training channel. Option C uses a wildcard prefix that could match unintended buckets.

---

### Question 7 -- MCQ | Module 1 | Beginner | Conceptual

**What is the primary purpose of SageMaker JumpStart?**

A) To provide pre-configured VPC networking for SageMaker resources
B) To automatically generate training scripts from raw datasets
C) To offer a catalog of pre-trained, fine-tunable foundation models and ML solutions deployable with minimal configuration
D) To manage IAM policies for SageMaker users

**Correct Answer: C**

**Rationale:** JumpStart is SageMaker's model hub offering pre-trained models (including foundation models for text, image, and tabular tasks) and ML solutions that can be deployed or fine-tuned with a few clicks. It does not configure VPCs (A), auto-generate scripts (B), or manage IAM (D).

---

### Question 8 -- Matching | Module 1 | Intermediate

**Match each SageMaker service to the ML lifecycle stage it primarily supports.**

| # | Service | | Stage |
|---|---------|---|-------|
| 1 | SageMaker Data Wrangler | | A) Monitor |
| 2 | SageMaker JumpStart | | B) Train & Tune |
| 3 | SageMaker Experiments | | C) Prepare |
| 4 | SageMaker Model Monitor | | D) Build |

**Correct Answers: 1-C, 2-D, 3-B, 4-A**

**Rationale:** Data Wrangler handles data preparation and feature engineering (Prepare). JumpStart provides pre-built models and solutions that accelerate the Build stage. Experiments tracks and compares training runs (Train & Tune). Model Monitor continuously evaluates deployed model quality (Monitor).

---

### Question 9 -- MCQ | Module 2 | Beginner | Conceptual

**What is the key difference between Bring-Your-Own-Model (BYOM) and Bring-Your-Own-Script (Script Mode) in SageMaker?**

A) BYOM uses SageMaker-managed containers while Script Mode uses custom Docker containers
B) BYOM requires packaging a complete Docker container with your model, while Script Mode injects your training script into a SageMaker-managed container
C) BYOM is for inference only, while Script Mode is for training only
D) There is no difference; they are two names for the same approach

**Correct Answer: B**

**Rationale:** BYOM requires you to build and push a complete Docker image (with all dependencies, serving logic, and the model) to Amazon ECR. Script Mode lets you write just a Python training script; SageMaker provides the pre-built container with the framework (scikit-learn, PyTorch, etc.) already installed. Option A reverses the container responsibility. Option C is incorrect because both can be used for training and inference. They are distinct approaches (D).

---

### Question 10 -- MCQ | Module 2 | Intermediate | Scenario-Based

**A data scientist writes a Script Mode training script but the training job fails immediately with a `FileNotFoundError` when trying to read the training data. The script uses `pd.read_csv("/data/train.csv")`. What is the most likely cause?**

A) The S3 bucket does not contain a file named `train.csv`
B) The instance type selected does not support CSV files
C) The training job needs a VPC endpoint to access local files
D) The script should read from the SageMaker-managed path `/opt/ml/input/data/<channel_name>/` instead of `/data/`

**Correct Answer: D**

**Rationale:** SageMaker downloads training data from S3 to `/opt/ml/input/data/<channel_name>/` inside the container. Scripts must read from this managed path (accessed via the `SM_CHANNEL_<NAME>` environment variable or `argparse`). Hardcoding `/data/train.csv` points to a path that does not exist in the container. While A could also cause failures, the error pattern described (immediate `FileNotFoundError` with a hardcoded path) points to incorrect path usage. Instance types (B) do not restrict file formats, and VPC endpoints (C) are for network traffic, not local file access.

---

### Question 11 -- MCQ | Module 2 | Advanced | Code-Reading

**Review the following Script Mode training script. Which line contains an error that will cause the training job to fail?**

```python
import argparse
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-estimators", type=int, default=100)
    args = parser.parse_args()

    train_dir = os.environ["SM_CHANNEL_TRAIN"]
    df = pd.read_csv(os.path.join(train_dir, "train.csv"))

    X = df.drop("label", axis=1)
    y = df["label"]

    model = RandomForestClassifier(n_estimators=args.n_estimators)
    model.fit(X, y)

    joblib.dump(model, "model.joblib")
```

A) The `argparse` argument uses a hyphen instead of an underscore
B) The `SM_CHANNEL_TRAIN` environment variable is incorrect
C) The model is saved to the current directory instead of `os.environ["SM_MODEL_DIR"]`
D) The `if __name__` guard is unnecessary in Script Mode

**Correct Answer: C**

**Rationale:** SageMaker expects model artifacts to be saved to the path specified by `SM_MODEL_DIR` (which defaults to `/opt/ml/model/`). Saving to the current working directory means SageMaker will not find the artifact to upload to S3, and the training job will produce no usable model. Hyphens in argparse arguments (A) are valid and automatically converted to underscores. `SM_CHANNEL_TRAIN` (B) is the correct environment variable. The `if __name__` guard (D) is required by Script Mode convention.

---

### Question 12 -- MCQ | Module 2 | Advanced | Scenario-Based

**A training job runs for 2 hours and completes successfully, but when you check the S3 output path, the `model.tar.gz` file is only 200 bytes. What is the most likely explanation?**

A) The training script saved an empty or near-empty file to `SM_MODEL_DIR`
B) SageMaker compresses model artifacts, so 200 bytes is normal for small models
C) The S3 upload was interrupted by a network timeout
D) The training instance ran out of disk space

**Correct Answer: A**

**Rationale:** SageMaker packages everything in `SM_MODEL_DIR` (`/opt/ml/model/`) into `model.tar.gz` and uploads it to S3. A 200-byte tar.gz typically indicates the directory was empty or contained only a trivially small file (e.g., the script saved metadata but not the actual model weights). If the instance ran out of disk (D), the job would likely fail. SageMaker does use tar.gz compression (B), but a properly trained model would be significantly larger. S3 upload interruptions (C) would cause a job failure, not a successful completion with a tiny file.

---

### Question 13 -- T/F | Module 2 | Beginner

**True or False: When SageMaker runs a Script Mode training job, it provisions a fresh compute instance, pulls the specified Docker container, downloads training data from S3, executes your script, uploads model artifacts to S3, and then terminates the instance.**

**Correct Answer: True**

**Rationale:** This accurately describes the anatomy of a SageMaker Training Job. The fully managed lifecycle includes provisioning, container setup, data download, script execution, artifact upload, and automatic teardown. This pay-per-use model means you are only billed for the time the training instance is running, and no infrastructure persists after the job completes.

---

### Question 14 -- MCQ | Module 2 | Advanced | Scenario-Based

**A training job fails and the console shows the status `Failed`. You open the training job detail page but the error message is truncated. Where should you look for the complete error output and stack trace?**

A) The SageMaker Model Registry
B) Amazon CloudWatch Logs, under the `/aws/sagemaker/TrainingJobs` log group
C) The S3 output path under `model.tar.gz`
D) The SageMaker Experiments dashboard

**Correct Answer: B**

**Rationale:** SageMaker streams all `stdout` and `stderr` from the training container to CloudWatch Logs under the `/aws/sagemaker/TrainingJobs` log group. Each training job creates its own log stream where you can find the full stack trace. The Model Registry (A) stores model metadata, not logs. The S3 output path (C) would not contain artifacts from a failed job. Experiments (D) track metrics and parameters but do not store raw container logs.

---

### Question 15 -- MCQ | Module 2 | Intermediate | Conceptual

**After a successful SageMaker training job, where is the `model.tar.gz` artifact stored in S3?**

A) `s3://<bucket>/models/<model-name>.tar.gz`
B) `s3://<bucket>/sagemaker/artifacts/latest/model.tar.gz`
C) `s3://<bucket>/<training-job-name>/model.tar.gz`
D) `s3://<bucket>/<output-path-prefix>/<training-job-name>/output/model.tar.gz`

**Correct Answer: D**

**Rationale:** SageMaker follows a deterministic output path pattern: the base S3 URI you specify as the output path, followed by the training job name, then `/output/model.tar.gz`. This structure enables implicit versioning since each training job has a unique name. The other options show incorrect path patterns that do not match SageMaker's convention.

---

### Question 16 -- Matching | Module 2 | Intermediate

**Match each Script Mode concept to its correct description.**

| # | Concept | | Description |
|---|---------|---|-------------|
| 1 | `SM_MODEL_DIR` | | A) CLI-style argument for passing tuning values to the training script |
| 2 | `SM_CHANNEL_TRAIN` | | B) The directory where SageMaker expects the trained model to be saved |
| 3 | Hyperparameter | | C) The local path where SageMaker downloads the training dataset |
| 4 | `if __name__ == "__main__":` | | D) Python guard ensuring the script runs as a main program, not an import |

**Correct Answers: 1-B, 2-C, 3-A, 4-D**

**Rationale:** `SM_MODEL_DIR` points to `/opt/ml/model/`, where saved artifacts are packaged and uploaded. `SM_CHANNEL_TRAIN` points to `/opt/ml/input/data/train/`, where SageMaker downloads training data. Hyperparameters are passed as command-line arguments that `argparse` parses. The main guard ensures the training logic executes when SageMaker invokes the script directly.

---

### Question 17 -- MCQ | Module 3 | Beginner | Conceptual

**What is the primary purpose of the SageMaker Model Registry?**

A) To store raw training datasets for model reproducibility
B) To automatically retrain models when new data arrives
C) To provide a central catalog for versioning, tracking, and governing trained model artifacts
D) To host real-time inference endpoints

**Correct Answer: C**

**Rationale:** The Model Registry is a governance and versioning hub where teams register model versions (Model Packages) under logical groups (Model Package Groups), track metadata, and manage approval workflows. It does not store raw data (A), trigger retraining (B), or host endpoints (D).

---

### Question 18 -- MCQ | Module 3 | Intermediate | Scenario-Based

**A data scientist registers a new model version in the Model Registry with a status of `PendingManualApproval`. The MLOps engineer reviews the model metrics and changes the status to `Approved`. What happens automatically when the status changes to `Approved`?**

A) Nothing happens automatically; the approval status is metadata that downstream automation (e.g., EventBridge rules or Pipelines) can react to
B) SageMaker automatically deploys the model to a real-time endpoint
C) The model artifact is automatically copied to a production S3 bucket
D) SageMaker sends an email notification to all Studio users

**Correct Answer: A**

**Rationale:** The approval status in the Model Registry is a governance metadata field. SageMaker does not take any automatic action when it changes. However, teams can build automation around it: EventBridge can detect the status change and trigger a deployment pipeline, or a SageMaker Pipeline's `ConditionStep` can check approval status. There is no built-in auto-deployment (B), auto-copy (C), or email notification (D).

---

### Question 19 -- MCQ | Module 3 | Advanced | Scenario-Based

**You deploy a model to a real-time endpoint and invoke it with a CSV payload, but the endpoint returns an `InternalServerError (500)`. The model worked correctly during local testing. What is the most productive first debugging step?**

A) Delete the endpoint and redeploy with a larger instance type
B) Check CloudWatch Logs for the endpoint's container logs to identify the stack trace
C) Retrain the model with a different algorithm
D) Change the `ContentType` from `text/csv` to `application/json`

**Correct Answer: B**

**Rationale:** A 500 error from an endpoint indicates the model container crashed or raised an unhandled exception. CloudWatch Logs (under `/aws/sagemaker/Endpoints/<endpoint-name>`) contain the container's `stdout` and `stderr`, which will show the exact error. Scaling up (A) would not fix a code-level error. Retraining (C) is premature without understanding the root cause. Changing `ContentType` (D) might help if the issue is a deserialization mismatch, but checking logs first provides the diagnostic evidence needed.

---

### Question 20 -- MCQ | Module 3 | Intermediate | Code-Reading

**What does the following `boto3` code do?**

```python
import boto3
runtime = boto3.client("sagemaker-runtime")
response = runtime.invoke_endpoint(
    EndpointName="fraud-rf-v1-endpoint",
    ContentType="text/csv",
    Body="2500.00,3,1,3,45,8",
)
result = response["Body"].read().decode("utf-8")
```

A) Deploys a new model called `fraud-rf-v1-endpoint`
B) Creates a batch transform job for the fraud model
C) Registers a new model version in the Model Registry
D) Sends a CSV-formatted inference request to a live SageMaker endpoint and reads the prediction response

**Correct Answer: D**

**Rationale:** The code uses the `sagemaker-runtime` client's `invoke_endpoint` method to send a real-time inference request. The `Body` contains a single CSV row of feature values, and the response is read and decoded to get the prediction. This is not a deployment action (A), a batch job (B), or a registry operation (C).

---

### Question 21 -- T/F | Module 3 | Beginner

**True or False: When you deploy a model to a SageMaker real-time endpoint through the console, SageMaker creates three resources: a Model, an Endpoint Configuration, and an Endpoint.**

**Correct Answer: True**

**Rationale:** SageMaker's deployment follows the three-object pattern. The Model defines the container image and S3 artifact location. The Endpoint Configuration specifies the instance type, count, and which Model to use. The Endpoint is the live HTTPS resource that serves predictions. All three must exist and be deleted during cleanup.

---

### Question 22 -- MCQ | Module 3 | Advanced | Scenario-Based

**After completing a lab, a student deletes the SageMaker Endpoint but forgets to delete the Endpoint Configuration and the Model. Which of the following is true?**

A) The student will continue to be billed for the Endpoint Configuration and Model at the same rate as the Endpoint
B) SageMaker automatically deletes the Endpoint Configuration and Model when the Endpoint is deleted
C) The student will not incur additional compute charges because Endpoint Configurations and Models are metadata-only resources
D) The Endpoint Configuration will automatically spin up a new Endpoint after 24 hours

**Correct Answer: C**

**Rationale:** Endpoint Configurations and Models are metadata resources that reference container images and S3 paths; they do not provision compute instances. Only the Endpoint itself (which runs ML instances) incurs hourly compute charges. However, best practice is still to delete all three to keep the environment clean. SageMaker does not auto-delete related resources (B) or auto-recreate endpoints (D). Billing does not apply to metadata resources (A).

---

### Question 23 -- MCQ | Module 3 | Intermediate | Scenario-Based

**A student finishes a SageMaker lab and wants to ensure no ongoing charges. Which set of cleanup actions is most thorough?**

A) Delete the Endpoint, Endpoint Configuration, and Model; stop any running Studio apps
B) Delete the SageMaker notebook instance only
C) Delete the S3 bucket containing training data
D) Log out of the AWS Console

**Correct Answer: A**

**Rationale:** The primary billing risk comes from running endpoints (compute charges) and running Studio apps (kernel instance charges). Deleting the three deployment resources and stopping Studio apps eliminates ongoing compute costs. Deleting only the notebook (B) does not address endpoints. Deleting the S3 bucket (C) removes data but does not stop compute billing. Logging out (D) does not stop running resources.

---

### Question 24 -- MCQ | Module 4 | Beginner | Conceptual

**Which of the following best describes MLOps?**

A) A set of manual procedures for deploying machine learning models
B) A discipline that combines ML, DevOps, and data engineering to automate, version, and monitor ML systems throughout their lifecycle
C) A specific AWS service for running machine learning pipelines
D) A machine learning algorithm optimized for operations research problems

**Correct Answer: B**

**Rationale:** MLOps extends DevOps principles (automation, CI/CD, monitoring) to the ML lifecycle, adding concerns like data versioning, model governance, and continuous training. It is a discipline and set of practices (B), not a manual process (A), a single service (C), or an algorithm (D). SageMaker Pipelines is a tool that supports MLOps practices, but MLOps itself is broader.

---

### Question 25 -- MCQ | Module 4 | Advanced | Scenario-Based

**A team has reached MLOps Level 0: they train models manually in notebooks and deploy them through the console. They want to move to Level 1. Which improvement most directly characterizes the transition from Level 0 to Level 1?**

A) Adding unit tests for data validation
B) Migrating all notebooks from SageMaker Studio to VS Code
C) Implementing A/B testing for production endpoints
D) Automating the training pipeline so that models can be retrained on new data without manual notebook execution

**Correct Answer: D**

**Rationale:** The key transition from Level 0 to Level 1 is automating the training pipeline (Continuous Training). At Level 0, everything is manual. Level 1 introduces pipeline automation so retraining can be triggered by new data or schedules without human intervention. Unit tests (A) and A/B testing (C) are valuable practices often associated with Level 2. IDE migration (B) is an environment choice, not an MLOps maturity indicator.

---

### Question 26 -- MCQ | Module 4 | Intermediate | Scenario-Based

**In a SageMaker Pipeline, a data scientist defines a `ProcessingStep` for preprocessing, a `TrainingStep` for model training, and a `RegisterModel` step for registry registration. She connects them in sequence. What structure does this create?**

A) A Directed Acyclic Graph (DAG) where each step depends on the output of the previous step
B) A circular dependency graph
C) A parallel execution graph where all steps run simultaneously
D) A recursive pipeline that re-executes until convergence

**Correct Answer: A**

**Rationale:** SageMaker Pipelines represent workflows as DAGs where steps have explicit input/output dependencies. When steps are connected sequentially (preprocessing output feeds training, training output feeds registration), the result is a linear DAG. DAGs cannot be circular (B) by definition. Steps do not run in parallel (C) unless they have no dependency on each other. Pipelines do not recursively re-execute (D).

---

### Question 27 -- T/F | Module 4 | Intermediate

**True or False: SageMaker Model Monitor can detect data drift by comparing incoming inference data against a baseline statistical profile captured from the training dataset.**

**Correct Answer: True**

**Rationale:** Model Monitor captures inference requests (via data capture on the endpoint), computes statistical properties of the incoming features, and compares them to a baseline generated from the training data. When the distribution of incoming data diverges beyond configured thresholds, Model Monitor raises alerts through CloudWatch. This is the core mechanism for detecting data drift in production.

---

### Question 28 -- Matching | Module 4 | Advanced

**Match each MLOps tool or service to the problem it solves.**

| # | Tool / Service | | Problem It Solves |
|---|----------------|---|-------------------|
| 1 | SageMaker Pipelines | | A) Detecting data and concept drift in production inference traffic |
| 2 | SageMaker Model Registry | | B) Triggering automated actions (e.g., retraining) in response to AWS events |
| 3 | SageMaker Model Monitor | | C) Orchestrating multi-step ML workflows as reproducible, versioned DAGs |
| 4 | Amazon EventBridge | | D) Cataloging, versioning, and governing model artifacts with approval workflows |

**Correct Answers: 1-C, 2-D, 3-A, 4-B**

**Rationale:** Pipelines orchestrate ML workflows as DAGs (C). Model Registry provides centralized model governance and versioning (D). Model Monitor detects drift by comparing inference data to training baselines (A). EventBridge is an event bus that can trigger Lambda functions, CodePipeline executions, or other targets in response to events like model approval status changes (B).

---

### Question 29 -- T/F | Module 1 | Intermediate

**True or False: When creating a SageMaker Studio Domain using Quick Setup, you must manually create a VPC and configure all subnet and security group settings before the domain can be provisioned.**

**Correct Answer: False**

**Rationale:** Quick Setup uses the default VPC, subnets, and security groups in your account, and automatically creates a default execution role. It is designed to get a domain running with minimal configuration. Manual VPC configuration is only required for custom or enterprise setups using the Standard Setup option.

---

### Question 30 -- MCQ | Module 2 | Intermediate | Scenario-Based

**A student configures a SageMaker Estimator with `instance_type="ml.m5.xlarge"` and `instance_count=1`. The training job starts but fails with an `Access Denied` error when downloading data from S3. The data exists in the correct bucket. What is the most likely cause?**

A) The `ml.m5.xlarge` instance type does not support S3 access
B) The training script has a syntax error
C) The SageMaker Execution Role attached to the Estimator lacks `s3:GetObject` and `s3:ListBucket` permissions for the training data bucket
D) The S3 bucket is in a different AWS region than the training job

**Correct Answer: C**

**Rationale:** An `Access Denied` error during data download indicates the IAM Execution Role does not have sufficient S3 permissions for the specified bucket. All SageMaker instance types support S3 access (A). Script syntax errors (B) would produce a different error after data download succeeds. Cross-region S3 access (D) is possible (though slower) and would not produce an `Access Denied` error; it would more likely succeed or produce a different error.

---
