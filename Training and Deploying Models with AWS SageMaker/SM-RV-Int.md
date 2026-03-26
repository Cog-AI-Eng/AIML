# AWS SageMaker Review Interview - Technical Interview Guide

**Activity ID:** SM-RV-Int
**Display Name:** AWS SageMaker Review Interview
**Duration:** 30 minutes
**Type:** Technical Interview

---

## Interview Overview

This interview assesses a candidate's understanding of AWS SageMaker across four modules: Foundations, Training, Deployment, and MLOps. The guide contains 14 prompts spanning Beginner, Intermediate, and Advanced difficulty levels. At least 70% of prompts are scenario-based, drawing from the console-first labs completed throughout the skill unit.

**Time allocation suggestion:**
- Module 1 (Foundations): ~7 minutes (Prompts 1-4)
- Module 2 (Training): ~8 minutes (Prompts 5-8)
- Module 3 (Deployment): ~8 minutes (Prompts 9-11)
- Module 4 (MLOps): ~5 minutes (Prompts 12-14)
- Buffer / follow-ups: ~2 minutes

---

## Prompt 1 -- SageMaker Ecosystem & ML Lifecycle (Beginner, Conceptual)

**Module:** 1 -- Foundations

> Walk me through the five stages of the SageMaker ML Lifecycle. For each stage, name one SageMaker service or feature that supports it, and briefly explain what happens at that stage.

<details>
<summary><b>Click to expand Interviewer Guide</b></summary>

#### Target Keywords
`Prepare`, `Build`, `Train & Tune`, `Deploy`, `Monitor`, `Data Wrangler`, `JumpStart`, `Experiments`, `Endpoints`, `Model Monitor`

#### Expected Good Answer
* Lists all five stages in order: Prepare, Build, Train & Tune, Deploy, Monitor.
* Maps at least one service per stage (e.g., Data Wrangler for Prepare, JumpStart or Feature Store for Build, Training Jobs / Experiments for Train & Tune, Endpoints for Deploy, Model Monitor for Monitor).
* Explains each stage in 1-2 sentences (e.g., "Prepare is about ingesting and labeling data so it is ready for feature engineering").
* Strong answers reference the Foundations reading and how the AIML skill's "ML Lifecycle & Reproducibility" topic maps to SageMaker's five-stage view.

#### Red Flags
* Cannot recall the stage order (e.g., puts Deploy before Train)
* Conflates Build with Train (they are separate stages in SageMaker's framework)
* Cannot name any SageMaker services -- just says "SageMaker does it all"
* Describes the lifecycle generically without tying stages to AWS services

#### Follow-Up Prompt
"If a team only has 2 weeks and a labeled dataset, which stages could they shortcut by using SageMaker JumpStart, and which stages are still mandatory?"

</details>

---

## Prompt 2 -- Studio Domains & User Personas (Intermediate, Scenario-Based)

**Module:** 1 -- Foundations

> In the Foundations lab, you created a SageMaker Studio Domain using Quick Setup. Imagine your organization needs to onboard three teams: a data engineering team that prefers classic JupyterLab, a business analytics team with no coding experience, and an ML platform team that needs the latest IDE features. How would you configure Studio to support all three, and what role does the execution role play in this setup?

<details>
<summary><b>Click to expand Interviewer Guide</b></summary>

#### Target Keywords
`Studio Domain`, `User Profile`, `Studio Classic`, `Canvas`, `Studio`, `Execution Role`, `Quick Setup`, `EFS`, `personas`

#### Expected Good Answer
* Explains that a single Domain can host multiple User Profiles, each configured for a different persona.
* Maps data engineers to Studio Classic (JupyterLab), business analysts to Canvas (no-code), and the ML platform team to Studio (full IDE).
* Explains that the execution role is an IAM Role that SageMaker assumes to access resources (S3, CloudWatch) on behalf of each user profile, and that Quick Setup auto-creates a default role.
* Mentions that EFS provides shared persistent storage across user profiles within the domain.
* Strong answers reference the lab experience of creating `fraudshield-domain` and the `analyst-profile` user profile.

#### Red Flags
* Does not know the difference between Studio, Studio Classic, and Canvas
* Thinks each persona needs a separate AWS account
* Confuses execution roles with IAM Users or console login credentials
* Cannot explain what Quick Setup automates vs. what Standard Setup requires

#### Follow-Up Prompt
"In your lab, you used Quick Setup with the default VPC. What would change if your security team required a private VPC with no internet access?"

</details>

---

## Prompt 3 -- IAM Least-Privilege for SageMaker (Advanced, Scenario-Based)

**Module:** 1 -- Foundations

> In the Foundations lab, you created a custom IAM policy that restricted a SageMaker execution role to specific S3 buckets. Explain the principle of least-privilege as it applies to SageMaker execution roles. Walk me through the specific permissions you would grant and why, using the FraudShield scenario as an example.

<details>
<summary><b>Click to expand Interviewer Guide</b></summary>

#### Target Keywords
`least-privilege`, `IAM Role`, `Trust Policy`, `sagemaker.amazonaws.com`, `sts:AssumeRole`, `s3:GetObject`, `s3:ListBucket`, `s3:PutObject`, `CloudWatch Logs`, `ARN`, `resource scoping`

#### Expected Good Answer
* Defines least-privilege: grant only the permissions needed for the task, nothing more.
* Describes the trust policy allowing `sagemaker.amazonaws.com` to assume the role via `sts:AssumeRole`.
* Lists specific S3 permissions scoped to bucket ARNs: `s3:GetObject` on data bucket objects, `s3:PutObject` on model output bucket objects, `s3:ListBucket` on both bucket ARNs.
* Explains why `s3:ListBucket` targets the bucket ARN (not object ARN) while `s3:GetObject`/`s3:PutObject` target object ARNs with `/*` suffix.
* Includes CloudWatch Logs permissions (`logs:CreateLogGroup`, `logs:CreateLogStream`, `logs:PutLogEvents`) for training job logging.
* Strong answers contrast this with the overly broad "Any S3 bucket" default role created during Quick Setup.

#### Red Flags
* Suggests `s3:*` on `*` as an acceptable policy
* Cannot distinguish between bucket-level and object-level ARN patterns
* Does not mention CloudWatch Logs permissions (SageMaker needs these for debugging)
* Confuses IAM Users with IAM Roles

#### Follow-Up Prompt
"What would happen if you forgot `s3:ListBucket` in your policy? How would the failure manifest during a training job?"

</details>

---

## Prompt 4 -- JumpStart Deployment & Three-Object Pattern (Intermediate, Scenario-Based)

**Module:** 1 -- Foundations

> In your Foundations lab, you deployed a model from SageMaker JumpStart. Describe what happened behind the scenes when you clicked 'Deploy.' What resources were created, and why did the cleanup guide require you to delete them in a specific order?

<details>
<summary><b>Click to expand Interviewer Guide</b></summary>

#### Target Keywords
`JumpStart`, `Model`, `Endpoint Configuration`, `Endpoint`, `three-object pattern`, `ml.m5.xlarge`, `cleanup`, `billing`, `container image`, `model artifact`

#### Expected Good Answer
* Identifies the three-object deployment pattern: Model (references container image + S3 artifact), Endpoint Configuration (instance type, variant, count), and Endpoint (live compute resource).
* Explains the deletion order: Endpoint first (stops billing), then Endpoint Configuration, then Model. Deleting the Endpoint first is critical because it is the only resource that incurs compute charges.
* Mentions that JumpStart abstracts the creation of all three objects behind a single "Deploy" button.
* Strong answers note the specific instance type used (e.g., `ml.m5.xlarge`) and the cost implications of leaving endpoints running.

#### Red Flags
* Cannot name the three resources created during deployment
* Does not understand that the Endpoint is what costs money (claims Endpoint Config also bills)
* Thinks cleanup is optional or that SageMaker auto-deletes resources
* Cannot explain what the Model resource actually references (thinks it stores the weights directly)

#### Follow-Up Prompt
"If you only delete the Endpoint but leave the Endpoint Configuration and Model, do you still incur charges? Why or why not?"

</details>

---

## Prompt 5 -- BYOM vs. Script Mode (Beginner, Conceptual)

**Module:** 2 -- Training

> Explain the difference between Bring-Your-Own-Model (BYOM) and Bring-Your-Own-Script (Script Mode) in SageMaker. When would you choose one over the other?

<details>
<summary><b>Click to expand Interviewer Guide</b></summary>

#### Target Keywords
`BYOM`, `Script Mode`, `Docker`, `ECR`, `pre-built container`, `framework container`, `scikit-learn`, `PyTorch`, `custom dependencies`, `entry_point`

#### Expected Good Answer
* Script Mode: you write a Python training script; SageMaker provides a pre-built framework container (e.g., scikit-learn, PyTorch) with the ML framework already installed. You specify `entry_point` and SageMaker injects your script into the container.
* BYOM: you build a complete Docker image with all dependencies, the model code, and the serving logic, then push it to Amazon ECR. SageMaker runs your container as-is.
* Guidance: use Script Mode when your code works with a supported framework and standard dependencies. Use BYOM when you need custom system libraries, proprietary frameworks, or specialized serving logic that the pre-built containers do not support.
* References the AIML skill's "Algorithm Selection Framework" concept -- the choice of approach depends on whether your algorithm fits a supported framework.

#### Red Flags
* Reverses the definitions (thinks Script Mode requires Docker)
* Cannot name any pre-built framework containers SageMaker provides
* Says "always use BYOM because it gives more control" without acknowledging the additional complexity
* Does not understand that Script Mode still runs inside a container

#### Follow-Up Prompt
"In the Training lab, you used Script Mode with scikit-learn. What would need to change if your training script required a C library that is not in the scikit-learn container?"

</details>

---

## Prompt 6 -- Script Mode Structure (Intermediate, Scenario-Based)

**Module:** 2 -- Training

> In the Training lab, you wrote a Script Mode training script for a RandomForest fraud classifier. Walk me through the required structure of that script: what must be at the top, what arguments does it parse, and where does it read data from and save the model to? Why are these conventions required?

<details>
<summary><b>Click to expand Interviewer Guide</b></summary>

#### Target Keywords
`if __name__ == "__main__"`, `argparse`, `SM_CHANNEL_TRAIN`, `SM_MODEL_DIR`, `/opt/ml/input/data/`, `/opt/ml/model/`, `hyperparameters`, `environment variables`, `model.tar.gz`

#### Expected Good Answer
* The script needs: imports, a `if __name__ == "__main__":` guard, an `argparse.ArgumentParser` to receive hyperparameters as CLI arguments, reading training data from `os.environ["SM_CHANNEL_TRAIN"]` (maps to `/opt/ml/input/data/train/`), and saving the trained model to `os.environ["SM_MODEL_DIR"]` (maps to `/opt/ml/model/`).
* Explains why: SageMaker invokes the script as `python train.py --arg1 val1 ...`, so the main guard is needed. Hyperparameters are passed as CLI args by the Estimator. Data paths are standardized so SageMaker can download from S3 before execution and upload artifacts after execution.
* The model must be saved to `SM_MODEL_DIR` because SageMaker packages that directory into `model.tar.gz` and uploads it to S3.
* Strong answers connect this to the lab's `train.py` script and the specific hyperparameters used (e.g., `--n-estimators`, `--random-state`).

#### Red Flags
* Does not know about `SM_CHANNEL_TRAIN` or `SM_MODEL_DIR` environment variables
* Thinks data is read directly from S3 inside the script (via `boto3`)
* Cannot explain why saving to the wrong directory produces an empty `model.tar.gz`
* Omits the main guard or `argparse`

#### Follow-Up Prompt
"What happens if your script saves the model to the current working directory instead of `SM_MODEL_DIR`? Will the training job fail or succeed, and what artifact will SageMaker upload?"

</details>

---

## Prompt 7 -- Training Job Anatomy & Debugging (Advanced, Scenario-Based)

**Module:** 2 -- Training

> Describe the full lifecycle of a SageMaker Training Job from the moment you click 'Create Training Job' in the console to the moment the job completes. Then, imagine the job fails with a `ModuleNotFoundError` for a package your script imports. Walk me through how you would diagnose and fix this.

<details>
<summary><b>Click to expand Interviewer Guide</b></summary>

#### Target Keywords
`Provisioning`, `container pull`, `data download`, `script execution`, `artifact upload`, `teardown`, `CloudWatch Logs`, `/aws/sagemaker/TrainingJobs`, `requirements.txt`, `ModuleNotFoundError`, `pre-built container`

#### Expected Good Answer
* Lifecycle: Provisioning (instance allocated) -> Container Pull (Docker image downloaded from ECR) -> Data Download (training data copied from S3 to `/opt/ml/input/data/`) -> Script Execution (your `entry_point` runs) -> Artifact Upload (contents of `/opt/ml/model/` packaged as `model.tar.gz` and uploaded to S3) -> Teardown (instance terminated).
* Debugging `ModuleNotFoundError`: Navigate to CloudWatch Logs under `/aws/sagemaker/TrainingJobs` log group, find the log stream for the failed job, and look for the stack trace. The error means a Python package is missing from the pre-built container.
* Fix: Either add a `requirements.txt` file in the same source directory (SageMaker installs it automatically before running the script) or switch to BYOM with a custom container if the dependency is complex.
* Strong answers reference the Training lab's CloudWatch debugging step and the console's training job detail page.

#### Red Flags
* Cannot describe the lifecycle stages in order
* Does not know that SageMaker terminates the instance after training (thinks it stays running)
* Suggests retraining with a different algorithm to fix a missing module error
* Does not mention CloudWatch Logs as the debugging tool

#### Follow-Up Prompt
"If the training job succeeds but the model performance is poor, where in this lifecycle would you add instrumentation to diagnose the issue? Think about what SageMaker Experiments could capture."

</details>

---

## Prompt 8 -- Model Artifacts & S3 Storage (Intermediate, Scenario-Based)

**Module:** 2 -- Training

> After your training job completed successfully in the Training lab, you navigated to S3 to find the model artifact. Explain the S3 path convention SageMaker uses for storing model artifacts. Why does SageMaker use this convention, and how does it enable implicit versioning?

<details>
<summary><b>Click to expand Interviewer Guide</b></summary>

#### Target Keywords
`model.tar.gz`, `s3://<bucket>/<prefix>/<job-name>/output/model.tar.gz`, `implicit versioning`, `training job name`, `unique job name`, `output_path`, `S3`

#### Expected Good Answer
* The artifact lands at: `s3://<bucket>/<output-path-prefix>/<training-job-name>/output/model.tar.gz`.
* SageMaker generates unique training job names (or uses the name you provide), which means each training run creates a distinct S3 path. This provides implicit versioning: you can always trace back from an artifact to the exact training job that produced it.
* The `model.tar.gz` is a gzipped tar archive of everything the script saved to `SM_MODEL_DIR`.
* Strong answers connect this to the Model Registry: when registering a model, you point to this S3 path as the artifact location, making it the bridge between training and deployment.

#### Red Flags
* Cannot recall the S3 path pattern
* Thinks SageMaker stores model artifacts in DynamoDB or a database
* Does not understand implicit versioning (thinks you must manually rename files)
* Confuses the model artifact with the training data

#### Follow-Up Prompt
"If two team members run training jobs with the same output path prefix, will their artifacts overwrite each other? Why or why not?"

</details>

---

## Prompt 9 -- Model Registry & Approval Workflows (Intermediate, Scenario-Based)

**Module:** 3 -- Deployment

> In the Deployment lab, you registered a trained model in the SageMaker Model Registry and then worked through an approval workflow. Explain the relationship between Model Package Groups and Model Packages. Then describe what happens -- and does not happen -- when you change a model version's approval status from `PendingManualApproval` to `Approved`.

<details>
<summary><b>Click to expand Interviewer Guide</b></summary>

#### Target Keywords
`Model Package Group`, `Model Package`, `versioning`, `PendingManualApproval`, `Approved`, `Rejected`, `EventBridge`, `automation`, `governance`, `metadata`

#### Expected Good Answer
* A Model Package Group is a logical container (like a Git repository) that holds multiple Model Packages (like commits/versions). Each Model Package contains a reference to the S3 model artifact, the container image, and metadata.
* When approval status changes to `Approved`: nothing happens automatically within SageMaker itself. The approval status is metadata. However, teams can build automation around it: EventBridge can detect the state change event and trigger a deployment pipeline, or a SageMaker Pipeline `ConditionStep` can check approval status before proceeding.
* The key insight is that the Registry provides governance (who approved what, when) but does not enforce deployment -- that is the responsibility of the automation layer (Pipelines, EventBridge, CodePipeline).
* Strong answers reference the lab's manual approval step in the console and how this mirrors real-world ML governance.

#### Red Flags
* Thinks changing status to `Approved` automatically deploys the model
* Cannot explain the Group/Package hierarchy
* Does not mention that approval status is a metadata field, not an action trigger
* Confuses the Model Registry with the Model resource in the three-object pattern

#### Follow-Up Prompt
"How would you set up automated deployment that triggers only when a model version is approved? Name the AWS services involved."

</details>

---

## Prompt 10 -- Real-time Endpoint Deployment (Advanced, Scenario-Based)

**Module:** 3 -- Deployment

> Walk me through the three-object deployment pattern in SageMaker. In the Deployment lab, you deployed a model to a real-time endpoint. Explain each resource, what it references, and the cost implications. Then describe the alternative inference modes SageMaker offers and when you would choose each.

<details>
<summary><b>Click to expand Interviewer Guide</b></summary>

#### Target Keywords
`Model`, `Endpoint Configuration`, `Endpoint`, `three-object pattern`, `real-time`, `batch transform`, `serverless inference`, `asynchronous inference`, `multi-model endpoint`, `ml.m5.xlarge`, `cost`

#### Expected Good Answer
* Model: references the Docker container image (from ECR) and the S3 path to `model.tar.gz`. It is metadata, not compute.
* Endpoint Configuration: specifies instance type, count, model reference, and production variant. Also metadata.
* Endpoint: the live resource that provisions ML instances, loads the model, and serves predictions over HTTPS. This is what incurs hourly compute charges.
* Alternative modes: Batch Transform (large offline datasets, no persistent endpoint), Serverless Inference (auto-scales to zero, pay per invocation, good for infrequent traffic), Asynchronous Inference (large payloads, minutes-long processing, queued), Multi-Model Endpoints (host multiple models on one instance to save cost).
* Strong answers reference the specific instance type used in the lab (`ml.m5.xlarge`) and the mandatory cleanup steps.

#### Red Flags
* Cannot name all three objects in the deployment pattern
* Thinks Endpoint Configuration also incurs charges
* Does not know about any alternative inference modes
* Cannot explain when serverless or batch would be preferred over real-time

#### Follow-Up Prompt
"Your endpoint receives 10 requests per day. You are currently using a real-time endpoint with `ml.m5.xlarge`. What alternative would you recommend and why?"

</details>

---

## Prompt 11 -- Endpoint Invocation & Debugging (Intermediate, Scenario-Based)

**Module:** 3 -- Deployment

> In the Deployment lab, you invoked your endpoint using `boto3`. Walk me through the invocation code: which client do you use, what parameters does `invoke_endpoint` require, and how do you read the response? Then tell me: if your endpoint returns a 500 error, what is your debugging approach?

<details>
<summary><b>Click to expand Interviewer Guide</b></summary>

#### Target Keywords
`sagemaker-runtime`, `invoke_endpoint`, `EndpointName`, `ContentType`, `Body`, `text/csv`, `application/json`, `response["Body"].read()`, `CloudWatch Logs`, `InternalServerError`, `deserialization`

#### Expected Good Answer
* Uses `boto3.client("sagemaker-runtime")` (not the regular `sagemaker` client).
* `invoke_endpoint` requires `EndpointName` (the endpoint name), `ContentType` (e.g., `text/csv` or `application/json`), and `Body` (the payload as a string or bytes).
* Response is read via `response["Body"].read().decode("utf-8")` -- it is a streaming body.
* For 500 errors: check CloudWatch Logs under `/aws/sagemaker/Endpoints/<endpoint-name>` for container logs and stack traces. Common causes include `ContentType` mismatch (sending CSV when the model expects JSON), wrong number of features, or model deserialization failures.
* Strong answers reference the lab's specific CSV payload format and the prediction output parsing.

#### Red Flags
* Uses `boto3.client("sagemaker")` instead of `sagemaker-runtime`
* Does not know that the response body must be `.read()` (treats it as a string directly)
* Cannot name `ContentType` as a required parameter
* Debugging approach is "redeploy" without checking logs first

#### Follow-Up Prompt
"What changes in your invocation code if the endpoint expects JSON instead of CSV? Show me how the `Body` and `ContentType` would change."

</details>

---

## Prompt 12 -- MLOps Principles & Maturity Levels (Beginner, Conceptual)

**Module:** 4 -- MLOps

> Explain what MLOps is and why it is different from traditional DevOps. Then describe the three MLOps maturity levels (0, 1, 2) and what distinguishes each.

<details>
<summary><b>Click to expand Interviewer Guide</b></summary>

#### Target Keywords
`MLOps`, `DevOps`, `CI/CD`, `Continuous Training`, `data versioning`, `model governance`, `Level 0`, `Level 1`, `Level 2`, `automation`, `monitoring`

#### Expected Good Answer
* MLOps extends DevOps to ML: it adds concerns like data versioning, model governance, experiment tracking, and continuous training (not just continuous integration/deployment of code).
* Two sources of change in ML systems: code changes AND data changes (unlike traditional software which only has code changes).
* Level 0: Manual -- data scientists train in notebooks, deploy manually, no automation.
* Level 1: Automated Training Pipeline -- training can be triggered by new data or schedules, pipeline is codified (e.g., SageMaker Pipelines), but deployment may still be manual or semi-automated.
* Level 2: Full CI/CD for ML -- code changes trigger automated testing, training, validation, and deployment. Includes monitoring, automatic retraining, and rollback capabilities.
* Strong answers reference the AIML skill's reproducibility concepts and how MLOps provides the infrastructure to enforce them at scale.

#### Red Flags
* Defines MLOps as "just DevOps for ML" without explaining the additional dimensions (data, models, experiments)
* Cannot explain any maturity level beyond "manual vs. automated"
* Does not mention data as a source of change
* Thinks Level 2 means "using more AWS services"

#### Follow-Up Prompt
"Your team is at Level 0. You can only implement one improvement this quarter. Would you prioritize automated training pipelines or automated monitoring? Justify your choice."

</details>

---

## Prompt 13 -- SageMaker Pipelines & DAGs (Advanced, Scenario-Based)

**Module:** 4 -- MLOps

> In the MLOps lab, you built a SageMaker Pipeline with preprocessing, training, and model registration steps. Explain what a DAG is in this context, how SageMaker Pipelines represents step dependencies, and how you would add a conditional quality gate that only registers the model if the F1 score exceeds 0.80.

<details>
<summary><b>Click to expand Interviewer Guide</b></summary>

#### Target Keywords
`DAG`, `Directed Acyclic Graph`, `ProcessingStep`, `TrainingStep`, `RegisterModel`, `ConditionStep`, `JsonGet`, `pipeline.upsert()`, `pipeline.start()`, `step dependencies`, `Pipeline parameters`

#### Expected Good Answer
* A DAG (Directed Acyclic Graph) is a graph where edges have direction and there are no cycles. In SageMaker Pipelines, each step is a node, and edges represent data/output dependencies between steps.
* Dependencies are implicit: when one step references another step's output (e.g., `TrainingStep` uses `ProcessingStep`'s output as its input channel), SageMaker infers the edge.
* For the quality gate: add an evaluation `ProcessingStep` after training that computes F1 and writes it to a JSON file. Then add a `ConditionStep` that uses `JsonGet` to extract the F1 value and compare it to 0.80. If the condition passes, the pipeline proceeds to `RegisterModel`; otherwise, it stops or routes to a "reject" branch.
* Strong answers reference the lab's `evaluate.py` script and the specific `ConditionStep` implementation.

#### Red Flags
* Cannot define DAG or explain what "acyclic" means (no cycles)
* Thinks steps run in parallel by default regardless of dependencies
* Does not know about `ConditionStep` or `JsonGet` for conditional logic
* Suggests hardcoding quality thresholds in the training script instead of using pipeline conditions

#### Follow-Up Prompt
"If your pipeline has a preprocessing step and two independent training steps (one for RandomForest, one for XGBoost), how would the DAG differ? Would the training steps run sequentially or in parallel?"

</details>

---

## Prompt 14 -- Model Monitor & Production Governance (Intermediate, Scenario-Based)

**Module:** 4 -- MLOps

> Your fraud detection model has been deployed to production for 3 months. The business team reports that false positive rates have increased significantly. Using your knowledge of SageMaker's MLOps tools, explain what might be happening and how you would set up proactive monitoring to catch this earlier next time.

<details>
<summary><b>Click to expand Interviewer Guide</b></summary>

#### Target Keywords
`data drift`, `concept drift`, `Model Monitor`, `baseline`, `data capture`, `monitoring schedule`, `CloudWatch`, `EventBridge`, `retraining trigger`, `statistical profile`

#### Expected Good Answer
* Diagnosis: the increase in false positives likely indicates data drift (the distribution of incoming features has shifted from what the model was trained on) or concept drift (the relationship between features and the target has changed -- e.g., fraud patterns evolved).
* Prevention: set up SageMaker Model Monitor with three components:
  1. **Baseline**: capture the statistical profile of the training data (feature distributions, missing value rates).
  2. **Data Capture**: enable on the endpoint to log incoming inference requests and responses.
  3. **Monitoring Schedule**: run periodic comparison jobs that compare captured data against the baseline and raise CloudWatch alarms when drift exceeds thresholds.
* Automation: use EventBridge to react to drift alerts -- trigger retraining via a SageMaker Pipeline or notify the team via SNS.
* Strong answers distinguish between data drift and concept drift and reference the AIML skill's discussion of evaluation metrics deteriorating in production.

#### Red Flags
* Does not know what data drift or concept drift is
* Suggests only retraining on a fixed schedule without monitoring (reactive, not proactive)
* Cannot name any component of Model Monitor (baseline, data capture, monitoring schedule)
* Thinks Model Monitor automatically retrains the model (it only detects drift and raises alerts)

#### Follow-Up Prompt
"Model Monitor detects feature drift, but your model's accuracy is still acceptable. Should you retrain immediately or wait? What factors would inform your decision?"

</details>

---
