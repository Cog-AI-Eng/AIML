# AdvSageMaker-Tracking Lecture - Instructor Guide

**Total Duration:** 180 Minutes (3 Stages)
**Consolidated Activities:** Experiments & Trials, Lineage Tracking Entities, Feature Store Lineage Integration, Reproducibility Patterns, Cross-account Sharing

| Block | Content | Minutes |
|-------|---------|---------|
| Stage 1 | Experiment Tracking and Run Comparison | 45 |
| Break 1 | Stretch / Questions | 10 |
| Stage 2 | Lineage Tracking and Feature Store Integration | 45 |
| Break 2 | Stretch / Questions | 10 |
| Stage 3 | Reproducibility Report and Cross-account Patterns | 45 |
| Buffer | Open Q&A, Wrap-Up | 25 |

---

## Lecture Overview

**Unified Scenario -- FraudShield Risk Analytics (Advanced)**

FraudShield's ML team has trained dozens of models across the previous two modules -- XGBoost classifiers, K-Means segmentation models, RCF anomaly detectors, and HPO-tuned variants. But a critical question has emerged during a compliance review: "Which model is currently in production, what data was it trained on, what hyperparameters were used, and can you reproduce the exact same result?" The team cannot answer this question with certainty because training runs were launched ad hoc, results were compared manually, and the connection between data, models, and endpoints was never formally tracked.

This module introduces the observability layer that production ML systems require. Associates will use SageMaker Experiments to organize, track, and compare training runs. They will navigate the Lineage Tracking graph to trace the complete provenance of a model -- from raw data in S3 through Feature Store, through training, to the deployed endpoint. The module culminates with reproducibility patterns and cross-account architectures, reflecting FraudShield's transition from a single-account prototype to a multi-account production deployment.

The scenario reflects a common inflection point in enterprise ML: the transition from "we can build models" to "we can govern, audit, and reproduce models." Experiment tracking and lineage are not optional features -- they are requirements for any regulated industry. FraudShield's financial services context makes this especially relevant, as regulators may ask for model provenance documentation at any time.

---

## Pre-Lecture Setup

### Instructor Checklist
- Verify that at least 3-5 completed training jobs exist in the account from the previous module (XGBoost, K-Means, RCF, or HPO trials). These will be associated with Experiments during Stage 1.
- If training jobs from the previous module were cleaned up, pre-run three quick XGBoost training jobs with different hyperparameter settings (vary `max_depth`: 3, 5, 8 and `eta`: 0.1, 0.2, 0.3) so Associates have runs to compare
- Verify the Feature Store feature group (`fraudshield-customer-features`) from Module 1 still exists (or recreate it)
- Pre-create one SageMaker Experiment named `fraudshield-baseline-experiment` with one Run associated with a completed training job. This demonstrates what the end state looks like before Associates create their own.
- Have the SageMaker Studio Experiments UI loaded and verify it displays correctly
- Confirm that the account has Model Registry access and at least one registered model package group (or create `fraudshield-models` as a model package group)
- Prepare a slide or diagram showing FraudShield's target multi-account architecture (data account, training account, deployment account) for Stage 3
- Verify IAM cross-account role policies are available for discussion (have JSON policy documents ready to display)
- Ensure ml.t3.medium notebook instances are available for any SDK demonstrations

### Student Prerequisites
- Completed Modules 1 and 2 (AdvSageMaker-Data and AdvSageMaker-Algorithms)
- Familiarity with SageMaker training jobs, model artifacts, and the Feature Store from previous modules
- Understanding of S3 bucket policies and IAM roles at a conceptual level
- Access to the SageMaker Studio domain with the execution role configured in Module 1
- At least 3 completed training jobs in the account (from Module 2 or pre-run by the instructor)

---

## Stage 1: Experiment Tracking and Run Comparison
**Duration:** 45 minutes
**Exit Criteria Addressed:**
- Create a SageMaker Experiment and associate training runs
- Use the console comparison view to evaluate metrics across runs
- Log custom metrics to an Experiment Run
- Explain the relationship between Experiments, Runs, and training jobs

### Instructor Opening (5 minutes)

> "In the last module you launched at least half a dozen training jobs -- XGBoost with different depths, K-Means with different cluster counts, an HPO job with 20 trials. Now imagine FraudShield has been operating for six months. There are hundreds of training jobs across multiple team members. Someone asks: 'Which model performed best on last quarter's data?' Without experiment tracking, you are scrolling through the training jobs list, opening each one, comparing metrics in separate tabs, and hoping you do not miss the best run. SageMaker Experiments gives you a structured way to organize, tag, compare, and query training runs. Let us set one up."

**Teaching Tip:** Before diving into the console, establish the terminology:
- **Experiment:** a named collection of related ML activities (e.g., "fraudshield-xgboost-v2")
- **Run:** a single training attempt within an Experiment (equivalent to one training job with specific hyperparameters)
- **Run Group (optional):** a logical grouping within an Experiment (e.g., grouping runs by data version or by algorithm family)

### STEP 1 -- Creating an Experiment from the Console (8 minutes)

**Console Navigation:**
1. Open SageMaker Studio
2. In the left sidebar, click **Home** > **Experiments**
3. Click **Create experiment**

4. Configure:
   - **Experiment name:** `fraudshield-xgb-optimization`
   - **Display name:** `FraudShield XGBoost Optimization`
   - **Description:** `Systematic comparison of XGBoost hyperparameter configurations for fraud detection on the FraudShield e-commerce transaction dataset.`

5. Click **Create**

**Instructor Note:** Point out that the Experiment is now visible in the Experiments list. It has zero Runs associated with it. Explain that Experiments are lightweight metadata containers -- creating one costs nothing and has no compute implications.

**Q&A Pause (1 minute):** "How would FraudShield organize its Experiments? One per algorithm? One per dataset version? One per quarter?" Discuss options and suggest a naming convention: `{project}-{algorithm}-{purpose}` (e.g., `fraudshield-xgb-optimization`, `fraudshield-kmeans-segmentation-v2`, `fraudshield-deepar-forecast-q4`).

### STEP 2 -- Associating Training Runs with the Experiment (12 minutes)

**Approach 1 -- Retroactively Associate Existing Training Jobs (Console Method):**

**Instructor Note:** SageMaker Experiments in Studio allows you to create Runs and associate them with existing training jobs. However, the cleanest approach is to associate training jobs at launch time. Show both methods.

1. In the Experiments view, click into `fraudshield-xgb-optimization`
2. Click **Create run**
3. Configure:
   - **Run name:** `xgb-depth3-eta01`
   - **Display name:** `XGBoost depth=3 eta=0.1`
   - Associate with the corresponding completed training job

**Approach 2 -- Associate at Training Time (SDK Method):**

Open a Studio notebook and demonstrate launching a training job that is automatically associated with the Experiment:

```python
import sagemaker
from sagemaker.xgboost import XGBoost
from sagemaker.experiments.run import Run

session = sagemaker.Session()
role = sagemaker.get_execution_role()

with Run(
    experiment_name="fraudshield-xgb-optimization",
    run_name="xgb-depth5-eta02",
    sagemaker_session=session,
) as run:
    run.log_parameter("max_depth", 5)
    run.log_parameter("eta", 0.2)
    run.log_parameter("scale_pos_weight", 10)
    run.log_parameter("num_round", 100)
    run.log_parameter("data_version", "2024-Q4")

    xgb = XGBoost(
        entry_point="train.py",
        role=role,
        instance_count=1,
        instance_type="ml.m5.xlarge",
        framework_version="1.5-1",
        hyperparameters={
            "max_depth": 5,
            "eta": 0.2,
            "objective": "binary:logistic",
            "num_round": 100,
            "eval_metric": "auc",
            "scale_pos_weight": 10,
        },
    )

    xgb.fit(
        inputs={
            "train": "s3://fraudshield-advanced-data/processed/train.csv",
            "validation": "s3://fraudshield-advanced-data/processed/test.csv",
        }
    )

    run.log_metric("validation:auc", 0.91)
```

**Pacing Guidance:** Walk through this code carefully. Highlight:
- The `Run` context manager automatically associates the training job with the Experiment
- `log_parameter` stores hyperparameters as searchable metadata (not just the ones SageMaker captures automatically)
- `log_metric` stores the final metric value (SageMaker also captures metrics from CloudWatch, but explicit logging ensures the exact metric you care about is recorded)
- `data_version` is a custom parameter -- you can log any key-value pair as metadata

**Instructor Note:** If time does not allow running three full training jobs, run one live and retroactively associate two existing jobs from Module 2.

### STEP 3 -- Comparing Runs in the Console (10 minutes)

**Console Navigation:**
1. In the Experiments view, click into `fraudshield-xgb-optimization`
2. The Runs list shows all associated runs with their parameters and metrics
3. Select three runs using the checkboxes
4. Click **Analyze** (or the chart icon) to open the comparison view

**Walk through the comparison features:**

a. **Table view:**
   - Shows parameters and metrics side by side for each selected run
   - Sort by any metric column to quickly identify the best run
   - Point out: "Run xgb-depth5-eta02 has the highest validation:auc at 0.91"

b. **Chart view:**
   - Create a scatter plot: X-axis = `max_depth`, Y-axis = `validation:auc`
   - Show how the metric varies with depth
   - Create a parallel coordinates chart if available: each axis is a parameter, the lines show how different parameter combinations map to the objective metric

**Teaching Tip:** This is the "aha moment" for many Associates. Instead of comparing training jobs one at a time, they see all results in a single view. Ask: "If FraudShield runs 50 experiments over a quarter, how much time does this comparison view save compared to manual inspection?"

c. **Filtering and searching:**
   - Show how to filter runs by parameter value (e.g., show only runs where `max_depth` > 4)
   - Show how to search runs by name pattern

**Q&A Pause (2 minutes):** "We logged `data_version` as a custom parameter. Why is this important for FraudShield?" Answer: when data changes (new fraud patterns emerge, new transaction categories are added), the team needs to compare models trained on different data versions. Without this metadata, it is impossible to determine whether a performance difference is due to hyperparameters or data.

### STEP 4 -- Custom Metric Logging (5 minutes)

**SDK Demonstration:**
Show how to log custom metrics during training (within a training script):

```python
from sagemaker.experiments.run import load_run

with load_run() as run:
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_data)
        val_auc = evaluate(model, val_data)
        precision = calculate_precision(model, val_data)
        recall = calculate_recall(model, val_data)
        f1 = 2 * precision * recall / (precision + recall)
        cost = calculate_fraud_cost(model, val_data)

        run.log_metric("train_loss", train_loss, step=epoch)
        run.log_metric("validation_auc", val_auc, step=epoch)
        run.log_metric("precision", precision, step=epoch)
        run.log_metric("recall", recall, step=epoch)
        run.log_metric("f1_score", f1, step=epoch)
        run.log_metric("fraud_cost_dollars", cost, step=epoch)
```

**Instructor Note:** Highlight two key points:
1. The `step` parameter creates a time series of metric values, enabling epoch-by-epoch comparison in the console charts
2. `fraud_cost_dollars` is a business metric, not a statistical one. FraudShield's leadership cares about dollar impact, not just AUC. Logging business metrics alongside statistical metrics is a best practice.

**Teaching Tip:** "load_run() works inside a training script running on a SageMaker training instance. It picks up the Experiment and Run context from the environment. You do not need to pass the Experiment name -- SageMaker injects it."

---

[PAUSE FOR BREAK -- 10 minutes]

---

## Stage 2: Lineage Tracking and Feature Store Integration
**Duration:** 45 minutes
**Exit Criteria Addressed:**
- Navigate the lineage graph for a trained model in the SageMaker console
- Identify lineage entity types: Artifact, Context, Action, Association
- Trace the provenance chain from an endpoint to raw training data
- Explain how Feature Store integrates with the lineage graph
- Query lineage programmatically using the SageMaker SDK

### Instructor Opening (3 minutes)

> "Experiment tracking tells you which model performed best. Lineage tracking tells you how that model came to exist. A regulator asks FraudShield: 'Show me the exact data, code, and configuration that produced the model scoring transactions right now.' With lineage tracking, you can answer that question by starting at the endpoint and tracing backwards through every transformation, training job, and data source. SageMaker builds this lineage graph automatically -- you just need to know how to read it."

### STEP 1 -- Understanding Lineage Entity Types (7 minutes)

**Conceptual Overview:**
Before navigating the console, establish the lineage vocabulary:

1. **Artifact:** a physical or logical object in the ML workflow
   - **Dataset Artifact:** S3 URI of a training dataset
   - **Model Artifact:** S3 URI of a model.tar.gz
   - **Image Artifact:** the container image used for training or inference
   - **Endpoint Artifact:** a deployed inference endpoint

2. **Action:** an operation that transforms inputs into outputs
   - **Training Job:** takes dataset artifacts as input, produces a model artifact as output
   - **Processing Job:** takes raw data as input, produces processed data as output
   - **Transform Job:** takes a model and data as input, produces predictions as output

3. **Context:** a grouping entity
   - **Experiment:** groups related actions
   - **Trial:** a specific attempt within an experiment
   - **Endpoint Context:** groups the model, endpoint config, and endpoint

4. **Association:** a directed edge connecting two lineage entities
   - Types: `ContributedTo`, `AssociatedWith`, `DerivedFrom`, `Produced`

**Teaching Tip:** Draw or describe the lineage graph for a typical FraudShield model:

```
[S3: raw data] --ContributedTo--> [Processing Job] --Produced--> [S3: processed data]
[S3: processed data] --ContributedTo--> [Training Job] --Produced--> [S3: model artifact]
[S3: model artifact] --ContributedTo--> [Model] --AssociatedWith--> [Endpoint]
[Container Image] --ContributedTo--> [Training Job]
```

Explain: "Every arrow is an Association. SageMaker creates most of these automatically when you use its managed services. The graph is queryable -- you can start at any node and traverse upstream or downstream."

### STEP 2 -- Navigating Lineage in the Console (12 minutes)

**Console Navigation:**
1. In SageMaker Studio, navigate to **Home** > **Models** (or **Endpoints** if a model is deployed)
2. Click on a model that was produced by a training job from Module 2

**Instructor Note:** If no models are registered, navigate to **Training** > **Training jobs**, select a completed job, and show the lineage from the training job page.

3. On the model details page, click the **Lineage** tab (or **Lineage** in the sidebar)
4. The lineage graph visualization appears, showing:
   - The model artifact at the center
   - Input arrows from training data (S3 artifacts)
   - Input arrow from the container image
   - The training job (Action node) connecting inputs to output
   - Output arrow to any endpoints using this model

**Walk through the graph interactively:**

a. **Click on the training data artifact:**
   - Shows the S3 URI, creation timestamp, and any associated metadata
   - Ask: "If FraudShield discovers that training data was corrupted, which models are affected?" Answer: follow the downstream lineage from that data artifact.

b. **Click on the training job (Action) node:**
   - Shows hyperparameters, instance type, training duration, and metric values
   - This is the "recipe" that produced the model

c. **Click on the container image artifact:**
   - Shows the ECR image URI and tag
   - Explain: "If AWS updates the XGBoost container image with a bug fix, lineage tells you which of your models were trained with the old version."

d. **If an endpoint exists, click on it:**
   - Shows the endpoint configuration, instance type, and creation date
   - This completes the chain: data -> training -> model -> endpoint

**Q&A Pause (2 minutes):** "Why does SageMaker track the container image in the lineage graph?" Guide Associates: reproducibility requires not just the same data and parameters, but the same software environment. A container version change can alter model behavior even with identical data and hyperparameters.

### STEP 3 -- Feature Store Lineage Integration (10 minutes)

**Conceptual Connection:**
Explain how Feature Store connects to the lineage graph:

- When training data is sourced from Feature Store (via an Athena query on the offline store), the Feature Group becomes a lineage Artifact
- The processing job that generates training data from Feature Store creates Associations between the Feature Group and the training dataset
- This means the lineage chain extends further: raw data -> Feature Store ingestion -> Feature Group -> training dataset -> training job -> model -> endpoint

**Console Navigation:**
1. Navigate to **Feature Store** > **Feature groups** > `fraudshield-customer-features`
2. Show the **Lineage** tab on the Feature Group page (if available in the current SageMaker version)
3. If the lineage tab shows associations, walk through them
4. If no direct associations exist (because no training job has used Feature Store as an input yet), demonstrate how to create one programmatically

**SDK Demonstration:**

```python
import sagemaker
from sagemaker.lineage.artifact import Artifact
from sagemaker.lineage.association import Association

session = sagemaker.Session()

feature_group_artifact = Artifact.create(
    artifact_name="fraudshield-customer-features",
    source_uri="arn:aws:sagemaker:us-east-1:123456789012:feature-group/fraudshield-customer-features",
    artifact_type="FeatureGroupDataSource",
    sagemaker_session=session,
)

training_data_artifact = Artifact.list(
    source_uri="s3://fraudshield-advanced-data/processed/train.csv",
    sagemaker_session=session,
)

Association.create(
    source_arn=feature_group_artifact.artifact_arn,
    destination_arn=list(training_data_artifact)[0].artifact_arn,
    association_type="ContributedTo",
    sagemaker_session=session,
)

print("Feature Store lineage association created")
```

**Instructor Note:** Walk through each step:
1. We create a custom Artifact representing the Feature Group
2. We find the existing Artifact for the training dataset (SageMaker auto-created this when the training job ran)
3. We create an Association linking the Feature Group to the training dataset
4. Now the lineage graph extends from Feature Store through to the deployed model

**Teaching Tip:** "In a mature pipeline, these associations are created automatically by the Pipeline steps. Manual creation is useful for retroactively documenting lineage for models that were trained before the pipeline was established."

### STEP 4 -- Querying Lineage Programmatically (10 minutes)

**SDK Demonstration:**
Show how to traverse the lineage graph in code:

```python
from sagemaker.lineage.context import Context
from sagemaker.lineage.artifact import Artifact
from sagemaker.lineage.association import Association
from sagemaker.lineage.query import LineageQuery, LineageFilter, LineageEntityEnum

session = sagemaker.Session()

model_artifact_arn = "arn:aws:sagemaker:us-east-1:123456789012:artifact/your-model-artifact-id"

query = LineageQuery(session)

query_filter = LineageFilter(
    entities=[LineageEntityEnum.ARTIFACT],
    sources=[LineageEntityEnum.DATASET],
)

upstream_results = query.query(
    start_arns=[model_artifact_arn],
    query_filter=query_filter,
    direction="Ascendants",
    include_edges=True,
)

print("Upstream data sources for this model:")
for vertex in upstream_results.vertices:
    print(f"  Type: {vertex.lineage_entity_type}, ARN: {vertex.arn}")

for edge in upstream_results.edges:
    print(f"  {edge.source_arn} --{edge.association_type}--> {edge.destination_arn}")
```

**Walk through the code:**
- `LineageQuery` is the entry point for programmatic lineage traversal
- `LineageFilter` restricts results to specific entity types (we want Dataset Artifacts)
- `direction="Ascendants"` means we traverse upstream (toward data sources)
- `direction="Descendants"` would traverse downstream (toward endpoints)
- The results include both vertices (nodes) and edges (associations)

**Downstream query example:**

```python
downstream_filter = LineageFilter(
    entities=[LineageEntityEnum.ARTIFACT],
    sources=[LineageEntityEnum.ENDPOINT],
)

downstream_results = query.query(
    start_arns=[model_artifact_arn],
    query_filter=downstream_filter,
    direction="Descendants",
    include_edges=True,
)

print("Endpoints serving this model:")
for vertex in downstream_results.vertices:
    print(f"  Endpoint ARN: {vertex.arn}")
```

**Teaching Tip:** Explain the compliance use case: "FraudShield's compliance team runs a weekly script that queries lineage for every production endpoint. The script generates a report showing: endpoint name, model version, training data source, training date, and hyperparameters. This report is archived for regulatory audit purposes."

**Q&A Pause (2 minutes):** "What happens to the lineage graph when you retrain a model and update an endpoint?" Answer: a new model artifact is created with its own upstream lineage. The endpoint's lineage association is updated to point to the new model. The old model's lineage remains intact -- you can always trace back to any previous model version.

---

[PAUSE FOR BREAK -- 10 minutes]

---

## Stage 3: Reproducibility Report and Cross-account Patterns
**Duration:** 45 minutes
**Exit Criteria Addressed:**
- Construct a reproducibility report using Experiments and Lineage data
- Explain the multi-account architecture pattern for production ML
- Configure IAM policies for cross-account Model Registry access
- Set up cross-account S3 access for shared training data and model artifacts

### Instructor Opening (3 minutes)

> "We can now track which model is best and trace how it was built. The final piece is proving that we can rebuild it. Reproducibility is not just a scientific principle -- for FraudShield, it is a regulatory requirement. A model risk management team must be able to take a production model, retrain it with the same inputs, and get the same result. We are also going to address the architecture question that comes up in every enterprise: how do you separate data, training, and deployment into different AWS accounts for security and governance?"

### STEP 1 -- Building a Reproducibility Report (12 minutes)

**Instructor Note:** This is a synthesis exercise that combines Experiments and Lineage concepts from Stages 1 and 2.

**Define the report structure:**
A reproducibility report for FraudShield should contain:

1. **Model Identity:** model name, version, creation date, Model Registry ARN
2. **Training Configuration:** algorithm, instance type, hyperparameters
3. **Data Provenance:** S3 URIs for training and validation data, data checksums (MD5 or SHA256), Feature Store feature group name and version
4. **Software Environment:** container image URI, framework version, Python package versions
5. **Results:** objective metric value, training duration, instance hours consumed
6. **Lineage Chain:** full upstream lineage from model to raw data

**SDK Demonstration -- Generating the Report:**

```python
import json
import sagemaker
from sagemaker.lineage.query import LineageQuery, LineageFilter, LineageEntityEnum

session = sagemaker.Session()
sm_client = session.sagemaker_client

training_job_name = "fraudshield-xgb-depth5-eta02"

job_info = sm_client.describe_training_job(TrainingJobName=training_job_name)

report = {
    "model_identity": {
        "training_job": training_job_name,
        "model_artifact": job_info["ModelArtifacts"]["S3ModelArtifacts"],
        "creation_date": str(job_info["CreationTime"]),
    },
    "training_configuration": {
        "algorithm_image": job_info["AlgorithmSpecification"]["TrainingImage"],
        "instance_type": job_info["ResourceConfig"]["InstanceType"],
        "instance_count": job_info["ResourceConfig"]["InstanceCount"],
        "hyperparameters": job_info["HyperParameters"],
    },
    "data_provenance": {
        "channels": {
            ch["ChannelName"]: ch["DataSource"]["S3DataSource"]["S3Uri"]
            for ch in job_info["InputDataConfig"]
        }
    },
    "results": {
        "final_metric_value": job_info.get("FinalMetricDataList", []),
        "training_duration_seconds": job_info["TrainingTimeInSeconds"],
        "billable_seconds": job_info["BillableTimeInSeconds"],
    },
}

print(json.dumps(report, indent=2, default=str))
```

**Walk through the output:**
- Point out each section and how it maps to the report structure
- Emphasize that all information comes from the SageMaker API -- nothing is manually entered
- Note that the container image URI includes the SHA256 digest, which pins the exact software version

**Extending with Lineage:**

```python
model_artifacts = Artifact.list(
    source_uri=job_info["ModelArtifacts"]["S3ModelArtifacts"],
    sagemaker_session=session,
)
model_artifact = list(model_artifacts)[0]

query = LineageQuery(session)
upstream = query.query(
    start_arns=[model_artifact.artifact_arn],
    direction="Ascendants",
    include_edges=True,
)

report["lineage_chain"] = {
    "upstream_artifacts": [
        {"arn": v.arn, "type": v.lineage_entity_type}
        for v in upstream.vertices
    ],
    "associations": [
        {
            "source": e.source_arn,
            "destination": e.destination_arn,
            "type": e.association_type,
        }
        for e in upstream.edges
    ],
}

print(json.dumps(report, indent=2, default=str))
```

**Teaching Tip:** "This report is a JSON document that can be versioned in Git, stored in a compliance database, or attached to a Model Registry model package. FraudShield generates this automatically every time a model is registered for production use."

**Q&A Pause (2 minutes):** "What is missing from this report for true reproducibility?" Guide Associates to identify: random seeds (must be set explicitly in training scripts), exact library versions within the container, data ordering (shuffling seeds), and any external dependencies.

### STEP 2 -- Cross-account Architecture Overview (10 minutes)

**Whiteboard/Diagram Discussion:**

Present FraudShield's target multi-account architecture:

```
Account A: Data Account
  - S3 buckets with raw and processed data
  - Feature Store feature groups
  - Data Wrangler flows
  - Glue Data Catalog

Account B: Training Account
  - SageMaker Studio domain
  - Training jobs and HPO tuning jobs
  - Experiments and lineage tracking
  - Model Registry (central)

Account C: Deployment Account
  - SageMaker endpoints (real-time inference)
  - Batch transform jobs
  - CloudWatch monitoring and alarms
  - Auto-scaling configuration
```

**Why three accounts?**
1. **Security isolation:** the deployment account has no access to raw data. If the endpoint is compromised, the blast radius is limited.
2. **Cost attribution:** each account has its own billing. Data storage costs are attributed to the data team; training costs to the ML team; inference costs to the product team.
3. **Permission boundaries:** data engineers have admin access to Account A but read-only access to Account B. ML engineers have admin access to Account B but no access to Account A's raw data. DevOps has admin access to Account C.
4. **Compliance:** regulated data stays in Account A. Model artifacts are the only thing that crosses account boundaries. Regulators can audit Account A independently.

**Instructor Note:** Emphasize that this is the target architecture. Many organizations start with a single account and gradually separate. FraudShield is at the point where they need to plan this transition.

### STEP 3 -- IAM Policies for Cross-account Model Registry (10 minutes)

**Scenario:** FraudShield's Model Registry lives in Account B (Training). Account C (Deployment) needs to pull approved model packages from the registry.

**Step-by-step policy setup:**

**In Account B (Training) -- Resource-based policy on the Model Package Group:**

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "AllowDeploymentAccountAccess",
            "Effect": "Allow",
            "Principal": {
                "AWS": "arn:aws:iam::DEPLOYMENT_ACCOUNT_ID:root"
            },
            "Action": [
                "sagemaker:DescribeModelPackage",
                "sagemaker:ListModelPackages",
                "sagemaker:DescribeModelPackageGroup"
            ],
            "Resource": "arn:aws:sagemaker:us-east-1:TRAINING_ACCOUNT_ID:model-package-group/fraudshield-models/*"
        }
    ]
}
```

**Instructor Note:** Walk through the policy:
- The Principal is the deployment account's root -- this allows any IAM role in that account to assume the permissions
- The Actions are read-only -- the deployment account can describe and list model packages but not create or delete them
- The Resource is scoped to the specific model package group

**In Account C (Deployment) -- IAM role policy:**

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "AllowCrossAccountModelRegistry",
            "Effect": "Allow",
            "Action": [
                "sagemaker:DescribeModelPackage",
                "sagemaker:ListModelPackages",
                "sagemaker:CreateModel",
                "sagemaker:CreateEndpointConfig",
                "sagemaker:CreateEndpoint"
            ],
            "Resource": "*"
        },
        {
            "Sid": "AllowS3ModelArtifactAccess",
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:ListBucket"
            ],
            "Resource": [
                "arn:aws:s3:::training-account-model-bucket",
                "arn:aws:s3:::training-account-model-bucket/*"
            ]
        }
    ]
}
```

**Teaching Tip:** Highlight the S3 permissions: "The model artifact (model.tar.gz) lives in Account B's S3 bucket. Account C needs cross-account S3 access to download it during endpoint creation. This requires both an IAM policy in Account C and a bucket policy in Account B."

**Q&A Pause (2 minutes):** "What happens if FraudShield deploys a model from Account B to Account C, and then Account B deletes the model artifact from S3?" Answer: the endpoint continues to serve (the model is loaded into memory on the endpoint instance), but any attempt to re-create or scale the endpoint will fail because the artifact is gone. Best practice: never delete model artifacts for production models.

### STEP 4 -- Cross-account S3 Access Patterns (8 minutes)

**Bucket Policy in Account A (Data) or Account B (Training):**

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "AllowCrossAccountRead",
            "Effect": "Allow",
            "Principal": {
                "AWS": "arn:aws:iam::DEPLOYMENT_ACCOUNT_ID:role/SageMakerDeploymentRole"
            },
            "Action": [
                "s3:GetObject",
                "s3:ListBucket"
            ],
            "Resource": [
                "arn:aws:s3:::fraudshield-model-artifacts",
                "arn:aws:s3:::fraudshield-model-artifacts/*"
            ]
        }
    ]
}
```

**Walk through the access flow:**
1. Account C's SageMaker deployment role assumes its IAM role
2. The role's policy allows `s3:GetObject` on Account B's bucket
3. Account B's bucket policy allows the specific role ARN from Account C
4. Both policies must be in place -- IAM policy in Account C AND bucket policy in Account B

**Teaching Tip:** "Cross-account access in AWS always requires a two-sided handshake: the source account must allow the external principal, and the external account must allow the action. If either side is missing, access is denied."

**Additional Patterns to Mention:**
- **KMS encryption:** if the S3 bucket uses a KMS key, the key policy must also allow cross-account access
- **VPC endpoints:** if accounts use VPC endpoints for S3, ensure the endpoint policies allow cross-account access
- **AWS Organizations:** if all accounts are in the same Organization, you can use Organization-level policies to simplify cross-account access

### STEP 5 -- Cleanup (2 minutes)

**This step is mandatory. Walk Associates through each cleanup action:**

1. **Experiments:** Experiments are metadata only -- no compute cost. They can remain for future reference. To delete if desired:
   ```python
   from sagemaker.experiments.experiment import Experiment
   experiment = Experiment.load(experiment_name="fraudshield-xgb-optimization", sagemaker_session=session)
   experiment.delete(action="--force")
   ```
   Note: this deletes all associated Runs. Use with caution.

2. **Lineage entities:** Custom Artifacts and Associations created in Stage 2 can be deleted:
   ```python
   Association.delete(source_arn=source_arn, destination_arn=dest_arn, sagemaker_session=session)
   feature_group_artifact.delete()
   ```

3. **Notebook instances:** Shut down any Studio notebook kernels:
   - Studio > Running Terminals and Kernels > Shut down all

4. **No endpoints were deployed** in this module. Confirm with Associates.

5. **Feature Store:** If the `fraudshield-customer-features` feature group will not be used in subsequent modules, delete it:
   ```python
   feature_group.delete()
   ```
   Remember to also delete the offline store data from S3.

**Instructor Note:** The most important cleanup item is notebook kernels. Experiments and lineage entities incur no ongoing cost but consume storage. Feature Store online stores incur per-read costs if queries continue.

---

## Post-Lecture Wrap-Up

### Key Takeaways (5 minutes)

1. **SageMaker Experiments** transform ad hoc training runs into organized, comparable, and searchable collections. Every production training job at FraudShield should be associated with an Experiment. Custom metric logging (including business metrics like fraud cost) enables stakeholder-relevant comparisons.

2. **Lineage Tracking** provides automatic provenance documentation for every model. The lineage graph connects data sources, processing steps, training jobs, model artifacts, and endpoints into a queryable graph. For regulated industries like financial services, lineage is not optional -- it is a compliance requirement.

3. **Reproducibility** requires deliberate effort beyond what SageMaker tracks automatically. Teams must set random seeds, pin container versions, version their training scripts, and generate reproducibility reports that capture the full context of a training run.

4. **Cross-account architecture** is the standard pattern for production ML. Separating data, training, and deployment into distinct accounts provides security isolation, cost attribution, and permission boundaries. IAM policies and S3 bucket policies must be configured on both sides of the account boundary.

### What Comes Next

This module completes the foundational infrastructure for FraudShield's advanced ML platform. Associates now have:
- A production-ready Studio domain with VPC isolation (Module 1)
- Feature Store for training-serving consistency (Module 1)
- Trained models using built-in algorithms and HPO (Module 2)
- Experiment tracking, lineage, and reproducibility patterns (Module 3)

The next skill unit will build on this foundation with SageMaker Pipelines (CI/CD for ML), Model Monitor (drift detection), and advanced deployment patterns (multi-model endpoints, inference pipelines). The Experiments and Lineage infrastructure created in this module will be integrated directly into those Pipeline steps.

### Common Pitfalls to Reinforce
- Not associating training jobs with Experiments at launch time (retroactive association is possible but cumbersome)
- Assuming SageMaker tracks everything automatically (custom metrics, data versions, and Feature Store associations require explicit logging)
- Forgetting to set random seeds in training scripts, which makes reproducibility impossible even with identical data and hyperparameters
- Configuring only one side of the cross-account access handshake (both IAM policy and resource policy are required)
- Deleting S3 model artifacts that are still referenced by production endpoints
