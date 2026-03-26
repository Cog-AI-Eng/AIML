# Reproducibility Patterns

**Estimated Time:** 10 Minutes

## Introduction

Reproducibility -- the ability to recreate a model's exact training conditions and produce the same results -- is a foundational requirement for both scientific rigor and production reliability. In the foundational AIML skill, you practiced reproducibility at the code level: setting `random_state=42`, documenting library versions, and using consistent train/test splits. At the SageMaker scale, reproducibility requires tracking infrastructure, data versions, and configuration across distributed cloud services.

This reading synthesizes the tracking tools from this module (Experiments, Lineage, Feature Store) into concrete reproducibility patterns that you can apply to any SageMaker ML project.

## Core Concepts

### The four pillars of ML reproducibility

To reproduce a model, you must capture and version four things:

| Pillar | What to Track | SageMaker Tool |
| :--- | :--- | :--- |
| **Data** | Exact dataset version used for training, including any feature transformations | Feature Store (Offline Store with event timestamps), S3 versioning |
| **Code** | Training script, preprocessing code, and library versions | Script Mode entry point, container image URI, Git integration |
| **Configuration** | Hyperparameters, instance types, resource allocation | Experiments (Run parameters), Training Job configuration |
| **Environment** | Python version, library versions, Docker image | SageMaker container URI (pinned version), requirements.txt |

### Pattern 1: Data versioning with Feature Store

Rather than reading training data from a mutable S3 path (where someone could overwrite the file), use the Feature Store Offline Store:

1. Ingest features into a Feature Group with event timestamps.
2. When building a training dataset, use a point-in-time query with a specific cutoff date (e.g., "all features as of 2026-03-01").
3. Record the query parameters (Feature Group name, cutoff timestamp, filter conditions) as metadata in the Experiment Run.
4. To reproduce: re-run the same query with the same cutoff. The Offline Store is append-only, so the result will be identical.

If you are not using Feature Store, enable **S3 versioning** on your data bucket and record the S3 Version ID of the training data file in the Experiment Run metadata.

### Pattern 2: Code versioning with container pinning

SageMaker training jobs run inside Docker containers. For reproducibility:

1. **Pin the container image** to a specific tag or digest rather than using `latest`. SageMaker's managed containers have version-specific URIs (e.g., `683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-xgboost:1.7-1`).
2. **Version your training script** in Git. Record the commit hash in the Experiment Run metadata.
3. **Pin library versions** in `requirements.txt` if your Script Mode entry point installs additional packages.

The combination of a pinned container + a specific Git commit + a pinned requirements file guarantees that the same code and dependencies run in reproduction.

### Pattern 3: Configuration capture with Experiments

SageMaker Experiments automatically captures hyperparameters for every Run associated with a training job. To enhance this:

1. Log custom parameters (preprocessing choices, feature selection decisions, split ratios) as Experiment Run parameters.
2. Log the instance type and instance count as metadata (SageMaker captures this automatically in the Training Job details, but explicitly logging it in the Run makes it visible in comparison views).
3. Tag Runs with meaningful identifiers (e.g., `model_version=v3.2`, `triggered_by=weekly_pipeline`).

### Pattern 4: Full-chain lineage for audit

When an auditor or regulator asks "how was this model built?", you need to produce the full provenance chain:

1. **Start at the deployed endpoint.** Query Lineage to find the Model Artifact.
2. **From the Model Artifact, trace upstream.** Find the Training Job Action, its input data Artifacts (with S3 Version IDs or Feature Group references), and the container image Artifact.
3. **From the Training Job, find the Experiment Run.** Pull the parameters, metrics, and custom metadata.
4. **From the data Artifacts, trace further upstream.** Find Processing Jobs or Data Wrangler Flows that produced the training data, and the raw data sources they ingested.

This chain provides a complete reproducibility record. Document it in a standardized format (a "Model Card" or "Reproducibility Report") for each production model.

### Pattern 5: Pipeline-as-code for end-to-end reproduction

The ultimate reproducibility pattern is to define your entire workflow -- data processing, training, evaluation, and deployment -- as a SageMaker Pipeline in version-controlled code. To reproduce any past model:

1. Check out the specific Git commit of the pipeline code.
2. Set the Pipeline Parameters to the same data paths, hyperparameter ranges, and thresholds used in the original execution.
3. Execute the pipeline. The same code, on the same data, with the same configuration, produces the same model.

Pipeline executions are recorded in the SageMaker Pipelines console with full step-level details, making it easy to find the execution ID for any past model and extract the exact parameters used.

## Connecting to Practice

This topic provides the synthesis of all tracking tools into actionable reproducibility practices. The next topic, *Cross-account Sharing*, covers how to maintain reproducibility and governance when ML resources span multiple AWS accounts. The module assignment will require you to produce a reproducibility report for a model that traces from endpoint to raw data using Experiments, Lineage, and Feature Store queries.

## Further Learning & Resources

**Documentation and reading**

- **[SageMaker ML Governance](https://docs.aws.amazon.com/sagemaker/latest/dg/governance.html)** - *Docs*: Overview of SageMaker governance capabilities including Experiments, Lineage, and Model Cards.
- **[Reproducible ML with SageMaker](https://aws.amazon.com/blogs/machine-learning/ensure-reproducibility-of-machine-learning-models-with-amazon-sagemaker/)** - *Blog*: AWS blog post with practical patterns for reproducible ML workflows.

**Interactive practice**

- **[ML Governance Workshop](https://catalog.workshops.aws/sagemaker-governance/en-US)** - *Interactive*: Hands-on lab covering Experiments, Lineage, and Model Cards in a governance workflow.
