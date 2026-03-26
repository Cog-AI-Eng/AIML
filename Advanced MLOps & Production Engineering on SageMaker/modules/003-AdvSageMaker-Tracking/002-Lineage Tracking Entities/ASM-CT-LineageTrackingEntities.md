# Lineage Tracking Entities

**Estimated Time:** 10 Minutes

## Introduction

Experiments track *what happened* during training (parameters, metrics, artifacts). Lineage Tracking answers a different question: *how did this model come to exist?* It records the full provenance chain -- which dataset was used, which processing job transformed it, which training job produced the model, and which endpoint serves it. This chain is essential for auditability, debugging, and regulatory compliance.

SageMaker **Lineage Tracking** automatically creates and links entities whenever you use SageMaker services. When you run a Processing Job, launch a Training Job, or deploy an endpoint, SageMaker records each entity and the relationships between them in a lineage graph. You can query this graph to trace any model artifact back to its source data, or trace forward from a dataset to see all models trained on it.

## Core Concepts

### Lineage entity types

SageMaker Lineage defines four entity types:

| Entity Type | What It Represents | Example |
| :--- | :--- | :--- |
| **Artifact** | A data object or model file | S3 dataset URI, model artifact URI, Docker image URI |
| **Action** | A processing step that transforms data | Training Job, Processing Job, Transform Job |
| **Context** | A grouping container | Experiment, Pipeline Execution, Endpoint |
| **Association** | A directional relationship between entities | "Training Job *used* this Dataset" or "Training Job *produced* this Model" |

### How lineage is created

Lineage entities are created automatically by SageMaker services. You do not need to write any lineage-specific code. For example, when you launch a Training Job:

1. SageMaker creates an **Artifact** for each input data channel (the S3 URIs you specified).
2. SageMaker creates an **Action** representing the Training Job itself.
3. SageMaker creates **Associations** linking the input Artifacts to the Action (with type `ContributedTo`).
4. When the job completes, SageMaker creates an **Artifact** for the model output in S3.
5. SageMaker creates an **Association** from the Action to the model Artifact (with type `Produced`).

The result is a subgraph: `Input Data --> Training Job --> Model Artifact`.

When you then deploy that model to an endpoint, SageMaker extends the graph: `Model Artifact --> Endpoint Context`.

### Viewing lineage in the console

1. Navigate to **SageMaker > Governance > Lineage**.
2. You can search for lineage entities by name, type, or ARN.
3. Click any entity to see its properties and associated entities.
4. The **Lineage graph** view (available from a model artifact's detail page) renders the full upstream and downstream chain visually. You can trace from a deployed endpoint back to the raw dataset it was trained on.

Alternatively, from any Training Job detail page, click the **Lineage** tab to see the entities associated with that job.

### Querying lineage programmatically

The `sagemaker.lineage` module in the SDK provides query capabilities:

- **Upstream query:** Given a model artifact, find all entities that contributed to it (datasets, processing jobs, training jobs).
- **Downstream query:** Given a dataset, find all models and endpoints that were derived from it.
- **Association traversal:** Walk the graph step by step, filtering by entity type or association type.

These queries are essential for answering compliance questions like: "Which customers' data was used to train the model currently serving in production?" or "If we discover a data quality issue in dataset X, which models are affected?"

### Association types

Associations have directional types that describe the relationship:

| Association Type | Meaning | Example |
| :--- | :--- | :--- |
| `ContributedTo` | Source contributed to the destination | Dataset --> Training Job |
| `AssociatedWith` | General association | Training Job --> Experiment Run |
| `DerivedFrom` | Destination was derived from the source | Model Artifact --> Training Job |
| `Produced` | Source produced the destination | Training Job --> Model Artifact |

### Custom lineage entities

While SageMaker creates lineage automatically for its managed services, you can also create custom lineage entities for external processing steps. For example, if your data pipeline includes a step that runs outside SageMaker (e.g., a Spark job on EMR), you can create an Artifact and Action manually using the SDK and link them into the existing lineage graph. This ensures full end-to-end traceability even when not all steps are SageMaker-managed.

## Connecting to Practice

Lineage Tracking gives you the auditability layer that complements Experiments' operational tracking. The next topic, *Feature Store Lineage Integration*, shows how Feature Store entities are automatically linked into the lineage graph, enabling traceability from features to models. The module assignment will require you to query the lineage graph for a trained model and produce a provenance report showing the complete upstream chain.

## Further Learning & Resources

**Documentation and reading**

- **[SageMaker Lineage Tracking](https://docs.aws.amazon.com/sagemaker/latest/dg/lineage-tracking.html)** - *Docs*: Complete reference for lineage entity types, automatic tracking, and SDK query APIs.
- **[Lineage Tracking API Reference](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_AddAssociation.html)** - *Docs*: API documentation for creating custom lineage entities and associations.

**Interactive practice**

- **[Lineage Tracking Example](https://github.com/aws/amazon-sagemaker-examples/tree/main/sagemaker-lineage)** - *Interactive*: Sample notebook demonstrating lineage queries across a multi-step ML pipeline.
