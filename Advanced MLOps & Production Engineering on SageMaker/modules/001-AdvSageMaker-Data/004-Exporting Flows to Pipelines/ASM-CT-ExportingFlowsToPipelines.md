# Exporting Flows to Pipelines

**Estimated Time:** 10 Minutes

## Introduction

Data Wrangler is powerful for interactive exploration, but a Flow sitting inside Studio is a manual step that someone must run by hand. For a production ML system, data preparation needs to be automated, reproducible, and triggered on a schedule or by an event. SageMaker Pipelines is the orchestration service that makes this possible: it lets you define a DAG of processing, training, evaluation, and deployment steps that runs end to end without human intervention.

The bridge between Data Wrangler and Pipelines is the **Export** feature. With a few clicks, Data Wrangler converts your interactive Flow into a Pipeline-compatible Processing Job that runs your exact transformations on the full dataset at scale. This reading covers the export options, what the generated code looks like, and how the exported step fits into a broader Pipeline DAG.

## Core Concepts

### Export destinations

When you click the **Export** button in a Data Wrangler Flow (the "+" icon at the end of your transformation chain, then **Export to**), you see several options:

| Export Target | What It Produces | When to Use |
| :--- | :--- | :--- |
| **SageMaker Pipelines** | A Jupyter notebook containing Pipeline step definitions that wrap your Flow as a Processing Job | Automating end-to-end ML workflows |
| **Python Code** | A standalone Python script with the sagemaker SDK calls to run the Flow as a Processing Job | Integrating into custom orchestration (Airflow, Step Functions) |
| **Feature Store** | Pipeline step code that processes data and ingests the output into a Feature Group | Feeding prepared features into the Online/Offline Store |
| **S3** | Pipeline step code that writes the processed output to a specified S3 location | Producing training-ready datasets for downstream training jobs |

The most common production path is **Export to SageMaker Pipelines** or **Export to Feature Store**, because these integrate directly with the SageMaker MLOps ecosystem.

### What happens during export

When you export a Flow to Pipelines, Data Wrangler generates a Jupyter notebook (saved to your Studio EFS) that contains:

1. **An upload step** that copies the `.flow` file to S3 so the Processing Job can access it.
2. **A Processing step definition** using `sagemaker.processing.Processor` configured with the Data Wrangler container image, instance type, and instance count. The `.flow` file is passed as input, and the processor replays every transformation on the full dataset.
3. **Pipeline construction code** that wraps the Processing step into a `sagemaker.workflow.pipeline.Pipeline` object. If you exported to Feature Store, the notebook also includes an ingestion step after processing.
4. **Pipeline execution code** that submits the pipeline and monitors its status.

You can run the generated notebook as-is to test the pipeline, then refactor the code into your production codebase. The key insight: the `.flow` file is the single source of truth for your transformations. If you update the Flow in Data Wrangler, re-export to regenerate the notebook.

### Scaling behavior

The interactive Data Wrangler session processes a sample (default: 50,000 rows) on a single instance. When exported to a Processing Job, the full dataset is processed. You control scaling through two parameters in the generated code:

- **`instance_type`**: The EC2 instance type for the processing cluster. The generated notebook defaults to the same instance type as your interactive session. For larger datasets, increase to `ml.m5.4xlarge` or `ml.m5.12xlarge`.
- **`instance_count`**: The number of instances in the processing cluster. Data Wrangler's PySpark backend distributes work across instances automatically when using multiple nodes.

### Console vs. SDK workflow

You can also configure Pipeline executions from the console:

1. Navigate to **SageMaker > Pipelines** to see existing pipelines and their execution history.
2. Click a pipeline name to see its DAG visualization -- each step appears as a node.
3. Click **Start an execution** to run the pipeline with optional parameter overrides.
4. Click into a running execution to see step-level status, logs (linked to CloudWatch), and output artifacts.

This console view is useful for monitoring and debugging, but the pipeline definition itself is managed in code (the SDK). The console is read-and-execute, not a visual pipeline builder.

### Connecting exports to training

A typical Module 1 pipeline looks like:

1. **Processing Step** (exported from Data Wrangler): reads raw data from S3, applies transforms, writes processed data back to S3 or Feature Store.
2. **Training Step**: reads processed data from S3 or queries Feature Store Offline, trains a model, outputs model artifacts to S3.
3. **Evaluation Step**: loads the model, runs it against a test set, computes metrics.
4. **Condition Step**: checks metrics against a threshold. If the model passes, proceed to registration; otherwise, fail the pipeline.

You will build pipelines with this structure in later modules. For now, the key takeaway is that the Data Wrangler export produces Step 1, and the downstream steps are added in code using the SageMaker Pipeline SDK.

## Connecting to Practice

This topic connects the interactive data preparation from *Data Wrangler Flows* to the automated pipeline world you will build in later modules. The next two topics -- *Canvas No-code ML* and *Autopilot AutoML Modes* -- explore alternative paths for model building that do not require custom training scripts. In the module lecture, you will export a Data Wrangler Flow to a Pipeline and execute it end to end. The assignment will require you to demonstrate a working export that processes the full dataset and writes output to S3.

## Further Learning & Resources

**Documentation and reading**

- **[Export a Data Wrangler Flow](https://docs.aws.amazon.com/sagemaker/latest/dg/data-wrangler-data-export.html)** - *Docs*: Step-by-step export instructions for each destination type.
- **[SageMaker Pipelines Developer Guide](https://docs.aws.amazon.com/sagemaker/latest/dg/pipelines.html)** - *Docs*: Complete reference for Pipeline step types, parameters, and execution management.

**Interactive practice**

- **[MLOps with SageMaker Pipelines Workshop](https://catalog.workshops.aws/sagemaker-pipelines/en-US)** - *Interactive*: Hands-on lab that builds a multi-step pipeline from data processing through model deployment.
