# Batch Transform Architecture

**Estimated Time:** 10 Minutes

## Introduction

Batch Transform is the simplest inference option in SageMaker: no endpoint to manage, no scaling to configure, no cleanup to remember. You point SageMaker at a model artifact and an input dataset in S3, it spins up a cluster, processes every record, writes the predictions to S3, and tears the cluster down. The entire lifecycle is a single API call.

Despite its simplicity, Batch Transform is the right choice for a large class of production workloads: nightly scoring runs, periodic re-ranking, retroactive analysis of historical data, and generating training labels from a teacher model. This reading covers the architecture, configuration, and operational patterns for Batch Transform jobs.

## Core Concepts

### How Batch Transform works

1. **You create a transform job** specifying: the model name (or model artifact S3 URI), the S3 input path, the S3 output path, instance type, and instance count.
2. **SageMaker provisions a cluster** of instances and loads the model on each.
3. **SageMaker splits the input data** across instances. Each instance processes its partition independently.
4. **Predictions are written** to the specified S3 output path. Output files mirror the input file names with a `.out` extension.
5. **SageMaker tears down the cluster** when processing is complete.

### Creating a Batch Transform job in the console

1. Navigate to **SageMaker > Inference > Batch transform > Create batch transform job**.
2. **Job name:** Enter a descriptive name (e.g., `fraud-scoring-2026-03`).
3. **Model name:** Select the model to use (must already be created in SageMaker).
4. **Instance type:** `ml.m5.xlarge` for CPU workloads. Match the instance type used during training for compatibility.
5. **Instance count:** Number of instances in the processing cluster. More instances = faster processing for large datasets.
6. **Input data configuration:**
   - **S3 data type:** `S3Prefix` (process all files under a prefix) or `ManifestFile` (process specific files listed in a manifest).
   - **S3 URI:** Path to input data.
   - **Content type:** `text/csv`, `application/json`, etc.
   - **Split type:** `Line` (each line is a record) or `None` (each file is a single record).
7. **Output data configuration:** S3 path for predictions.
8. Click **Create job**.

### Join source with predictions

By default, Batch Transform output contains only predictions. To include the original input alongside predictions (useful for matching predictions back to records):

- Set **Join source** to `Input` in the output configuration. SageMaker appends each prediction to its corresponding input record.
- Set **Accept** to specify the output format (e.g., `text/csv`).

This is essential for production scoring where you need to associate predictions with customer IDs, order IDs, or other identifiers from the input data.

### Data distribution strategies

When using multiple instances, SageMaker distributes input data using one of two strategies:

- **FullyReplicated:** Every instance receives the entire dataset. Use this when the model needs access to all data (rare for standard inference).
- **ShardedByS3Key (default):** Each instance receives a different partition of the input data. This is the standard approach for parallel scoring.

### Error handling

Batch Transform jobs can fail partially: some records may produce errors while others process successfully. Configuration options:

- **Max payload size:** Maximum size (in MB) of each request sent to the model. If individual records exceed this, the job fails for those records.
- **Max concurrent transforms:** Number of records processed in parallel on each instance. Higher values increase throughput but require more memory.
- **Failed records:** SageMaker writes error information to the output path. Check for `.out` files alongside error logs in CloudWatch.

### Batch Transform vs. offline pipeline scoring

An alternative to Batch Transform is running scoring inside a SageMaker Processing Job or Pipeline step. The comparison:

| Factor | Batch Transform | Processing Job |
| :--- | :--- | :--- |
| Model loading | Automatic (SageMaker loads the model) | Manual (your script loads the model) |
| Data splitting | Automatic across instances | Manual (your script partitions data) |
| Output format | Prediction per record, auto-joined | Fully custom |
| Flexibility | Limited to model's inference logic | Full custom code |
| Use case | Standard model scoring | Custom post-processing, multi-model logic |

Use Batch Transform when you have a standard model and want SageMaker to handle the infrastructure. Use Processing Jobs when you need custom scoring logic, post-processing, or multi-model ensemble scoring.

## Connecting to Practice

Batch Transform handles the bulk scoring use case from the Inference Decision Matrix. The next topic, *Multi-Model Endpoints*, covers how to serve multiple models from a single real-time endpoint for cost efficiency. The module assignment will require you to run a Batch Transform job with join source enabled, verify the output, and compare processing time across different instance counts.

## Further Learning & Resources

**Documentation and reading**

- **[Batch Transform](https://docs.aws.amazon.com/sagemaker/latest/dg/batch-transform.html)** - *Docs*: Complete reference for Batch Transform job configuration, data formats, and error handling.
- **[Batch Transform Best Practices](https://docs.aws.amazon.com/sagemaker/latest/dg/batch-transform-data-processing.html)** - *Docs*: Guidance on data processing options, join source, and filtering.

**Interactive practice**

- **[Batch Transform Example](https://github.com/aws/amazon-sagemaker-examples/tree/main/sagemaker_batch_transform)** - *Interactive*: Sample notebook demonstrating Batch Transform with join source and multi-instance processing.
