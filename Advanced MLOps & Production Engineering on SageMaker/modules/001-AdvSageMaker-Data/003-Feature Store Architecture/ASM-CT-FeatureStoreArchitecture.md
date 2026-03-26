# Feature Store Architecture

**Estimated Time:** 10 Minutes

## Introduction

In a typical ML workflow, feature engineering happens inside a training notebook: you compute features, feed them to a model, and move on. The problem surfaces when a second team member needs the same features for a different model, or when the inference pipeline needs to replicate the exact same transformations at prediction time. Without a shared system, teams end up with duplicated feature code, inconsistent transformations between training and serving, and no way to audit which features went into a specific model version.

SageMaker Feature Store solves this by providing a centralized, managed repository for storing, versioning, and serving ML features. It offers two storage layers -- an **Online Store** for low-latency lookups during inference and an **Offline Store** for batch training -- backed by the same feature definitions. This dual-store architecture ensures that the features your model trains on are identical to the features it receives at prediction time, eliminating training-serving skew.

This reading covers the Feature Store architecture, how to create Feature Groups in the console, and how the online and offline stores interact.

## Core Concepts

### Feature Groups

A **Feature Group** is a logical collection of features with a shared schema. Think of it as a table: each row is a **Record** identified by a **Record Identifier** (a primary key), and each column is a **Feature** with a defined data type (String, Integral, or Fractional). Every Feature Group also requires an **Event Time** feature -- a timestamp that enables point-in-time lookups for historical training data.

To create a Feature Group in the console: **SageMaker > Feature Store > Feature groups > Create feature group**. You define the group name, record identifier, event time feature, and feature definitions (name + type pairs). You also choose whether to enable the Online Store, the Offline Store, or both.

### Online Store

The Online Store is backed by a low-latency key-value store. When a feature record is ingested, it is immediately available for retrieval via `GetRecord` API calls. Typical latency is single-digit milliseconds.

Use the Online Store when:
- Your inference endpoint needs real-time feature lookups (e.g., retrieving the latest customer transaction count before making a fraud prediction).
- You need the most recent version of a feature value, not historical snapshots.

The Online Store keeps only the latest version of each record (keyed by the record identifier). Older versions are overwritten on update.

### Offline Store

The Offline Store persists all feature records (including historical versions) to Amazon S3 in Parquet format. It is optimized for batch reads -- the kind you do when assembling a training dataset.

Use the Offline Store when:
- Building training datasets that require point-in-time joins (e.g., "what were this customer's features at the moment each historical transaction occurred?").
- Running batch inference across millions of records.
- Auditing which feature values were used for a specific model training run.

The Offline Store is append-only: new records and updates are added as new rows with their event timestamps. SageMaker periodically syncs ingested records from the Online Store to the Offline Store (typically within 15 minutes).

### Dual-store architecture

The power of Feature Store is running both stores from a single Feature Group definition:

1. Your data pipeline ingests features using `PutRecord` (online) or batch ingestion.
2. The Online Store immediately reflects the latest values for serving.
3. SageMaker automatically replicates records to the Offline Store in S3.
4. Training jobs read from the Offline Store using Athena queries or direct S3 access.

This means the feature definitions, data types, and transformations are identical between training and serving. If you change a feature computation, you update the ingestion pipeline once, and both stores reflect the change.

### Creating a Feature Group in the console

1. Navigate to **SageMaker > Feature Store > Feature groups**.
2. Click **Create feature group**.
3. **Group name:** Use a descriptive name (e.g., `customer-transaction-features`).
4. **Record identifier:** Select or type the feature name that serves as the primary key (e.g., `customer_id`).
5. **Event time:** Select or type the timestamp feature (e.g., `event_time`).
6. **Feature definitions:** Add each feature with its name and type. You can upload a schema JSON file or define features manually.
7. **Online Store configuration:** Toggle on and accept defaults (or configure a custom KMS key for encryption).
8. **Offline Store configuration:** Toggle on and specify the S3 URI where Parquet files will be stored (e.g., `s3://your-bucket/feature-store/`).
9. **IAM role:** Select an execution role with permissions to write to S3 and the Feature Store service.
10. Click **Create feature group**.

After creation, the Feature Group appears in the console list. You can inspect its schema, view ingestion metrics, and run sample queries from the console.

### Ingesting features

Features can be ingested through:
- **`PutRecord` API** (single record, real-time): Used in streaming pipelines or Lambda functions to update the Online Store as events arrive.
- **Batch ingestion** (bulk, offline): Used to backfill the Offline Store from historical data. You can run this as a SageMaker Processing Job or a Data Wrangler export.

### Querying the Offline Store

The Offline Store data in S3 is registered as an AWS Glue Data Catalog table automatically. This means you can query it with Amazon Athena using standard SQL. SageMaker provides a helper method in the SDK to construct point-in-time join queries, but you can also write the Athena SQL manually in the console at **Amazon Athena > Query editor**.

## Connecting to Practice

Feature Store is the destination for the data pipeline you are building in this module. In the previous topic, *Data Wrangler Flows*, you learned to prepare data visually. The next topic, *Exporting Flows to Pipelines*, shows how to push Data Wrangler output directly into a Feature Group as a Pipeline step. The module assignment will have you create a Feature Group, ingest features, and query the Offline Store with Athena.

## Further Learning & Resources

**Documentation and reading**

- **[SageMaker Feature Store Developer Guide](https://docs.aws.amazon.com/sagemaker/latest/dg/feature-store.html)** - *Docs*: Complete reference for Feature Store concepts, API operations, and security configuration.
- **[Feature Store Quotas and Limits](https://docs.aws.amazon.com/sagemaker/latest/dg/feature-store-quotas.html)** - *Docs*: Service limits for record size, feature count, and ingestion throughput.

**Interactive practice**

- **[SageMaker Feature Store Workshop](https://catalog.workshops.aws/sagemaker-featurestore/en-US)** - *Interactive*: End-to-end lab covering Feature Group creation, ingestion patterns, and point-in-time queries.
