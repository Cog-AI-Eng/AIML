# Guide 3: Create a Feature Group

This guide walks you through creating a SageMaker Feature Store feature group with both online and offline stores enabled. You will define the feature definitions for FraudShield transaction data and ingest sample records so that downstream training and inference jobs can query curated features.

---

## Steps

### Step 1 -- Navigate to Feature Store
1. In the SageMaker console left navigation pane, expand **Data** and click **Feature Store**.
2. Click **Create feature group**.

### Step 2 -- Configure Feature Group Settings
1. Enter the feature group name: `fraudshield-txn-features`.
2. For **Record identifier feature name**, enter `transaction_id`.
3. For **Event time feature name**, enter `event_time`.
4. Under **Online store**, toggle the switch to **Enable**.
5. Under **Offline store**, toggle the switch to **Enable**.
6. For the offline store S3 URI, enter `s3://fraudshield-data-<account-id>/feature-store/`.
7. Leave the data catalog settings at their defaults (this registers the offline store in the AWS Glue Data Catalog).

### Step 3 -- Define Feature Definitions
1. Click **Add feature** and define the following features (adjust types as needed for your dataset):
   - `transaction_id` -- Type: **String**
   - `event_time` -- Type: **Fractional**
   - `transaction_amount` -- Type: **Fractional**
   - `customer_age` -- Type: **Integral**
   - `payment_method_encoded` -- Type: **Fractional**
   - `is_fraud` -- Type: **Integral**
2. Add any additional transformed features from your Data Wrangler output.
3. Verify each feature name and type. Names must be lowercase with underscores only.

### Step 4 -- Set IAM Role and Create
1. Under **IAM role**, select the execution role you created in Guide 1 (`AmazonSageMaker-ExecutionRole-...`).
2. Review the summary: group name, record identifier, event time, online store enabled, offline store S3 path.
3. Click **Create feature group**.
4. Wait for the status to change to **Created** (this typically takes 1-2 minutes).

### Step 5 -- Ingest Sample Records
1. Open SageMaker Studio and launch a **System terminal** or a **Notebook** (use `ml.t3.medium` instance type).
2. Run the following Python snippet to ingest sample records (update bucket and feature values to match your data):
   ```python
   import boto3, time
   sm = boto3.client("sagemaker-featurestore-runtime", region_name="us-east-1")
   records = [
       {"transaction_id": "TXN-0001", "event_time": str(time.time()),
        "transaction_amount": 149.99, "customer_age": 34, "payment_method_encoded": 1.0, "is_fraud": 0},
       {"transaction_id": "TXN-0002", "event_time": str(time.time()),
        "transaction_amount": 4999.00, "customer_age": 28, "payment_method_encoded": 0.0, "is_fraud": 1},
   ]
   for r in records:
       sm.put_record(FeatureGroupName="fraudshield-txn-features",
                     Record=[{"FeatureName": k, "ValueAsString": str(v)} for k, v in r.items()])
   print("Records ingested.")
   ```
3. Verify ingestion by calling `get_record` with `transaction_id = "TXN-0001"` and confirming values are returned.

### Step 6 -- Verify Online and Offline Stores
1. Return to the **Feature Store** page in the SageMaker console.
2. Click `fraudshield-txn-features` and open the **Online store** tab to confirm records exist.
3. Open the **Offline store** tab and note the S3 path and Glue table name.
4. Navigate to S3 and verify that Parquet files have been written under the offline store prefix (this may take up to 15 minutes after ingestion).

---

## Presentation Checkpoint
Be prepared to show:
- The feature group detail page with online and offline stores both showing as **Active**.
- The feature definitions list with correct names and types.
- The result of a `get_record` call returning a sample FraudShield transaction.

---

## Key Concepts
- **Feature Group:** A logical collection of features stored together, serving as the unit of organization in SageMaker Feature Store.
- **Online Store:** A low-latency key-value store backed by an in-memory cache, designed for real-time inference lookups.
- **Offline Store:** A batch-oriented store that writes feature values as Parquet files in S3, suitable for training dataset generation.
- **Record Identifier:** The primary key column that uniquely identifies each record in the feature group.
- **Event Time:** A timestamp column that enables point-in-time queries to prevent data leakage during training.
