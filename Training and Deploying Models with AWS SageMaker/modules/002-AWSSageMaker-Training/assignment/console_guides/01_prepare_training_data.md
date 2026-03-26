# Guide 1: Prepare Training Data in S3

Before SageMaker can train a model, the training data must be in Amazon S3. SageMaker downloads data from S3 into the training container at runtime. Organizing your bucket with clear prefixes is essential for managing data channels, model artifacts, and experiment outputs.

---

## Steps

### Step 1 -- Create an S3 Bucket

1. In the **AWS Management Console**, navigate to **S3** (search for "S3" in the top search bar).
2. Click **Create bucket**.
3. Configure:
   - **Bucket name:** `fraudshield-training-data-<your-initials>` (bucket names must be globally unique, so append your initials or a short identifier)
   - **Region:** Same as your SageMaker domain (`us-east-1`)
   - **Object Ownership:** ACLs disabled (recommended)
   - **Block Public Access:** Leave all blocks enabled (default -- training data should never be public)
   - **Versioning:** Disabled (fine for this lab)
4. Click **Create bucket**.

### Step 2 -- Create the Prefix Structure

S3 uses prefixes (similar to folders) to organize data. Create the following structure:

1. Open your new bucket by clicking its name.
2. Click **Create folder** and create these prefixes:
   - `data/train/` -- for training CSV files
   - `data/validation/` -- for validation CSV files (optional for this lab)
   - `output/` -- SageMaker will write model artifacts here
3. After creation, your bucket should have:

```
fraudshield-training-data-<initials>/
├── data/
│   ├── train/
│   └── validation/
└── output/
```

### Step 3 -- Upload Training Data

This lab uses the **FraudShield synthetic dataset** provided in the course materials. You can find it at:

```
data/fraudshield_transactions.csv
```

This CSV contains 2,000 transactions with 6 features and a binary `is_fraud` label (~5% fraud rate). See `data/README.md` for the full schema.

To upload:
1. Navigate into the `data/train/` prefix.
2. Click **Upload**.
3. Click **Add files** and select `fraudshield_transactions.csv` from your local copy of the course materials.
4. Click **Upload**.

### Step 4 -- Verify the Upload

1. Navigate to `data/train/` in the bucket.
2. Click on the uploaded file.
3. Note the **S3 URI** at the top of the object details page. It will look like:
   ```
   s3://fraudshield-training-data-<initials>/data/train/fraudshield_transactions.csv
   ```
4. Also note the **ARN** -- this is the resource identifier you would use in IAM policies.
5. The S3 URI for the *prefix* (not the file) is what you will provide to SageMaker as the input channel:
   ```
   s3://fraudshield-training-data-<initials>/data/train/
   ```

### Step 5 -- Connect This to the IAM Role

Recall the custom IAM role you created in Module 1 (Guide 4). That role's S3 policy referenced `fraudshield-training-data`. If your bucket name is different, you would need to update the policy ARNs to match your actual bucket name. This is the principle of least privilege in action -- permissions are scoped to specific resources.

---

## Presentation Checkpoint

Be prepared to show:
- Your S3 bucket with the organized prefix structure (`data/train/`, `data/validation/`, `output/`)
- The uploaded training data file and its S3 URI
- Explain: Why does SageMaker require data in S3 rather than reading from your local machine? (SageMaker training jobs run on managed EC2 instances that do not have access to your local filesystem. S3 is the bridge between your data and SageMaker's compute.)
- Explain: Why is the prefix structure important? (Each prefix becomes an "input channel" in SageMaker -- the `train` channel, the `validation` channel, etc. This maps to `SM_CHANNEL_TRAIN` and `SM_CHANNEL_VALIDATION` environment variables inside the training container.)

---

## AIML Connection

The *ML Lifecycle & Reproducibility* reading emphasized separating training and validation data to prevent data leakage. The S3 prefix structure you just created enforces this separation at the storage level -- training data and validation data live in different locations, making it impossible to accidentally mix them.
