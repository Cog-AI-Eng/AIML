# AdvSageMaker-Data Lecture - Instructor Guide

**Total Duration:** 180 Minutes (3 Stages)
**Consolidated Activities:** Studio & Classic Domains, Data Wrangler Flows, Feature Store Architecture, Exporting Flows to Pipelines, Canvas No-code ML, Autopilot AutoML Modes, Canvas vs. Autopilot

| Block | Content | Minutes |
|-------|---------|---------|
| Stage 1 | Studio Domain Configuration and Data Wrangler Exploration | 45 |
| Break 1 | Stretch / Questions | 10 |
| Stage 2 | Feature Store Setup and Data Pipeline Export | 45 |
| Break 2 | Stretch / Questions | 10 |
| Stage 3 | Canvas Quick Build and Autopilot Comparison | 45 |
| Buffer | Open Q&A, Wrap-Up | 25 |

---

## Lecture Overview

**Unified Scenario -- FraudShield Risk Analytics (Advanced)**

Associates return as ML engineers at FraudShield, where the fraud detection platform is maturing beyond proof-of-concept notebooks into production-grade infrastructure. The data science team has identified critical gaps: raw transaction data arrives in inconsistent formats, feature engineering steps are not reproducible, and the handoff between data preparation and model training is entirely manual. Leadership has mandated a shift toward managed data pipelines and self-service analytics tooling.

In this module, Associates tackle the data foundation layer. They will configure a production-ready SageMaker Studio domain with proper VPC isolation, build a repeatable Data Wrangler flow to transform FraudShield's e-commerce transaction data, and establish a Feature Store that serves features to both training pipelines and real-time inference endpoints. The module culminates with a comparison of Canvas and Autopilot -- two approaches that empower the broader fraud analytics team to build models without writing code, while engineers retain oversight through Studio integration.

This lecture consolidates the readings on Studio & Classic Domains, Data Wrangler Flows, Feature Store Architecture, Exporting Flows to Pipelines, Canvas No-code ML, Autopilot AutoML Modes, and Canvas vs. Autopilot into a single hands-on session that mirrors the real-world progression from raw data to ML-ready pipelines.

---

## Pre-Lecture Setup

### Instructor Checklist
- Verify that the FraudShield e-commerce transaction dataset (CSV, approximately 4000 rows) is uploaded to a versioned S3 bucket (e.g., `s3://fraudshield-advanced-data/raw/ecommerce_transactions.csv`)
- Confirm the AWS account has SageMaker Studio domain quota available (if a domain already exists from foundational modules, document the upgrade path)
- Pre-create a VPC with at least two private subnets and a NAT gateway for the Custom Domain setup demonstration
- Verify IAM roles exist: `SageMakerStudioExecutionRole` with S3, SageMaker, and Feature Store permissions
- Test that Data Wrangler launches successfully in Studio (first launch can take 5-8 minutes; do a dry run)
- Confirm Canvas is enabled in the Studio domain settings
- Prepare a backup Data Wrangler flow file (.flow) in case live creation encounters delays
- Have the AWS Pricing Calculator tab open to discuss Free Tier boundaries during the domain setup
- Verify ml.t3.medium notebook instances and ml.m5.xlarge training instances are available in the target region

### Student Prerequisites
- Completed the foundational SageMaker skill modules (Studio basics, IAM for SageMaker, training jobs, endpoints)
- AWS Console access with permissions to create SageMaker domains, Feature Store feature groups, and S3 buckets
- Familiarity with the FraudShield transaction dataset schema (transaction_id, amount, category, customer_id, is_fraud, timestamp, etc.)
- Basic understanding of feature engineering concepts (encoding, scaling, imputation)
- A running SageMaker Studio user profile or the ability to create one during the lecture

---

## Stage 1: Studio Domain Configuration and Data Wrangler Exploration
**Duration:** 45 minutes
**Exit Criteria Addressed:**
- Configure a SageMaker Studio domain using Custom Setup with VPC integration
- Distinguish between Studio and Classic domain modes and their use cases
- Create a Data Wrangler flow that imports, profiles, and transforms tabular data
- Apply encoding, imputation, and scaling transforms within Data Wrangler

### Instructor Opening (5 minutes)

> "In our foundational modules you got Studio running with Quick Setup -- one click and you were in. That works for experimentation, but FraudShield's security team has flagged a problem: Quick Setup places your notebooks on the public internet with default networking. When you are processing transaction data that includes customer identifiers and purchase histories, that is a non-starter. Today we are going to do what a real ML platform engineer does on day one: set up a production-grade Studio domain with VPC isolation, and then immediately put it to work with Data Wrangler to build a repeatable data transformation pipeline."

**Teaching Tip:** Ask Associates to raise their hands if they have ever configured a VPC in AWS. For those who have not, reassure them that the focus here is on the SageMaker-specific networking decisions, not VPC fundamentals.

### STEP 1 -- Custom Domain Setup from the Console (15 minutes)

**Console Navigation:**
1. Open the AWS Console and navigate to **Amazon SageMaker** > **Admin configurations** > **Domains**
2. Click **Create domain**
3. Select **Set up for organizations** (Custom Setup) -- do NOT select Quick Setup

**Instructor Note:** Pause here and display both options side by side. Explain the differences:
- Quick Setup: creates a default VPC, public subnets, broad IAM role. Suitable for individual experimentation only.
- Custom Setup: lets you specify VPC, subnets, encryption keys, and fine-grained IAM. Required for any environment handling sensitive data.

4. On the **Network** page, configure:
   - VPC: select the pre-created VPC with private subnets
   - Subnets: select two private subnets in different AZs
   - Security group: use the default or a pre-configured group that allows outbound HTTPS (443) to S3 and SageMaker API endpoints
   - Network access: select **VPC Only** mode

**Pacing Guidance:** Walk through each field slowly. For each setting, pause and ask: "Why does FraudShield care about this setting?" This reinforces the security rationale.

5. On the **IAM Role** page:
   - Select the pre-created `SageMakerStudioExecutionRole`
   - Show the trust policy (sagemaker.amazonaws.com) and the managed policies attached

6. On the **Storage** page:
   - Accept the default EFS configuration
   - Point out the KMS key option and explain that FraudShield would use a customer-managed key in production

7. Review and click **Create domain** (this takes 3-5 minutes)

**Q&A Pause (2 minutes):** While the domain provisions, take questions. Common questions:
- "Can we convert a Quick Setup domain to Custom?" -- Yes, but it requires re-creating user profiles and is disruptive. Better to start with Custom.
- "What about the Classic domain?" -- Classic provides individual notebook instances rather than the shared JupyterLab environment. Studio is the recommended path for new projects.

**Instructor Note:** If the domain was pre-created to save time, walk through the configuration screens in read-only mode by clicking into the existing domain's settings and reviewing each section.

### STEP 2 -- Creating a User Profile and Launching Studio (5 minutes)

1. Inside the newly created domain, click **Add user**
2. Set the user profile name to `fraudshield-analyst`
3. Assign the execution role
4. Click **Create user** and then **Launch** > **Studio**

**Teaching Tip:** While Studio loads (first launch takes 2-3 minutes), use the time to show the domain details page. Point out the VPC configuration, the EFS volume ID, and the list of apps. Explain that each user profile gets its own EFS home directory.

### STEP 3 -- Creating a Data Wrangler Flow (18 minutes)

**Console Navigation:**
1. In SageMaker Studio, click **Data** in the left sidebar, then **Data Wrangler**
2. Click **New flow** -- this launches a Data Wrangler instance (ml.t3.medium by default, which is within Free Tier constraints)

**Instructor Note:** The first Data Wrangler instance takes 3-5 minutes to spin up. Use this time to explain the architecture: Data Wrangler runs on a dedicated instance separate from the JupyterLab server. The flow is saved as a `.flow` file (JSON under the hood) that can be versioned in Git.

3. Once loaded, click **Import data** > **Amazon S3**
4. Navigate to `s3://fraudshield-advanced-data/raw/ecommerce_transactions.csv`
5. Select the file and click **Import**

**Data Profiling:**
6. Once imported, click on the dataset node and select **Add analysis** > **Data Quality and Insights Report**
7. Walk through the generated report:
   - Point out missing value percentages
   - Highlight the class imbalance in the `is_fraud` column
   - Show the correlation matrix and note which features correlate with fraud

**Pacing Guidance:** Spend 3-4 minutes on the data profile. This is a critical teaching moment -- Associates need to understand that Data Wrangler's profiling capability replaces the manual EDA they did in notebooks during foundational modules.

**Applying Transforms:**
8. Click the **+** icon after the import node and select **Add transform**
9. Apply the following transforms in sequence:

   a. **Handle Missing Values:**
      - Select **Impute** > choose the `amount` column > select **Median** imputation
      - Instructor Note: explain why median is preferred over mean for financial data with potential outliers

   b. **Encode Categoricals:**
      - Select **Encode** > **One-hot encode** > choose the `category` column
      - Show the preview to confirm new binary columns are created

   c. **Scale Numeric Features:**
      - Select **Transform** > **Standard scaler** > choose the `amount` column
      - Explain that SageMaker built-in algorithms like XGBoost do not strictly require scaling, but linear models and K-Means do -- building the pipeline with scaling keeps it algorithm-agnostic

10. Preview the transformed dataset and verify the output schema

**Q&A Pause (2 minutes):** Ask Associates: "We just built a three-step transformation pipeline visually. What happens when FraudShield gets new transaction data next week? Do we rebuild this?" Lead into the discussion of flow export in Stage 2.

**Teaching Tip:** If time permits, show one additional analysis: a histogram of the `amount` column before and after scaling to visualize the effect.

---

[PAUSE FOR BREAK -- 10 minutes]

---

## Stage 2: Feature Store Setup and Data Pipeline Export
**Duration:** 45 minutes
**Exit Criteria Addressed:**
- Explain the architecture of SageMaker Feature Store (online and offline stores)
- Create a Feature Group with a defined schema
- Ingest records into the Feature Store using the SDK
- Export a Data Wrangler flow as a SageMaker Pipeline step

### Instructor Opening (3 minutes)

> "You have a clean, transformed dataset sitting in Data Wrangler. But here is the problem FraudShield faces every day: the same features get re-engineered by different team members in different notebooks. The fraud scoring API needs features at inference time, but those features are computed differently than the ones used in training. Feature Store solves both problems -- it gives you a single source of truth for features, with an online store for millisecond lookups at inference time and an offline store for batch training. Let us build one."

### STEP 1 -- Understanding Feature Store Architecture (7 minutes)

**Console Navigation:**
1. In the AWS Console, navigate to **Amazon SageMaker** > **Feature Store** > **Feature groups**
2. If no feature groups exist, the page will show an overview diagram -- use this as a teaching aid

**Whiteboard Moment:** Sketch or describe the two-store architecture:
- **Online Store:** DynamoDB-backed, single-digit millisecond reads, stores the latest feature values per record identifier. Used at inference time.
- **Offline Store:** S3-backed (Parquet format), stores the full history of feature values with timestamps. Used for training dataset generation via Athena queries.
- **Record Identifier:** a unique key per entity (e.g., `customer_id` for FraudShield)
- **Event Time:** timestamp indicating when the feature values were observed

**Teaching Tip:** Ask: "Why would FraudShield need both stores?" Guide Associates to the answer: the fraud scoring endpoint needs to look up a customer's features in real time (online), while the training pipeline needs historical features joined with historical labels (offline).

### STEP 2 -- Creating a Feature Group from the Console (12 minutes)

1. Click **Create feature group**
2. Configure:
   - **Feature group name:** `fraudshield-customer-features`
   - **Record identifier:** `customer_id`
   - **Event time feature:** `event_time`
   - **Online store:** Enable
   - **Offline store:** Enable, specify the S3 URI: `s3://fraudshield-advanced-data/feature-store/`

3. Define the feature schema:
   - `customer_id` (String) -- record identifier
   - `event_time` (Fractional) -- event time
   - `avg_transaction_amount` (Fractional)
   - `transaction_count_30d` (Integral)
   - `distinct_categories` (Integral)
   - `fraud_rate_historical` (Fractional)
   - `account_age_days` (Integral)

**Instructor Note:** Walk through each feature definition. Explain that the schema is fixed at creation time -- you cannot add columns later without creating a new feature group. This is an important design consideration.

4. Set the IAM role to the execution role
5. Review and click **Create feature group**

**Q&A Pause (2 minutes):** Common questions:
- "Can we update feature values?" -- Yes, you ingest new records with the same `customer_id` and a newer `event_time`. The online store keeps only the latest; the offline store keeps the full history.
- "How does this relate to a data warehouse?" -- Feature Store is purpose-built for ML features, not general analytics. It ensures training-serving consistency.

### STEP 3 -- Ingesting Sample Features via SDK (10 minutes)

**Instructor Note:** This is the first point in the lecture where we move from console-first to SDK code. Remind Associates: "We created the Feature Group from the console to understand the configuration. Now we will ingest data programmatically because in production, ingestion happens from a pipeline, not from a UI."

Open a SageMaker Studio notebook (ml.t3.medium kernel) and walk through the following code:

```python
import sagemaker
from sagemaker.feature_store.feature_group import FeatureGroup
import pandas as pd
import time

session = sagemaker.Session()
region = session.boto_region_name

feature_group_name = "fraudshield-customer-features"
feature_group = FeatureGroup(name=feature_group_name, sagemaker_session=session)

sample_data = pd.DataFrame({
    "customer_id": ["C001", "C002", "C003", "C004", "C005"],
    "event_time": [float(round(time.time()))] * 5,
    "avg_transaction_amount": [150.25, 89.99, 320.50, 45.00, 210.75],
    "transaction_count_30d": [12, 3, 28, 1, 15],
    "distinct_categories": [4, 2, 8, 1, 5],
    "fraud_rate_historical": [0.02, 0.0, 0.15, 0.0, 0.05],
    "account_age_days": [365, 30, 720, 7, 180],
})

feature_group.ingest(data_frame=sample_data, max_workers=3, wait=True)
print("Ingestion complete")
```

**Pacing Guidance:** Type or paste this code live. After each block (imports, DataFrame creation, ingestion), pause and explain what is happening. Highlight that `ingest()` writes to both online and offline stores simultaneously.

**Verification from the Console:**
1. Return to **Feature Store** > **Feature groups** > `fraudshield-customer-features`
2. Click **Run query** to open the Athena query interface
3. Run: `SELECT * FROM "sagemaker_featurestore"."fraudshield-customer-features" LIMIT 10;`
4. Show the results -- the offline store data may take 5-15 minutes to appear; note this latency for Associates

**Teaching Tip:** For the online store, demonstrate a `get_record` call:

```python
record = feature_group.get_record(record_identifier_value_as_string="C001")
print(record)
```

This returns instantly and shows the latest feature values.

### STEP 4 -- Exporting the Data Wrangler Flow to a Pipeline (11 minutes)

**Console Navigation:**
1. Return to the Data Wrangler flow created in Stage 1
2. Click the **Export** tab (or the export icon on the final transform node)
3. Select **Export to** > **SageMaker Pipelines (via Jupyter Notebook)**

**Instructor Note:** Data Wrangler generates a notebook that contains all the SDK code needed to create a SageMaker Pipeline with a Processing step. Walk through the generated code, highlighting:

- The `.flow` file is uploaded to S3 and referenced as an input
- The Processing step uses `sagemaker.processing.Processor` with the Data Wrangler container image
- The output is written to S3 in the format specified (CSV or Parquet)

4. Show the generated notebook structure:
   - Cell 1: imports and session setup
   - Cell 2: upload the `.flow` file to S3
   - Cell 3: define the Processing step
   - Cell 4: define the Pipeline and create/update it
   - Cell 5: start a pipeline execution

5. Do NOT run the full pipeline yet (that is for the MLOps pipeline module). Instead, run cells 1-3 to show that the Pipeline definition is created without errors.

**Q&A Pause (2 minutes):** "We just turned a visual flow into a code-based pipeline step. Why is this important?" Guide Associates to answer: reproducibility, version control, automation, and integration with CI/CD.

---

[PAUSE FOR BREAK -- 10 minutes]

---

## Stage 3: Canvas Quick Build and Autopilot Comparison
**Duration:** 45 minutes
**Exit Criteria Addressed:**
- Launch SageMaker Canvas and build a Quick Build model
- Launch an Autopilot job and distinguish between HPO and Ensembling modes
- Compare Canvas and Autopilot outputs for the same dataset
- Share a Canvas model with Studio for further refinement

### Instructor Opening (3 minutes)

> "FraudShield's data analysts want to experiment with fraud detection models but they do not write Python. The engineering team needs a way to let analysts build baseline models without creating bottlenecks. AWS provides two paths: Canvas for a fully visual, no-code experience, and Autopilot for an automated ML service that generates code you can inspect and modify. We are going to try both on the same dataset and see where each fits in FraudShield's workflow."

### STEP 1 -- Launching SageMaker Canvas (5 minutes)

**Console Navigation:**
1. In SageMaker Studio, click the **Canvas** icon in the left application launcher
2. If Canvas is not yet enabled, navigate to the **Domain settings** > **Canvas settings** and enable it

**Instructor Note:** Canvas launches as a separate application within the Studio domain. The first launch provisions a dedicated Canvas app and takes 3-5 minutes. Use this time to explain the Canvas architecture:
- Canvas runs on its own compute, separate from JupyterLab
- It connects to the same S3 buckets and IAM roles as Studio
- Models built in Canvas can be shared back to Studio as SageMaker model artifacts

3. Once Canvas loads, walk through the interface:
   - **My Models:** where saved models appear
   - **My Datasets:** where imported datasets appear
   - **Ready-to-use models:** pre-trained models for common tasks (sentiment, entity extraction)

### STEP 2 -- Building a Quick Build Model in Canvas (15 minutes)

1. Click **My Datasets** > **Import** > **S3**
2. Navigate to `s3://fraudshield-advanced-data/raw/ecommerce_transactions.csv` and import

**Pacing Guidance:** Walk through the import step by step. Canvas will profile the dataset automatically -- show the column statistics and data type detection.

3. Click **My Models** > **New Model**
4. Name the model: `fraudshield-fraud-detector`
5. Select the imported dataset
6. Select the target column: `is_fraud`

**Instructor Note:** Canvas automatically detects this as a binary classification problem. Point out the problem type indicator and explain that Canvas supports classification, regression, and time series forecasting.

7. Select **Quick Build** (not Standard Build)

**Teaching Tip:** Explain the difference:
- **Quick Build:** trains a sample of algorithms for 2-15 minutes. Good for rapid iteration and baseline establishment. Does not perform exhaustive HPO.
- **Standard Build:** runs the full Autopilot pipeline under the hood (up to 2 hours). Produces a more optimized model.

8. Click **Build** and wait for completion (Quick Build typically takes 5-10 minutes)

**While waiting, discuss:**
- Canvas pricing: Canvas usage is billed per session hour. Quick Build is significantly cheaper than Standard Build.
- When to use Canvas vs. writing code: Canvas is ideal for business analysts who need to validate hypotheses quickly. Engineers should use it for rapid prototyping before investing in custom training scripts.

9. When the build completes, review the results:
   - Model accuracy / F1 score
   - Feature importance ranking
   - Column impact analysis

**Q&A Pause (2 minutes):** "The Quick Build model shows 92% accuracy. Should FraudShield deploy this to production?" Guide Associates to consider: class imbalance effects on accuracy, the need for precision/recall analysis, and the lack of hyperparameter control.

### STEP 3 -- Launching an Autopilot Job (12 minutes)

**Console Navigation:**
1. Navigate to **Amazon SageMaker** > **AutoML** > **Create AutoML job** (from the main SageMaker console, not Canvas)
2. Configure the job:
   - **Job name:** `fraudshield-autopilot-hpo`
   - **Input data:** `s3://fraudshield-advanced-data/raw/ecommerce_transactions.csv`
   - **Target column:** `is_fraud`
   - **Output S3 location:** `s3://fraudshield-advanced-data/autopilot-output/`

3. Under **Training method**, show the three modes:
   - **Ensembling:** trains multiple algorithms and creates a weighted ensemble. Higher accuracy, longer runtime. Uses AutoGluon under the hood.
   - **Hyperparameter Optimization (HPO):** trains a single algorithm family with Bayesian HPO. Produces a single deployable model. More interpretable.
   - **Auto:** lets Autopilot choose based on dataset characteristics.

**Instructor Note:** Select **HPO** mode for this exercise. Explain that HPO mode is preferred when FraudShield needs a single, interpretable model rather than a complex ensemble.

4. Under **Advanced settings:**
   - Set max candidates to **10** (to limit cost and time)
   - Set max runtime per training job to **300 seconds**
   - Set the objective metric to **F1**

5. Review the IAM role configuration and click **Create job**

**Teaching Tip:** While the Autopilot job runs (it will take 15-30 minutes to complete fully), switch between the console views:
- Show the **Job details** page with the list of trials as they appear
- Show the **Candidates** tab as models are trained and ranked
- Explain that Autopilot generates two notebooks: a Data Exploration notebook and a Candidate Definition notebook

6. If time permits and some candidates have completed, click into one candidate to show:
   - The algorithm used
   - The hyperparameters selected
   - The objective metric value
   - The model artifact location in S3

### STEP 4 -- Comparing Canvas and Autopilot Results (5 minutes)

Create a comparison table on screen or whiteboard:

| Dimension | Canvas Quick Build | Autopilot HPO |
|-----------|-------------------|---------------|
| Setup time | 2 minutes (visual) | 5 minutes (console form) |
| Training time | 5-10 minutes | 15-120 minutes |
| Code generated | None | Two notebooks (exploration + candidates) |
| Customization | Minimal | Full control over objective, candidates, runtime |
| Deployment | One-click from Canvas | Deploy from console or SDK |
| Audience | Business analysts | ML engineers, data scientists |

**Instructor Note:** Emphasize that these are complementary, not competing tools. FraudShield's workflow: analysts prototype in Canvas, engineers validate and optimize with Autopilot, then production models are built with custom training scripts.

### STEP 5 -- Canvas-to-Studio Model Sharing (3 minutes)

1. Return to Canvas
2. On the Quick Build model results page, click **Share** > **Share to Studio**
3. Select the target user profile and add a description
4. Switch to Studio and navigate to **Models** > **Shared models**
5. Show that the Canvas model appears as a SageMaker model artifact that can be deployed, registered in Model Registry, or used as a baseline for further training

**Teaching Tip:** This sharing workflow is critical for FraudShield's team dynamics -- it bridges the gap between no-code exploration and production engineering.

### STEP 6 -- Cleanup (2 minutes)

**This step is mandatory. Walk Associates through each cleanup action:**

1. **Canvas:** Close the Canvas application from Studio to stop billing
   - Studio > Canvas app > **Shut down**
2. **Data Wrangler:** Shut down the Data Wrangler instance
   - Studio > Running instances > Shut down the Data Wrangler kernel
3. **Autopilot:** The job will complete on its own; no ongoing charges after completion. Delete the job artifacts from S3 if desired:
   - Navigate to S3 > `fraudshield-advanced-data/autopilot-output/` > Delete
4. **Feature Store:** If this was a demonstration-only feature group:
   ```python
   feature_group.delete()
   ```
   Note: deleting a feature group removes the online store immediately but the offline store data in S3 must be deleted separately.
5. **Studio Domain:** Do NOT delete the domain if it will be used in subsequent modules. If this is the final module, follow the full domain deletion process (delete apps > delete user profiles > delete domain).

**Instructor Note:** Verbally confirm with every student that they have shut down Canvas and Data Wrangler instances. These are the most common sources of unexpected charges.

---

## Post-Lecture Wrap-Up

### Key Takeaways (5 minutes)

1. **Custom Domain Setup** is non-negotiable for production workloads. VPC-Only mode, private subnets, and customer-managed encryption keys form the security baseline for any regulated industry like financial services.

2. **Data Wrangler** transforms visual data preparation into reproducible pipeline steps. The flow file is the bridge between interactive exploration and automated pipelines.

3. **Feature Store** eliminates training-serving skew by providing a unified feature repository with both real-time (online) and batch (offline) access patterns.

4. **Canvas and Autopilot** serve different audiences but share the same underlying infrastructure. FraudShield benefits from both: Canvas for analyst-driven exploration, Autopilot for engineer-supervised optimization.

### What Comes Next

In the next module (AdvSageMaker-Algorithms), associates will move from data preparation to model training. They will use the features stored in Feature Store as inputs to SageMaker built-in algorithms -- XGBoost for fraud classification, K-Means for customer segmentation, and Random Cut Forest for anomaly detection. The Data Wrangler flow and Feature Store created today become the foundation for those training jobs.

### Common Pitfalls to Reinforce
- Forgetting to shut down Canvas and Data Wrangler instances (billing continues until explicitly stopped)
- Using Quick Setup for domains that will eventually handle sensitive data (migration is painful)
- Not setting an `event_time` feature in Feature Store (required for point-in-time correct training datasets)
- Assuming Autopilot replaces custom training -- it is a starting point and baseline generator, not an end-to-end solution for complex use cases
