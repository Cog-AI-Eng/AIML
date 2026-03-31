# Week 2 Tuesday -- Advanced SageMaker Data Engineering

**Total Duration:** 185 Minutes (3 Stages)
**Consolidated Activities:**
- SM Data Engineering: Feature Store, Feature Groups, Online/Offline Stores
- SM Tools: Data Wrangler, Canvas, Autopilot
- SM Pipelines: Connecting Feature Store to Pipelines, Pipeline Steps, Model Registry Integration

| Block | Content | Minutes |
|-------|---------|---------|
| Stage 1 | Feature Store Fundamentals | 55 |
| Break 1 | Stretch / Questions | 5 |
| Stage 2 | Data Wrangler, Canvas, and Autopilot | 55 |
| Break 2 | Stretch / Questions | 5 |
| Stage 3 | Exporting to Pipelines | 55 |
| Buffer | Open Q&A, Summary, Wednesday Preview | 10 |

---

## Lecture Overview

**Unified Scenario -- FraudShield Risk Analytics**

On Monday, associates deployed their trained models to real-time endpoints, invoked them with validation data, and evaluated performance with precision, recall, F1, and confusion matrices. They also opened the LSTM black box and learned the MLOps concepts that frame automated ML workflows. But the models consumed raw CSVs directly -- no feature management, no consistency guarantees, and no reuse across teams.

Today shifts the focus from model deployment to the data that feeds those models. Feature engineering is often the most time-consuming and error-prone part of any ML workflow. If the features used during training differ even slightly from those used at inference time, the model's performance degrades silently -- a problem called training-serving skew. SageMaker Feature Store solves this by providing a centralized, versioned, consistent feature pipeline for both training and real-time inference.

After mastering Feature Store, associates explore the higher-level tools that sit on top of SageMaker's data and training infrastructure: Data Wrangler for visual data preparation, Canvas for no-code ML, and Autopilot for automated model selection and tuning. The day closes by connecting these data engineering components to the pipelines introduced on Monday, completing the picture from raw data to registered model.

1. **"How do we stop copy-pasting feature logic between training and inference?"** (Feature Store, Feature Groups, online/offline stores)
2. **"What tools exist for people who are not writing Python all day?"** (Data Wrangler visual flows, Canvas no-code ML)
3. **"How do we automatically find the best model for our data?"** (Canvas vs Autopilot, AutoML)
4. **"How do we wire feature engineering into an automated pipeline?"** (Pipeline steps, Model Registry integration)

Each stage builds on Monday's deployment and MLOps concepts, adding the data engineering layer underneath.

---

## Pre-Lecture Setup

### Instructor Checklist

- [ ] Monday's notebook completed (RF and CNN endpoints deleted, Model Registry populated)
- [ ] SageMaker execution role ARN ready with Feature Store permissions
- [ ] Companion lecture notebook (`W2-Tuesday-notebook.ipynb`) open and tested
- [ ] AWS account with SageMaker, Feature Store, and Athena access verified
- [ ] Synthetic fraud dataset available (same schema as Friday/Monday)
- [ ] This instructor guide open in a second tab
- [ ] Verify Athena workgroup exists for offline store queries

### Student Prerequisites

- [ ] Completed readings: Feature Store CT, Data Wrangler CT, Canvas CT, Autopilot CT, Advanced SageMaker Data CTs
- [ ] Monday's notebook completed (models deployed, evaluated, endpoints cleaned up)
- [ ] AWS credentials configured, SageMaker SDK installed
- [ ] Familiarity with Model Registry and Pipelines concepts from Monday

---

# STAGE 1 -- Feature Store Fundamentals (55 min)

> **Goal:** Understand why feature management matters, create a Feature Group for FraudShield fraud detection features, ingest records, and query both the online store (real-time) and offline store (batch/training). Associates see how Feature Store eliminates training-serving skew.

**Exit Criteria Addressed:**
- Explain training-serving skew and how Feature Store prevents it (Required)
- Create a Feature Group with a defined schema (Required)
- Ingest records into a Feature Group (Required)
- Retrieve features from the online store for real-time inference (Required)
- Query the offline store for batch training data (Required)

### Instructor Opening (3 minutes -- talk, no code)

> "Monday you deployed models and invoked them with CSV files you built by hand. That works for a demo, but imagine FraudShield in production: the training pipeline computes features one way, the inference endpoint computes them another way, and six months later nobody remembers which version of the feature logic is correct. This is training-serving skew, and it is one of the most common reasons ML systems fail silently in production. Feature Store solves this by giving you a single source of truth for features that serves both training and real-time inference."

---

## STEP 1 -- Why Feature Management Matters (7 minutes)

**Pacing: conceptual with notebook markdown.**

> "Picture this: your data scientist computes `transaction_count_24h` in a Pandas notebook. Your ML engineer rewrites that logic in a Lambda function for real-time inference. They are slightly different -- one uses a 24-hour rolling window, the other uses calendar day. The model trained on one definition but serves predictions using the other. Accuracy drops 8% and nobody knows why."

Key points:
- Feature engineering is 60-80% of ML work
- Training-serving skew is silent -- metrics look fine in dev, fail in production
- Feature Store provides a single definition, single computation, dual-access pattern
- Online store: low-latency key-value lookups for real-time inference
- Offline store: historical feature values in S3 (Parquet) for training and batch inference

**Discussion Prompt:** "Think about FraudShield. If the fraud model uses `merchant_risk_score` during training but the production system computes it differently, what happens to fraud detection accuracy?"

---

## STEP 2 -- Feature Store Architecture (5 minutes)

**Pacing: notebook markdown with conceptual diagram.**

> "A Feature Group is like a table. Each row is an entity -- in our case, a transaction. Each column is a feature. Every Feature Group has two required special columns: a record identifier and an event time. The online store holds the latest value per record ID. The offline store holds the full history, partitioned in S3 as Parquet files."

Walk through the conceptual diagram:

```
Ingestion --> Feature Group --> Online Store (DynamoDB, latest values)
                            --> Offline Store (S3/Parquet, full history)
                                    |
                                    v
                              Athena (SQL queries for training datasets)
```

---

## STEP 3 -- Create a Feature Group (12 minutes)

**Pacing: live code in notebook. Step through each parameter.**

> "We define a Feature Group called `fraudshield-transaction-features`. The schema mirrors the features we have been using all week: amount, hour, distance_from_home, transaction_count_24h, is_international, merchant_risk_score, plus the target label. We also include `record_id` as the record identifier and `event_time` as the event timestamp."

Walk through the code:
1. Define feature definitions with types (Fractional, Integral, String)
2. Create the Feature Group with both online and offline store enabled
3. Wait for the Feature Group status to become `Created`

> "Notice we enable both the online and offline store. This is the dual-access pattern: the same Feature Group serves real-time lookups and batch training queries."

[PAUSE -- Verify every student's Feature Group reaches `Created` status before proceeding.]

---

## STEP 4 -- Ingest Data into the Feature Group (10 minutes)

**Pacing: live code. Associates run the ingestion cell.**

> "Now we ingest our synthetic fraud data into the Feature Group. The Feature Store SDK provides a `put_record` method for single records and an `ingest` method for bulk ingestion from a DataFrame. We use bulk ingestion because we have 2,000 records."

Key points:
- Each record must include the record identifier and event time
- Ingestion is idempotent -- re-ingesting the same record ID with a newer event time overwrites the online store
- The offline store appends (it keeps history)
- Ingestion runs asynchronously; monitor with the ingestion manager

**Discussion Prompt:** "Why does the online store overwrite while the offline store appends? Think about what each store is used for."

---

## STEP 5 -- Query the Online Store (8 minutes)

**Pacing: live code. This is the "aha" moment for real-time feature serving.**

> "Imagine a transaction just arrived at FraudShield. You need the features for that customer in single-digit milliseconds to decide if the transaction is fraudulent. You call `get_record` with the record ID, and Feature Store returns the latest feature vector. This is exactly what your inference endpoint would do in production -- no CSV files, no feature recomputation."

Walk through:
1. Call `get_record` with a known record ID
2. Inspect the returned feature vector
3. Show that the values match what was ingested

---

## STEP 6 -- Query the Offline Store (10 minutes)

**Pacing: live code with Athena query.**

> "For training, you need historical data -- not just the latest value, but all values over time. The offline store writes Parquet files to S3, and SageMaker creates an Athena table automatically. You query it with standard SQL."

Walk through:
1. Show the Athena query that selects from the offline store table
2. Run the query and retrieve results
3. Explain partitioning by year/month/day/hour

> "This is how you build training datasets in production. You query the offline store for the time period you care about, and you are guaranteed the features match what the online store serves at inference time."

[PAUSE FOR BREAK - 5 MINS]

---

# STAGE 2 -- Data Wrangler, Canvas, and Autopilot (55 min)

> **Goal:** Understand the higher-level SageMaker tools for data preparation and automated ML. Associates learn when to use Data Wrangler for visual data flows, Canvas for no-code ML, and Autopilot for automated model selection and tuning. They run an Autopilot job on fraud data.

**Exit Criteria Addressed:**
- Describe Data Wrangler's role in visual data preparation (Required)
- Explain when to use Canvas for no-code ML (Required)
- Compare Canvas and Autopilot on scope, audience, and control (Required)
- Configure and launch an Autopilot AutoML job (Required)

### Instructor Opening (2 minutes)

> "Feature Store gives you a managed place to store features. But how do you get from raw data to those features in the first place? And what if the person doing that work is not an ML engineer? SageMaker offers three tools at different levels of abstraction: Data Wrangler for visual data preparation, Canvas for no-code ML, and Autopilot for automated model selection. Let us look at each one."

---

## STEP 7 -- Data Wrangler: Visual Data Preparation (12 minutes)

**Pacing: conceptual with notebook markdown. Data Wrangler runs inside Studio -- show screenshots or diagrams rather than live code.**

> "Data Wrangler is a visual interface inside SageMaker Studio for data preparation and feature engineering. You import data from S3, Athena, Redshift, or other sources, then build a flow of transforms: rename columns, handle missing values, encode categoricals, normalize numerics, engineer new features. At the end, you export the flow as a processing job, a Feature Store ingestion pipeline, or a notebook."

Key points:
- Import sources: S3, Athena, Redshift, Snowflake, Databricks
- 300+ built-in transforms (no code required)
- Data quality and insights report (statistics, target leakage, anomaly detection)
- Export options: Processing Job, Pipeline, Feature Store, Notebook

Walk through the conceptual workflow:

```
Raw Data --> Import --> Transform --> Analyze --> Export
  (S3)      (Wrangler)  (visual)    (quality)   (Pipeline / Feature Store / Notebook)
```

> "The key insight is that Data Wrangler is not a separate product -- it generates the same SageMaker Processing Jobs and Feature Store ingestion code that you would write by hand. It is an accelerator, not a replacement."

**Discussion Prompt:** "When would you use Data Wrangler instead of writing Pandas code? When would you not?"

---

## STEP 8 -- Canvas: No-Code Machine Learning (5 minutes)

**Pacing: conceptual with notebook markdown. Canvas runs as a standalone Studio app.**

> "Canvas takes the abstraction one level higher. A business analyst who has never written a line of code can upload a CSV, select a target column, and Canvas builds, trains, and evaluates multiple models automatically. It even generates predictions and a model analysis report."

Key points:
- Target audience: business analysts, domain experts, citizen data scientists
- No code required -- purely visual interface
- Supports tabular (classification, regression), time-series forecasting, image classification, text analysis
- Under the hood: uses Autopilot for model training
- Limitations: less control, limited model customization, no custom containers

> "Think of Canvas as Autopilot with a friendlier interface and fewer knobs. If your stakeholder at FraudShield wants to experiment with different target variables or data subsets without waiting for the ML team, Canvas lets them do that safely."

---

## STEP 8b -- Canvas Walkthrough: Quick Build on FraudShield Data (15 minutes)

**Pacing: instructor-led walkthrough using the step-by-step guide in the notebook. Walk through each step on screen while associates follow along. The Quick Build itself takes 2-15 minutes -- use that wait time for the Canvas vs Autopilot comparison discussion (Step 9).**

> "Let us actually use Canvas. We are going to build a fraud detection model without writing a single line of code. Follow the walkthrough in the notebook."

Walk through the notebook's Canvas walkthrough cell:

1. **Launch Canvas** from the Studio Domain (warn: first launch can take several minutes to provision)
2. **Import the FraudShield CSV** from S3 -- same data they ingested into Feature Store earlier
3. **Create a new model** -- name it `fraudshield-canvas-model`, select Predictive analysis
4. **Select target column** -- `is_fraud` or `target`. Point out that Canvas auto-detects column types
5. **Run Quick Build** -- click the button and note the time estimate

> "While Quick Build trains, look at the notebook's Canvas vs Autopilot comparison. We will come back to the results."

6. **Review model analysis** once Quick Build completes:
   - Overall accuracy
   - Column impact chart -- ask: "How does this compare to the feature importance from Monday's Random Forest?"
   - Class-level metrics if available
7. **Generate predictions** -- single prediction with high-risk values, then low-risk values. Compare confidence scores.
8. **Clean up** -- delete model and dataset from Canvas

> "You just built a fraud detection model in under 15 minutes with zero code. The trade-off is control. You cannot choose the algorithm, customize preprocessing, or plug this into a pipeline. But for a product manager who needs a quick answer to 'does this data have signal?' -- this is the right tool."

**Timing note:** If Canvas provisioning or Quick Build takes longer than expected, move to Step 9 (comparison discussion) during the wait and return to results when ready. Do not let provisioning delays stall the lecture.

---

## STEP 9 -- Canvas vs Autopilot Comparison (8 minutes)

**Pacing: notebook markdown with comparison table.**

> "Here is the real distinction. Canvas and Autopilot solve overlapping problems, but for different audiences with different needs."

| Dimension | Canvas | Autopilot |
|-----------|--------|-----------|
| **Audience** | Business analysts, no-code users | ML engineers, data scientists |
| **Interface** | Visual point-and-click in Studio | Python SDK or Studio UI |
| **Input** | Upload CSV or connect to data source | S3 path to CSV |
| **Model Control** | Minimal -- select target column, Canvas decides the rest | Moderate -- set objective metric, problem type, max candidates |
| **Algorithm Selection** | Automatic (uses Autopilot under the hood) | Automatic with visibility into candidates |
| **Generated Artifacts** | Predictions, model analysis report | Notebooks, model artifacts, tuning logs |
| **Custom Preprocessing** | Built-in transforms only | Custom preprocessing via generated notebooks |
| **Deployment** | One-click deploy from Canvas UI | SDK deploy with full endpoint configuration |
| **Cost Control** | Limited -- runs until done | Max candidates, max runtime, instance type selection |
| **Best For** | Quick experiments, stakeholder self-service | Production AutoML, baseline benchmarking |

**Discussion Prompt:** "A product manager at FraudShield wants to test whether adding `customer_tenure` as a feature improves fraud detection. Should they use Canvas or Autopilot? What about the ML team building the production model?"

---

## STEP 10 -- Autopilot AutoML on Fraud Data (15 minutes)

**Pacing: live code. The Autopilot job itself takes time, so launch it and discuss while it runs.**

> "Autopilot is the programmatic AutoML service. You point it at a dataset in S3, tell it which column is the target, and it automatically tries multiple algorithms, feature engineering strategies, and hyperparameter combinations. It produces a leaderboard of candidates ranked by your chosen metric."

Walk through the code:
1. Upload the fraud dataset to S3 (reuse from Monday's data)
2. Configure the Autopilot job: input data, target column, problem type, objective metric
3. Launch the job
4. While it runs, explain what Autopilot does internally:
   - Analyzes the dataset (data type inference, statistics)
   - Generates candidate pipelines (feature engineering + algorithm)
   - Trains and tunes candidates
   - Ranks by objective metric (F1 for fraud detection)
5. Check job status and review the leaderboard

> "We set F1 as the objective metric because accuracy is misleading for imbalanced fraud data. A model that always predicts 'not fraud' gets 98% accuracy but catches zero fraud. F1 balances precision and recall."

Key API parameters:
- `AutoMLJobName`: unique job identifier
- `InputDataConfig`: S3 path, target column name
- `OutputDataConfig`: where to store artifacts
- `AutoMLJobObjective`: the metric to optimize (F1, Accuracy, AUC, etc.)
- `ProblemType`: BinaryClassification, MulticlassClassification, Regression
- `AutoMLJobConfig.CompletionCriteria`: max candidates, max runtime

> "In production, you would set `MaxCandidates` to control cost. Each candidate is a full training job. For this demo we keep it small."

[PAUSE FOR BREAK - 5 MINS]

---

# STAGE 3 -- Exporting to Pipelines (55 min)

> **Goal:** Connect Feature Store and data engineering to the pipeline infrastructure introduced on Monday. Associates build a conceptual pipeline that reads from Feature Store, trains a model, evaluates it, and registers it in Model Registry. This closes the loop from raw data to deployed model.

**Exit Criteria Addressed:**
- Describe how Feature Store integrates with SageMaker Pipelines (Required)
- Define a pipeline with ProcessingStep, TrainingStep, and RegisterModel (Required)
- Connect pipeline registration to Monday's Model Registry workflow (Required)

### Instructor Opening (3 minutes)

> "On Monday you learned that pipelines are DAGs -- directed acyclic graphs -- where each node is a step. You saw the conceptual flow: Preprocess, Train, Evaluate, Register. Today we make that concrete. The preprocessing step reads from Feature Store instead of a raw CSV. The training step trains a model. The registration step puts it in the Model Registry you set up on Monday. This is how production ML systems work."

---

## STEP 11 -- Feature Store in Pipelines (10 minutes)

**Pacing: conceptual with notebook markdown, then code.**

> "The offline store is the bridge between Feature Store and Pipelines. Your pipeline's first step is a ProcessingStep that queries the offline store via Athena, pulls the training dataset, and writes it to S3. From there, the TrainingStep picks it up. This means every pipeline run uses features computed and managed by Feature Store -- no ad-hoc CSV generation."

Show the updated pipeline diagram:

```
Feature Store (offline) --> ProcessingStep (Athena query --> S3)
                                |
                                v
                          TrainingStep (train model)
                                |
                                v
                          EvaluateStep (compute metrics)
                                |
                                v
                          ConditionStep (F1 >= 0.85?)
                                |
                          Yes --+--> RegisterModel (Model Registry)
                                |
                          No  --+--> Fail / Alert
```

---

## STEP 12 -- Build a Pipeline Definition (20 minutes)

**Pacing: live code, step by step. This is the core hands-on activity for Stage 3.**

> "We define three pipeline steps. The ProcessingStep runs a script that queries the offline store and produces a training CSV. The TrainingStep trains a Random Forest. The RegisterModel step puts the trained model into Model Registry with PendingManualApproval status -- the same registry you used on Monday."

Walk through:
1. Define pipeline parameters (instance type, model approval status)
2. Define a ProcessingStep with a SKLearnProcessor
3. Define a TrainingStep using the same SKLearn estimator from Friday
4. Define a RegisterModel step that targets the Model Package Group from Monday
5. Create the pipeline object linking the steps
6. Upsert (create or update) the pipeline

> "Notice we use `upsert` rather than `create`. If the pipeline already exists, it updates the definition. This is idempotent -- safe to run multiple times."

**Discussion Prompt:** "What happens if the ProcessingStep fails? Does the TrainingStep still run? Why or why not?" (No -- DAG dependencies mean downstream steps only run if upstream steps succeed.)

---

## STEP 13 -- Connect to Monday's Model Registry (10 minutes)

**Pacing: conceptual with notebook markdown, then verify.**

> "On Monday you created the `fraudshield-rf` Model Package Group and manually registered a model version. The RegisterModel pipeline step does exactly the same thing, but automatically. Every time the pipeline runs successfully, a new model version appears in the registry with PendingManualApproval status. Your approval workflow from Monday is the deployment gate."

Key points:
- RegisterModel creates a new Model Package (version) in the existing group
- Approval status is set by the pipeline (PendingManualApproval)
- Human review or automated evaluation promotes to Approved
- The full lifecycle: Feature Store --> Pipeline --> Model Registry --> Endpoint

> "This is the complete MLOps picture. Monday you did each step manually. Now you have the infrastructure to automate it. Tomorrow, we build on this with more advanced pipeline patterns."

---

## STEP 14 -- Cleanup (7 minutes)

**Pacing: live code. EVERY student must complete this.**

> "Delete the Feature Group. If you launched an Autopilot job, let it finish or stop it. Clean up any artifacts created today. Check your billing dashboard."

Delete order:
1. Feature Group (this removes both online and offline store data)
2. Stop Autopilot job if still running
3. Delete any endpoints that were created
4. Verify nothing is left running

> "Make this a habit. Every day ends with cleanup. In production, cleanup is automated. In a learning environment, it is your responsibility."

[PAUSE -- Walk the room and verify every student has cleaned up.]

---

## Wrap-up and Q&A Buffer (10 minutes)

### Summary (4 minutes)

> "Today you accomplished three things. First, you learned Feature Store: you created a Feature Group, ingested fraud data, queried the online store for real-time feature retrieval, and queried the offline store for batch training data. You understand why training-serving skew is dangerous and how Feature Store prevents it. Second, you explored the higher-level SageMaker tools: Data Wrangler for visual data preparation, Canvas for no-code ML, and Autopilot for automated model selection. You know when to use each one and how they compare. Third, you connected Feature Store to Pipelines, building the complete data flow from feature ingestion to model registration. This ties directly to Monday's Model Registry and deployment workflow."

### Wednesday Preview (2 minutes)

> "Wednesday we move into NLP and Transformers. The question shifts from 'how do we manage data and features' to 'how do we process and understand text.' You will learn tokenization, attention mechanisms, and the Transformer architecture that powers modern language models. Read the NLP and Transformer CTs before Wednesday."

### Open Q&A (4 minutes)

---

## Instructor Notes -- Common Issues

| Issue | Resolution |
|-------|-----------|
| Feature Group stuck on "Creating" | Feature Group creation can take 1-3 minutes. Wait and re-check status. If stuck beyond 5 minutes, check IAM permissions for Feature Store. |
| `ResourceInUse` error on Feature Group creation | A Feature Group with the same name already exists. Either delete it first or use a unique name with a timestamp suffix. |
| Ingestion fails with `ValidationException` | Schema mismatch. Verify DataFrame column names and types match the Feature Group definition exactly. Check that `record_id` and `event_time` columns exist. |
| Online store `get_record` returns empty | Ingestion is asynchronous. Wait 30-60 seconds after ingestion completes, then retry. |
| Offline store Athena query returns no results | Offline store data takes 5-15 minutes to appear in the Glue catalog. Run `MSCK REPAIR TABLE` to refresh partitions, then retry. |
| Autopilot job takes too long | Set `MaxCandidates` to 5 and `MaxRuntimePerTrainingJobInSeconds` to 300 for demo purposes. Stop the job if it exceeds 20 minutes. |
| Canvas not available in Studio | Canvas requires SageMaker Studio and may not be enabled in all regions. Use the conceptual material and skip live demo. |
| Data Wrangler instance takes long to start | Data Wrangler launches an `ml.m5.4xlarge` instance by default. It can take 3-5 minutes. This is conceptual today -- no need to launch if time is short. |
| Pipeline `upsert` fails with permission error | The SageMaker execution role needs `sagemaker:CreatePipeline` and `sagemaker:UpdatePipeline` permissions. |
| Student did not clean up Monday's endpoints | Help them delete immediately. Check billing together. Reinforce the cleanup habit. |
