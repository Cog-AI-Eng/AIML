# AdvSageMaker-Algorithms Lecture - Instructor Guide

**Total Duration:** 180 Minutes (3 Stages)
**Consolidated Activities:** XGBoost Architecture, K-Means Clustering, Random Cut Forest, BlazingText & Word2Vec, DeepAR Time Series, HPO Job Anatomy, Optimization Strategies, Parameter Ranges in Pipelines

| Block | Content | Minutes |
|-------|---------|---------|
| Stage 1 | Built-in Algorithms Survey -- XGBoost and K-Means | 45 |
| Break 1 | Stretch / Questions | 10 |
| Stage 2 | Specialized Algorithms -- RCF, BlazingText, DeepAR | 45 |
| Break 2 | Stretch / Questions | 10 |
| Stage 3 | Hyperparameter Optimization at Scale | 45 |
| Buffer | Open Q&A, Wrap-Up | 25 |

---

## Lecture Overview

**Unified Scenario -- FraudShield Risk Analytics (Advanced)**

FraudShield's data foundation is now in place -- the team has a production-ready Studio domain, a Feature Store serving customer features, and a Data Wrangler pipeline transforming raw transaction data. The next challenge is algorithmic: the current fraud detection model is a single logistic regression running in a notebook. Leadership wants a suite of models that address different aspects of the fraud problem: classification of known fraud patterns, clustering to discover new customer segments, anomaly detection for novel attack vectors, text classification for flagging suspicious transaction descriptions, and time-series forecasting for predicting fraud volume.

In this module, Associates explore SageMaker's built-in algorithm library to address each of these use cases. They will train XGBoost and K-Means from the console using Algorithm Mode, then move to specialized algorithms -- Random Cut Forest for anomaly detection, BlazingText for text classification, and DeepAR for time-series forecasting. The module culminates with hyperparameter optimization, where Associates configure multi-trial tuning jobs that systematically search for optimal model configurations.

Throughout the lecture, Associates will contrast Algorithm Mode (container-managed, no code required) with Script Mode (custom training scripts using the algorithm's container as a base). This distinction is critical for FraudShield's team: junior engineers can launch Algorithm Mode jobs from the console, while senior engineers write Script Mode scripts for custom loss functions and evaluation metrics. Both approaches produce the same model artifacts and are deployable through the same pipeline infrastructure.

---

## Pre-Lecture Setup

### Instructor Checklist
- Verify the FraudShield dataset is available in S3 in the required formats:
  - CSV format for XGBoost and general use: `s3://fraudshield-advanced-data/processed/train.csv` and `test.csv` (target column as the first column, no headers, for Algorithm Mode)
  - A customer features CSV for K-Means: `s3://fraudshield-advanced-data/processed/customer_features.csv` (numeric only, no target column)
  - A transaction time series JSON for RCF: `s3://fraudshield-advanced-data/processed/transaction_timeseries.csv`
  - A text classification dataset for BlazingText: `s3://fraudshield-advanced-data/processed/transaction_descriptions.txt` (BlazingText format: `__label__fraud` or `__label__legit` followed by text)
  - A time series JSON Lines file for DeepAR: `s3://fraudshield-advanced-data/processed/daily_fraud_counts.jsonl`
- Pre-create an IAM role for training jobs with S3 read/write access and SageMaker full access
- Verify ml.m5.xlarge instances are available in the target region (check Service Quotas)
- Pre-run at least one XGBoost training job to ensure the container image pulls correctly (first pull can take 2-3 minutes)
- Prepare a completed HPO job (launched 30+ minutes before lecture) so Associates can see finished results during Stage 3
- Have the SageMaker built-in algorithm documentation tabs open for quick reference during Q&A

### Student Prerequisites
- Completed Module 1 (AdvSageMaker-Data) -- Associates should have a running Studio domain and familiarity with Feature Store
- Understanding of classification, clustering, and regression problem types from foundational ML modules
- Basic knowledge of XGBoost (tree ensembles, objective functions) from the Applied ML Foundations skill
- Familiarity with the SageMaker training job lifecycle (training, model artifact upload to S3, deployment)
- Access to the formatted datasets in S3 (instructor should share the bucket path before the lecture)

---

## Stage 1: Built-in Algorithms Survey -- XGBoost and K-Means
**Duration:** 45 minutes
**Exit Criteria Addressed:**
- Launch an XGBoost training job using Algorithm Mode from the SageMaker console
- Distinguish between Algorithm Mode and Script Mode for built-in algorithms
- Configure a K-Means clustering job and interpret cluster assignments
- Select appropriate instance types for training within Free Tier constraints

### Instructor Opening (5 minutes)

> "SageMaker ships with over 17 built-in algorithms, each packaged in an optimized container that AWS maintains. You do not install XGBoost with pip. You do not write a Dockerfile. You point the service at your data, set hyperparameters, and it runs. Today we start with the two algorithms FraudShield needs most urgently: XGBoost for classifying transactions as fraudulent or legitimate, and K-Means for discovering customer segments that the fraud team has not considered. We will launch both from the console first, then discuss when and why you would switch to Script Mode."

### STEP 1 -- XGBoost Training Job via Console (Algorithm Mode) (15 minutes)

**Console Navigation:**
1. Navigate to **Amazon SageMaker** > **Training** > **Training jobs**
2. Click **Create training job**

3. Configure the job:
   - **Job name:** `fraudshield-xgb-algorithm-mode`
   - **IAM role:** select the pre-created training role
   - **Algorithm source:** select **SageMaker built-in algorithm**
   - **Algorithm:** scroll to and select **XGBoost**
   - **Algorithm version:** select the latest (e.g., 1.5-1 or later)

**Instructor Note:** Pause here. Explain Algorithm Mode vs. Script Mode:
- **Algorithm Mode:** You select the algorithm from the dropdown, set hyperparameters in the console, and provide data in the algorithm's expected format (CSV with target as column 0, no headers). No training script required.
- **Script Mode:** You provide a custom `train.py` script and use the XGBoost container as the base image. This gives you control over data loading, custom metrics, custom evaluation logic, and even custom loss functions.

4. **Resource configuration:**
   - Instance type: `ml.m5.xlarge` (1 instance)
   - Volume size: 10 GB
   - Max runtime: 3600 seconds

**Teaching Tip:** Explain the instance selection rationale: "ml.m5.xlarge gives us 4 vCPUs and 16 GB RAM. For a dataset of 4000 rows, this is more than sufficient. In production with millions of rows, you would scale up to ml.m5.4xlarge or use distributed training with multiple instances."

5. **Hyperparameters:**
   - `objective`: `binary:logistic`
   - `num_round`: `100`
   - `max_depth`: `5`
   - `eta`: `0.2`
   - `eval_metric`: `auc`
   - `scale_pos_weight`: `10` (explain: this accounts for class imbalance in fraud detection)

**Pacing Guidance:** For each hyperparameter, pause and explain its purpose. Ask Associates to predict what happens if `max_depth` is set to 15 (overfitting) or `eta` is set to 1.0 (aggressive learning, unstable convergence).

6. **Input data configuration:**
   - Channel name: `train`
   - S3 location: `s3://fraudshield-advanced-data/processed/train.csv`
   - Content type: `text/csv`
   - Channel name: `validation`
   - S3 location: `s3://fraudshield-advanced-data/processed/test.csv`
   - Content type: `text/csv`

7. **Output data configuration:**
   - S3 output path: `s3://fraudshield-advanced-data/models/`

8. Click **Create training job**

**Instructor Note:** The job will run for 3-5 minutes. While it runs, navigate to the job details page and show:
- The **Monitor** section with CloudWatch metrics (train:auc, validation:auc)
- The **Logs** link to CloudWatch Logs
- The status transitions: Creating > Training > Uploading > Completed

**Q&A Pause (2 minutes):** "We just trained XGBoost without writing a single line of code. When would FraudShield need to switch to Script Mode?" Guide answers toward: custom evaluation metrics (e.g., cost-sensitive fraud loss), custom data augmentation, integration with Feature Store for data loading, or use of additional XGBoost callbacks.

### STEP 2 -- K-Means Clustering from the Console (13 minutes)

**Console Navigation:**
1. Navigate to **Training** > **Training jobs** > **Create training job**
2. Configure:
   - **Job name:** `fraudshield-kmeans-segmentation`
   - **Algorithm:** select **K-Means** from the built-in list
   - **Instance type:** `ml.m5.xlarge`

3. **Hyperparameters:**
   - `k`: `5` (number of clusters)
   - `feature_dim`: set to the number of numeric features in the customer features dataset
   - `mini_batch_size`: `500`
   - `init_method`: `kmeans++`

**Teaching Tip:** Discuss the choice of `k=5`: "For FraudShield, we hypothesize that customers fall into roughly five behavioral segments -- high-value frequent shoppers, occasional buyers, new accounts, dormant accounts, and suspicious-pattern accounts. In practice, you would use the elbow method or silhouette analysis to determine k. We will revisit this in HPO where we can tune k systematically."

4. **Input data:**
   - Channel: `train`
   - S3 location: `s3://fraudshield-advanced-data/processed/customer_features.csv`
   - Content type: `text/csv`

5. Launch the job

**Instructor Note:** While the K-Means job runs (2-4 minutes), explain how K-Means output works in SageMaker:
- The model artifact contains the cluster centroids
- When you deploy the model and send inference requests, it returns the closest cluster index and the distance to that centroid
- FraudShield can use cluster assignments as an additional feature in the XGBoost model (cluster-based fraud rate varies significantly)

6. Once complete, show the model artifact in S3:
   - Navigate to the output path
   - Show the `model.tar.gz` file
   - Explain that this contains the serialized cluster centroids

### STEP 3 -- Algorithm Mode vs. Script Mode Comparison (5 minutes)

Present or narrate this comparison:

| Aspect | Algorithm Mode | Script Mode |
|--------|---------------|-------------|
| Setup | Console or SDK, no code | Requires a `train.py` script |
| Data format | Algorithm-specific (CSV column 0 = target) | Flexible (you control data loading) |
| Hyperparameters | Set via console or API | Set via console/API + accessible in script |
| Custom metrics | Limited to built-in metrics | Full control -- log any metric |
| Custom preprocessing | None (data must be pre-formatted) | Full control in the training script |
| Container | AWS-managed, optimized | Same AWS container, your script injected |
| Use case | Rapid prototyping, standard workflows | Production, custom requirements |

**Teaching Tip:** Show a minimal Script Mode training script for XGBoost to illustrate the difference:

```python
import argparse
import os
import pandas as pd
import xgboost as xgb

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_depth", type=int, default=5)
    parser.add_argument("--eta", type=float, default=0.2)
    parser.add_argument("--num_round", type=int, default=100)
    args = parser.parse_args()

    train_path = os.path.join(os.environ["SM_CHANNEL_TRAIN"], "train.csv")
    df = pd.read_csv(train_path, header=None)
    y = df.iloc[:, 0]
    X = df.iloc[:, 1:]

    dtrain = xgb.DMatrix(X, label=y)
    params = {
        "objective": "binary:logistic",
        "max_depth": args.max_depth,
        "eta": args.eta,
        "eval_metric": "auc",
    }
    model = xgb.train(params, dtrain, num_boost_round=args.num_round)

    model_dir = os.environ["SM_MODEL_DIR"]
    model.save_model(os.path.join(model_dir, "xgboost-model"))
```

Explain: "The environment variables (`SM_CHANNEL_TRAIN`, `SM_MODEL_DIR`) are injected by SageMaker. Your script reads from them and writes the model artifact. Everything else -- container management, instance provisioning, S3 upload -- is handled by the platform."

---

[PAUSE FOR BREAK -- 10 minutes]

---

## Stage 2: Specialized Algorithms -- RCF, BlazingText, DeepAR
**Duration:** 45 minutes
**Exit Criteria Addressed:**
- Train a Random Cut Forest model for anomaly detection
- Configure BlazingText for text classification with supervised mode
- Set up a DeepAR forecasting model with categorical covariates
- Select the appropriate built-in algorithm for a given business problem

### Instructor Opening (3 minutes)

> "XGBoost and K-Means cover classification and clustering, but FraudShield's threat landscape is broader. What about transactions that do not match any known fraud pattern? That is anomaly detection -- Random Cut Forest. What about flagging transaction descriptions like 'wire transfer to offshore account'? That is text classification -- BlazingText. What about predicting next week's fraud volume so the ops team can staff appropriately? That is time-series forecasting -- DeepAR. Each of these is a purpose-built algorithm optimized for its specific problem type."

### STEP 1 -- Random Cut Forest for Anomaly Detection (15 minutes)

**Concept Introduction (3 minutes):**
Explain Random Cut Forest (RCF) at a high level:
- RCF is an unsupervised algorithm that assigns an anomaly score to each data point
- It works by building an ensemble of random decision trees. Points that require fewer cuts to isolate receive higher anomaly scores.
- SageMaker's RCF is optimized for streaming data -- it can update incrementally as new data arrives
- For FraudShield: RCF detects transactions that are statistically unusual compared to the normal pattern, even if they do not match known fraud signatures

**Console Navigation:**
1. Navigate to **Training** > **Training jobs** > **Create training job**
2. Configure:
   - **Job name:** `fraudshield-rcf-anomaly`
   - **Algorithm:** select **Random Cut Forest**
   - **Instance type:** `ml.m5.xlarge`

3. **Hyperparameters:**
   - `num_trees`: `100` (more trees = more stable scores, but longer training)
   - `num_samples_per_tree`: `256`
   - `feature_dim`: set to the number of features in the time series dataset

4. **Input data:**
   - Channel: `train`
   - S3 location: `s3://fraudshield-advanced-data/processed/transaction_timeseries.csv`
   - Content type: `text/csv`

5. Launch the training job

**Instructor Note:** While the job trains (2-3 minutes), discuss the inference workflow:
- RCF returns an anomaly score (not a binary label)
- FraudShield must define a threshold: transactions with scores above the threshold are flagged for review
- The threshold is a business decision, not a model parameter. Show how to determine it:
  - Score all historical data
  - Pick the threshold at the 99th percentile of scores
  - Monitor the false positive rate and adjust

**Teaching Tip:** Draw a number line on screen: normal transactions cluster at low scores (1-3), while anomalous ones spike to 8-12+. The threshold sits between these clusters.

**Q&A Pause (2 minutes):** "How does RCF differ from the XGBoost fraud classifier we just built?" Key answer: XGBoost requires labeled data and detects known patterns. RCF is unsupervised and detects unknown anomalies. FraudShield should use both in parallel.

### STEP 2 -- BlazingText for Text Classification (13 minutes)

**Concept Introduction (3 minutes):**
- BlazingText implements the Word2Vec algorithm (for unsupervised word embeddings) and a text classification algorithm (supervised, based on FastText)
- For FraudShield: classify transaction descriptions or merchant names into categories like "high_risk" or "normal"
- BlazingText is extremely fast -- it can train on millions of text records in minutes
- Input format: one document per line, prefixed with `__label__labelname`

**Console Navigation:**
1. Create a new training job:
   - **Job name:** `fraudshield-blazingtext-classify`
   - **Algorithm:** select **BlazingText**
   - **Instance type:** `ml.m5.xlarge`

2. **Hyperparameters:**
   - `mode`: `supervised` (not `skipgram` or `cbow` -- those are for word embeddings)
   - `epochs`: `10`
   - `learning_rate`: `0.05`
   - `word_ngrams`: `2` (captures bigrams like "wire transfer")
   - `vector_dim`: `100`
   - `min_count`: `2` (ignore words appearing less than twice)

**Instructor Note:** Explain each hyperparameter choice:
- `word_ngrams=2` is important for fraud detection because phrases like "offshore account" or "gift card" carry more signal than individual words
- `min_count=2` eliminates noise from typos or one-off merchant names

3. **Input data:**
   - Channel: `train`
   - S3 location: `s3://fraudshield-advanced-data/processed/transaction_descriptions.txt`
   - Content type: `text/plain`

4. Launch the job

**Teaching Tip:** Show a few sample lines from the input file:
```
__label__fraud wire transfer to offshore account urgent
__label__legit grocery store purchase weekly shopping
__label__fraud gift card bulk purchase reseller
__label__legit monthly subscription streaming service
```

Explain: "Each line is a training example. The `__label__` prefix tells BlazingText which class the text belongs to. At inference time, you send raw text and get back the predicted label with a confidence score."

5. Once training completes, discuss the inference output format:
   - BlazingText returns a JSON array of labels and probabilities
   - Example: `{"label": ["__label__fraud"], "prob": [0.87]}`

### STEP 3 -- DeepAR Time Series Forecasting (11 minutes)

**Concept Introduction (3 minutes):**
- DeepAR is an autoregressive RNN-based algorithm for probabilistic time-series forecasting
- Unlike classical methods (ARIMA), DeepAR learns across multiple related time series simultaneously
- For FraudShield: predict daily fraud counts per transaction category, accounting for seasonality and trends
- DeepAR outputs quantile forecasts (P10, P50, P90), giving FraudShield confidence intervals for staffing decisions

**Console Navigation:**
1. Create a new training job:
   - **Job name:** `fraudshield-deepar-forecast`
   - **Algorithm:** select **DeepAR Forecasting**
   - **Instance type:** `ml.m5.xlarge`

2. **Hyperparameters:**
   - `time_freq`: `D` (daily granularity)
   - `prediction_length`: `14` (forecast 14 days ahead)
   - `context_length`: `28` (use 28 days of history as context)
   - `epochs`: `50`
   - `num_cells`: `40`
   - `num_layers`: `2`
   - `mini_batch_size`: `32`

**Instructor Note:** Explain the key design decisions:
- `prediction_length` should match the business planning horizon. FraudShield's ops team plans two weeks ahead.
- `context_length` should be at least 1-2x the prediction length. More context helps capture weekly patterns.
- `num_cells` and `num_layers` control model complexity. For a dataset with a few hundred time steps, 40 cells and 2 layers is sufficient.

3. **Input data format:** Show the JSON Lines format required by DeepAR:
```json
{"start": "2024-01-01", "target": [12, 15, 8, 22, 30, 18, 10, ...], "cat": [0]}
{"start": "2024-01-01", "target": [5, 3, 7, 2, 8, 4, 6, ...], "cat": [1]}
```

Explain each field:
- `start`: the timestamp of the first observation
- `target`: the array of time-series values
- `cat`: categorical covariates (e.g., transaction category index). DeepAR learns category-specific patterns while sharing information across categories.

4. Launch the job

**Teaching Tip:** Discuss when FraudShield would choose DeepAR over simpler approaches:
- ARIMA: works well for a single time series with clear trend/seasonality. Does not learn across multiple series.
- Prophet: good for single series with holidays. Does not handle categorical covariates natively.
- DeepAR: excels when you have many related time series (fraud counts by category, by region, by payment method) and want to learn shared patterns.

**Q&A Pause (2 minutes):** Recap all five algorithms covered so far and their FraudShield use cases. Ask Associates to match each algorithm to a business question:
1. "Is this transaction fraudulent?" -- XGBoost
2. "Which customer segment does this user belong to?" -- K-Means
3. "Is this transaction abnormally different from the norm?" -- RCF
4. "Does this transaction description look suspicious?" -- BlazingText
5. "How many fraud cases should we expect next week?" -- DeepAR

---

[PAUSE FOR BREAK -- 10 minutes]

---

## Stage 3: Hyperparameter Optimization at Scale
**Duration:** 45 minutes
**Exit Criteria Addressed:**
- Configure an HPO tuning job from the SageMaker console
- Distinguish between Bayesian, Random, and Grid search strategies
- Set parameter ranges (continuous, integer, categorical) for tuning
- Monitor HPO job progress and identify the best trial
- Describe how to embed an HPO step in a SageMaker Pipeline

### Instructor Opening (3 minutes)

> "We trained XGBoost with max_depth=5 and eta=0.2. Were those the best values? We have no idea -- we picked reasonable defaults. Hyperparameter optimization removes the guesswork. Instead of running one training job, you run 20 or 50, each with different hyperparameter combinations chosen by an optimization algorithm. SageMaker manages the infrastructure: it spins up instances, runs the trials, tracks the results, and identifies the winner. Let us set up a tuning job for FraudShield's XGBoost model."

### STEP 1 -- Anatomy of an HPO Job (7 minutes)

**Conceptual Overview:**
Before going to the console, explain the four components of an HPO job:

1. **Objective Metric:** The metric to optimize (e.g., `validation:auc`). Must be logged by the training algorithm.
2. **Parameter Ranges:** The hyperparameters to tune and their valid ranges. Three types:
   - **Continuous:** float values in a range (e.g., `eta` between 0.01 and 0.5)
   - **Integer:** integer values in a range (e.g., `max_depth` between 3 and 10)
   - **Categorical:** discrete choices (e.g., `objective` in [`binary:logistic`, `binary:hinge`])
3. **Strategy:** How the optimizer selects hyperparameter combinations:
   - **Bayesian:** builds a probabilistic model of the objective function, selects points likely to improve. Best for < 50 trials.
   - **Random:** samples uniformly from the parameter space. Good baseline, scales well.
   - **Grid:** exhaustive search over a predefined grid. Only feasible for small parameter spaces.
4. **Resource Limits:** Maximum number of training jobs (trials), maximum parallel jobs, and per-job runtime limits.

**Teaching Tip:** Use an analogy: "Bayesian optimization is like an experienced chef adjusting a recipe. After each attempt, they update their mental model of what works. Random search is like trying random recipes. Grid search is like systematically trying every combination in a cookbook. For FraudShield, Bayesian is usually the best choice because each training job costs money and time."

### STEP 2 -- Creating an HPO Job from the Console (15 minutes)

**Console Navigation:**
1. Navigate to **Amazon SageMaker** > **Training** > **Hyperparameter tuning jobs**
2. Click **Create hyperparameter tuning job**

3. **Job configuration:**
   - **Job name:** `fraudshield-xgb-hpo`
   - **Strategy:** select **Bayesian**
   - **Objective metric:** `validation:auc`
   - **Objective type:** Maximize

4. **Training job definition:**
   - **Algorithm:** XGBoost (built-in)
   - **Instance type:** `ml.m5.xlarge`
   - **Instance count:** 1
   - **Input data:** same train/validation channels as Stage 1

5. **Hyperparameter ranges:**

   Configure the following ranges:

   | Parameter | Type | Range |
   |-----------|------|-------|
   | `max_depth` | Integer | 3 -- 10 |
   | `eta` | Continuous | 0.01 -- 0.5 |
   | `min_child_weight` | Continuous | 1 -- 10 |
   | `subsample` | Continuous | 0.5 -- 1.0 |
   | `colsample_bytree` | Continuous | 0.5 -- 1.0 |
   | `num_round` | Integer | 50 -- 300 |
   | `scale_pos_weight` | Continuous | 1 -- 20 |
   | `gamma` | Continuous | 0 -- 5 |

   **Static hyperparameters** (not tuned):
   - `objective`: `binary:logistic`
   - `eval_metric`: `auc`

**Instructor Note:** Walk through each parameter range deliberately. For each one, explain:
- Why it matters for fraud detection (e.g., `scale_pos_weight` compensates for class imbalance)
- Why the range was chosen (e.g., `max_depth` above 10 risks overfitting on 4000 rows)

6. **Resource limits:**
   - **Max training jobs:** `20`
   - **Max parallel training jobs:** `4`

**Teaching Tip:** Explain the parallelism tradeoff: "With Bayesian optimization, each completed trial informs the next choice. Running 4 in parallel means the optimizer has fewer completed results to learn from at each decision point. If you set parallel to 1, Bayesian is most effective but slowest. Setting it to 20 (all trials in parallel) makes Bayesian equivalent to random search. Four parallel jobs is a practical balance for FraudShield."

7. Review the configuration and click **Create hyperparameter tuning job**

### STEP 3 -- Monitoring the HPO Job (8 minutes)

**Console Navigation:**
1. Navigate to the tuning job details page
2. Show the **Training jobs** tab:
   - Each row is a trial with its hyperparameter values, objective metric, and status
   - Sort by the objective metric to identify the current best trial
   - Show the status column: InProgress, Completed, Failed, Stopped

3. Show the **Best training job** section at the top:
   - Displays the best objective metric value found so far
   - Links to the training job details for the best trial

**Instructor Note:** If using the pre-run HPO job (launched before the lecture), show the completed results. If the live job has not finished, show the first few completed trials and discuss the progression.

4. Click into one completed trial to show:
   - The specific hyperparameter values chosen by the Bayesian optimizer
   - The training and validation metric curves in CloudWatch
   - The model artifact location in S3

**Teaching Tip:** Point out how the Bayesian optimizer concentrates trials in promising regions: "Look at the first 5 trials -- the eta values are spread across the full range (0.01 to 0.5). By trials 15-20, the optimizer has narrowed in on a region around 0.1-0.2 because it learned that this range produces higher AUC values."

### STEP 4 -- Random vs. Bayesian Convergence (5 minutes)

**Discussion-Based Section:**

Present or narrate the convergence comparison:

| Aspect | Random Search | Bayesian Search |
|--------|--------------|----------------|
| Trial 5 best AUC | ~0.85 (lucky) or ~0.78 (unlucky) | ~0.83 (informed start) |
| Trial 10 best AUC | ~0.87 | ~0.89 |
| Trial 20 best AUC | ~0.89 | ~0.92 |
| Parallelism impact | None (trials are independent) | Reduces effectiveness |
| Cold start | None | First 5 trials are quasi-random |
| Best for | Large parameter spaces, high parallelism | Moderate parameter spaces, limited budget |

**Instructor Note:** If you have results from both a Random and Bayesian job on the same dataset, show them side by side. Otherwise, present these as typical patterns from AWS documentation and internal benchmarks.

**Q&A Pause (2 minutes):** "When would FraudShield choose Random over Bayesian?" Answer: when running on a very large parameter space (10+ parameters), when budget allows 100+ trials, or when running with very high parallelism where Bayesian's sequential advantage is negated.

### STEP 5 -- HPO as a Pipeline Tuning Step (5 minutes)

**Conceptual Overview:**
Explain how HPO fits into a SageMaker Pipeline:

1. The `TuningStep` in SageMaker Pipelines wraps a tuning job
2. It accepts the same parameter ranges, strategy, and resource limits as a standalone tuning job
3. The output is the best model artifact, which feeds into subsequent pipeline steps (evaluation, registration, deployment)

Show the SDK code structure (do not run it -- this is a preview for the Pipelines module):

```python
from sagemaker.tuner import HyperparameterTuner, ContinuousParameter, IntegerParameter
from sagemaker.workflow.steps import TuningStep

xgb_tuner = HyperparameterTuner(
    estimator=xgb_estimator,
    objective_metric_name="validation:auc",
    objective_type="Maximize",
    hyperparameter_ranges={
        "max_depth": IntegerParameter(3, 10),
        "eta": ContinuousParameter(0.01, 0.5),
        "min_child_weight": ContinuousParameter(1, 10),
        "subsample": ContinuousParameter(0.5, 1.0),
    },
    max_jobs=20,
    max_parallel_jobs=4,
    strategy="Bayesian",
)

tuning_step = TuningStep(
    name="FraudShieldHPO",
    step_args=xgb_tuner.fit(inputs={"train": train_input, "validation": val_input}),
)
```

**Teaching Tip:** Highlight that the Pipeline version uses the exact same tuning configuration as the console job. The difference is automation: the Pipeline runs this step automatically whenever new data arrives, ensuring FraudShield's model is always optimized on the latest data.

### STEP 6 -- Cleanup (2 minutes)

**This step is mandatory. Walk Associates through each cleanup action:**

1. **Training jobs:** Training jobs stop automatically when complete. No ongoing charges. Verify all jobs show "Completed" or "Stopped" status.
2. **HPO job:** The tuning job stops when all trials are complete. No ongoing charges after completion.
3. **Model artifacts in S3:** Each training job and HPO trial produces a `model.tar.gz` in S3. For exercises that will not be used in subsequent modules, delete these:
   - Navigate to `s3://fraudshield-advanced-data/models/` and remove the exercise artifacts
4. **Notebook instances:** If Associates opened Studio notebooks during the lecture, verify the kernel is shut down:
   - Studio > Running Terminals and Kernels > Shut down all
5. **No endpoints were deployed** in this module (deployment is covered in later modules). Confirm with Associates that no one accidentally deployed a model to a real-time endpoint.

**Instructor Note:** Training jobs are one-time costs. The primary concern is orphaned notebook instances and any endpoints that Associates may have created by exploring on their own. Walk the room (or ask in chat) to confirm cleanup is complete.

---

## Post-Lecture Wrap-Up

### Key Takeaways (5 minutes)

1. **Built-in algorithms** are the fastest path from data to trained model on SageMaker. Algorithm Mode requires zero code; Script Mode provides full customization while retaining the managed infrastructure benefits.

2. **Algorithm selection** should be driven by the business problem, not familiarity. FraudShield uses five different algorithms because fraud detection is a multi-faceted problem: classification (XGBoost), segmentation (K-Means), anomaly detection (RCF), text analysis (BlazingText), and forecasting (DeepAR).

3. **Hyperparameter optimization** transforms model training from guesswork into systematic search. Bayesian optimization is the default choice for budget-conscious teams. Always set resource limits (max jobs, max runtime) to control costs.

4. **The console-to-code progression** is intentional: understand the configuration in the console, then automate it in Pipelines. Every console setting maps to an SDK parameter.

### What Comes Next

In the next module (AdvSageMaker-Tracking), Associates will learn to track and compare the training jobs and HPO trials they created today. They will create SageMaker Experiments, associate training runs, compare metrics across experiments, and build lineage graphs that trace a deployed model back to its training data and hyperparameters. The HPO results from today's Stage 3 will serve as the input for experiment comparison exercises.

### Common Pitfalls to Reinforce
- Using the wrong data format for Algorithm Mode (headers, wrong column order) -- this produces cryptic container errors
- Setting HPO parallelism too high, which reduces Bayesian optimization effectiveness
- Forgetting to set `scale_pos_weight` for imbalanced classification (XGBoost defaults assume balanced classes)
- Choosing an instance type larger than needed for small datasets (wastes money, provides no speed improvement)
- Not checking that the objective metric name matches exactly what the algorithm logs (typos cause the HPO job to fail)
