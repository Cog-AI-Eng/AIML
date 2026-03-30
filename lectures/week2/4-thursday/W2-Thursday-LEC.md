# Week 2 Thursday -- Advanced SageMaker Built-in Algorithms

**Total Duration:** 185 Minutes (3 Stages)
**Consolidated Activities:**
- SM Built-in Algorithms: XGBoost for Tabular Data, K-Means Clustering, Random Cut Forest Anomaly Detection
- SM Conceptual Algorithms: BlazingText / Word2Vec (NLP), DeepAR (Time Series)
- SM Tuning: Hyperparameter Optimization Jobs, Bayesian & Random Strategies, Parameter Ranges

| Block | Content | Minutes |
|-------|---------|---------|
| Stage 1 | XGBoost for Fraud Detection | 55 |
| Break 1 | Stretch / Questions | 5 |
| Stage 2 | Other Built-in Algorithms | 55 |
| Break 2 | Stretch / Questions | 5 |
| Stage 3 | Hyperparameter Optimization | 55 |
| Buffer | Open Q&A, Summary, Friday Preview | 10 |

---

## Lecture Overview

**Unified Scenario -- FraudShield Risk Analytics**

Monday used Script Mode to train a Random Forest. Today we use SageMaker's built-in algorithms -- managed, optimized implementations where you supply data and hyperparameters only. XGBoost typically outperforms Random Forest on tabular data. We also explore unsupervised algorithms and then systematically tune hyperparameters with HPO.

1. **"What if there is a stronger algorithm for our fraud data?"** (XGBoost as a built-in algorithm, compare to Monday's Random Forest)
2. **"Can we find structure in fraud data without labels?"** (K-Means clustering, Random Cut Forest anomaly detection)
3. **"What about text and time series?"** (BlazingText, DeepAR -- conceptual overview)
4. **"How do we systematically find the best hyperparameters?"** (HPO jobs, Bayesian optimization, connecting to Wednesday's experiment tracking)

Each stage builds on the FraudShield scenario: Stage 1 replaces the RF with a stronger tabular model, Stage 2 adds unsupervised perspectives, and Stage 3 automates the search for optimal settings.

---

## Pre-Lecture Setup

### Instructor Checklist

- [ ] Monday's fraud data confirmed in S3 (`train.csv`, `validation.csv`)
- [ ] SageMaker execution role ARN ready
- [ ] Companion lecture notebook (`W2-Thursday-notebook.ipynb`) open and tested
- [ ] AWS account with SageMaker access verified
- [ ] XGBoost built-in container image confirmed for target region
- [ ] Budget verified -- HPO launches multiple training jobs
- [ ] Wednesday's experiment results available for reference
- [ ] This instructor guide open in a second tab

### Student Prerequisites

- [ ] Completed readings: XGBoost Architecture CT, K-Means Clustering CT, Random Cut Forest CT, BlazingText & Word2Vec CT, DeepAR Time Series CT, HPO Job Anatomy CT, Optimization Strategies CT, Parameter Ranges in Pipelines CT
- [ ] Monday's notebook completed (fraud data in S3, RF model trained and evaluated)
- [ ] Wednesday's experiments notebook completed (experiment tracking concepts understood)
- [ ] AWS credentials configured, SageMaker SDK installed

---

# STAGE 1 -- XGBoost for Fraud Detection (55 min)

> **Goal:** Train an XGBoost model on the same fraud data from Monday using SageMaker's built-in algorithm, then evaluate and compare it to the Random Forest. Associates experience the built-in algorithm workflow: supply data and hyperparameters, SageMaker handles everything else.

**Exit Criteria Addressed:**
- Configure and train an XGBoost model using the SageMaker built-in algorithm container (Required)
- Prepare tabular data in the XGBoost built-in format: target column first, no header (Required)
- Compare Script Mode (Random Forest) versus built-in algorithm (XGBoost) workflows (Required)
- Evaluate model predictions with precision, recall, F1, and confusion matrix (Required)

### Instructor Opening (3 minutes -- talk, no code)

> "Monday you trained a Random Forest using Script Mode -- you wrote `train.py`, packaged your own code, and SageMaker ran it. Today we flip the model. SageMaker's built-in XGBoost is a managed, optimized implementation of gradient boosted trees. You do not write training code. You supply data in the right format, set hyperparameters, and call `.fit()`. The infrastructure, the algorithm, and the distributed training are all managed for you."

---

## STEP 1 -- Setup and Data Preparation (10 minutes)

**Pacing: live code in notebook.** Associates run the setup cells and data preparation cells.

Narrate: "We reconnect to our SageMaker session and prepare the fraud data in XGBoost's required format. Built-in XGBoost expects CSV with the target column first and no header row. This is different from scikit-learn, where target is typically the last column."

Key points:
- XGBoost built-in format: target column first, no header, no index
- This is a contract with the container -- violating it produces silent errors
- Upload train and validation sets to separate S3 channels

[PAUSE -- Verify every student has data uploaded to S3 before proceeding.]

---

## STEP 2 -- XGBoost Built-in vs Script Mode (8 minutes)

**Pacing: conceptual with notebook markdown, then code.**

> "There are two ways to use XGBoost on SageMaker. Built-in: you use `image_uris.retrieve('xgboost', region, version)` and configure hyperparameters via `.set_hyperparameters()`. Script Mode: you write your own training script and import xgboost yourself. Built-in is simpler when you need standard XGBoost. Script Mode gives you full control when you need custom preprocessing or custom loss functions."

| Aspect | Script Mode (Monday's RF) | Built-in (Today's XGBoost) |
|--------|---------------------------|---------------------------|
| Training code | You write `train.py` | None -- managed by container |
| Model framework | Any (sklearn, PyTorch, etc.) | XGBoost only |
| Data format | Flexible (you parse it) | Target first, no header |
| Hyperparameters | Argparse in your script | `.set_hyperparameters()` |
| Optimization | Your responsibility | AWS-optimized, distributed |

---

## STEP 3 -- Configure and Train XGBoost (15 minutes)

**Pacing: live code, step by step.**

> "We configure the XGBoost estimator. Note the pattern: retrieve the container image URI, create an Estimator, set hyperparameters, and call `.fit()` with S3 input channels. No training script at all."

Walk through each hyperparameter:
- `max_depth=5`: tree depth controls complexity
- `eta=0.2`: learning rate -- how much each tree contributes
- `num_round=100`: number of boosting rounds
- `objective="binary:logistic"`: binary classification with probability output
- `eval_metric="auc"`: area under ROC curve

While training runs (3-5 minutes), discuss:
- How gradient boosting works: each tree corrects the errors of the ensemble so far
- Why XGBoost often outperforms Random Forest: sequential correction vs independent trees
- The built-in container handles distributed training automatically for large datasets

---

## STEP 4 -- Deploy and Evaluate (12 minutes)

**Pacing: live code. This is the comparison moment.**

> "Now we deploy the XGBoost model and evaluate it on the same validation data we used for Monday's Random Forest. This gives us a fair comparison."

Walk through:
1. Deploy XGBoost model to a real-time endpoint
2. Send validation data through the endpoint
3. Compute metrics: precision, recall, F1, AUC
4. Generate confusion matrix
5. Compare to Monday's RF metrics side by side

> "Look at the metrics. XGBoost will typically show higher recall and AUC than Random Forest on this data. Gradient boosting focuses each new tree on the examples the ensemble currently gets wrong -- it actively hunts for the hard cases that Random Forest misses."

**Discussion Prompt:** "We went from Random Forest to XGBoost and saw a performance improvement. But we only tried one set of hyperparameters. How do we know `max_depth=5` is the right choice? We will answer that in Stage 3."

---

## STEP 5 -- Cleanup XGBoost Endpoint (5 minutes)

**Pacing: live code. Mandatory.**

> "Delete the endpoint immediately. We will create a new endpoint during HPO in Stage 3, so clean up now to avoid overlapping costs."

Delete order: endpoint -> endpoint config -> model.

[PAUSE FOR BREAK - 5 MINS]

---

# STAGE 2 -- Other Built-in Algorithms (55 min)

> **Goal:** Explore SageMaker's unsupervised built-in algorithms (K-Means, Random Cut Forest) on the fraud data, and conceptually survey NLP (BlazingText) and time series (DeepAR) algorithms. Associates see that built-in algorithms extend beyond supervised classification.

**Exit Criteria Addressed:**
- Configure and train a K-Means clustering model to find natural groupings in data (Required)
- Configure and train a Random Cut Forest model for anomaly detection (Required)
- Describe BlazingText and Word2Vec as SageMaker built-in NLP algorithms (Required)
- Describe DeepAR as a SageMaker built-in time series forecasting algorithm (Required)

### Instructor Opening (2 minutes)

> "Stage 1 was supervised -- we had labels and trained a classifier. But what if you do not have labels? What if you want to discover structure in the data or flag outliers automatically? SageMaker's unsupervised built-in algorithms handle exactly that."

---

## STEP 6 -- K-Means Clustering Overview (5 minutes)

**Pacing: conceptual with notebook markdown.**

> "K-Means partitions data into k clusters by minimizing the distance from each point to its cluster center. For fraud detection, clusters might reveal natural transaction segments: small retail purchases, large international transfers, suspicious late-night transactions. The algorithm does not know about fraud -- it finds groupings based on the feature values alone."

Key points:
- Unsupervised: no target column
- Choose k (number of clusters) based on domain knowledge or elbow method
- SageMaker's built-in K-Means is optimized for large datasets and supports GPU instances

---

## STEP 7 -- Train K-Means on Fraud Features (12 minutes)

**Pacing: live code.**

> "We train K-Means on the fraud features -- amount, hour, distance, transaction count, is_international, merchant risk score. We omit the target column because this is unsupervised. We will see whether the clusters correlate with fraud labels after training."

Walk through:
1. Prepare data: features only, no target column
2. Configure K-Means estimator with `image_uris.retrieve("kmeans", region)`
3. Set hyperparameters: `k=4`, `feature_dim=6`
4. Launch training with `.fit()`
5. Deploy and predict cluster assignments

---

## STEP 8 -- Visualize Cluster Assignments (8 minutes)

**Pacing: run visualization cell, discuss.**

> "We project the 6-dimensional data down to 2 dimensions using the first two features (amount and hour) and color each point by its cluster assignment. Overlaying the actual fraud labels shows whether the clusters capture any fraud signal."

**Discussion Prompt:** "Do the clusters separate fraud from non-fraud? If cluster 2 has 80% fraud transactions, what could FraudShield do with that information even without a supervised model?" (Segment-level rules, alert thresholds, targeted review queues.)

---

## STEP 9 -- Cleanup K-Means Endpoint (3 minutes)

**Pacing: live code. Mandatory.**

Delete the K-Means endpoint before proceeding.

---

## STEP 10 -- Random Cut Forest for Anomaly Detection (12 minutes)

**Pacing: conceptual intro, then live code.**

> "Random Cut Forest is SageMaker's built-in anomaly detection algorithm. It assigns an anomaly score to each data point. Points that are 'easy to isolate' -- far from the normal distribution -- get high scores. For FraudShield, high-anomaly transactions are candidates for fraud review."

Walk through:
1. Configure RCF estimator with `image_uris.retrieve("randomcutforest", region)`
2. Set hyperparameters: `num_trees=50`, `num_samples_per_tree=256`
3. Train on fraud features (unsupervised -- no target column)
4. Deploy and obtain anomaly scores
5. Compare anomaly scores to actual fraud labels

> "Notice that the highest anomaly scores tend to correspond to fraudulent transactions. RCF does not know what fraud is -- it identifies statistical outliers. In production, you could combine RCF anomaly scores with XGBoost predictions for a two-layer defense."

---

## STEP 11 -- Cleanup RCF Endpoint (2 minutes)

**Pacing: live code. Mandatory.**

Delete the RCF endpoint.

---

## STEP 12 -- BlazingText and Word2Vec (5 minutes)

**Pacing: conceptual only, notebook markdown.**

> "BlazingText is SageMaker's built-in NLP algorithm. It implements Word2Vec (word embeddings) and text classification. On Wednesday you saw transformers -- BlazingText predates transformers and is still useful for fast text classification and embedding generation on very large corpora."

Key points:
- Word2Vec: learns vector representations of words from context
- Text classification mode: supervised, predicts labels from text
- Connection to Wednesday: transformers replaced Word2Vec for most NLP tasks, but BlazingText remains useful for high-throughput, low-latency scenarios

---

## STEP 13 -- DeepAR Time Series Forecasting (5 minutes)

**Pacing: conceptual only, notebook markdown.**

> "DeepAR is SageMaker's built-in time series forecasting algorithm. It uses autoregressive RNNs to generate probabilistic forecasts. For FraudShield, DeepAR could forecast daily transaction volumes or expected fraud rates. Anomalies in actual vs. forecasted values signal emerging fraud patterns."

Key points:
- Input: multiple related time series (e.g., daily transaction volumes per merchant)
- Output: probabilistic forecasts (mean, quantiles)
- Connection to Monday's RNNs: DeepAR uses the same recurrent architecture under the hood

[PAUSE FOR BREAK - 5 MINS]

---

# STAGE 3 -- Hyperparameter Optimization (55 min)

> **Goal:** Use SageMaker Hyperparameter Optimization to systematically search for the best XGBoost hyperparameters. Associates connect HPO to Wednesday's experiment tracking -- HPO is automated, systematic experimentation.

**Exit Criteria Addressed:**
- Describe HPO job anatomy: objective metric, parameter ranges, tuning strategy, max jobs (Required)
- Configure and launch an HPO job using HyperparameterTuner (Required)
- Analyze HPO results to identify the best trial and optimal hyperparameters (Required)
- Connect HPO to experiment tracking as systematic, automated experimentation (Required)

### Instructor Opening (3 minutes -- talk, no code)

> "In Stage 1 you trained XGBoost with `max_depth=5`, `eta=0.2`, `num_round=100`. But how do we know those are the best values? We could try different combinations manually -- that is what Wednesday's experiments were about. HPO automates that search. You define the ranges, the objective metric, and SageMaker launches multiple training jobs, learning from each result to find the optimum."

---

## STEP 14 -- HPO Concepts (8 minutes)

**Pacing: conceptual with notebook markdown.**

> "An HPO job has four components: the objective metric (what to optimize), the parameter ranges (the search space), the strategy (how to explore), and resource limits (how many jobs to run)."

Key concepts:
- **Objective metric:** e.g., `validation:auc` -- must match an eval metric the algorithm logs
- **Parameter ranges:** Continuous (eta), Integer (max_depth, num_round), Categorical
- **Strategies:** Bayesian (learns from prior trials, more efficient) vs Random (embarrassingly parallel, no learning)
- **Early stopping:** kill trials that fall behind the best result
- **Max jobs and max parallel jobs:** budget controls

> "Bayesian optimization treats HPO as a meta-learning problem. After each trial, it updates a probabilistic model of the objective function and picks the next hyperparameters where the expected improvement is highest. Random search ignores prior results entirely. For small budgets, Bayesian is almost always better."

---

## STEP 15 -- Configure HyperparameterTuner (10 minutes)

**Pacing: live code, step by step.**

> "We reuse the XGBoost estimator from Stage 1 but wrap it in a HyperparameterTuner. We define ranges for `max_depth`, `eta`, and `num_round`. The tuner will launch multiple training jobs, each with different hyperparameter values."

Walk through:
1. Define parameter ranges using `IntegerParameter`, `ContinuousParameter`
2. Create `HyperparameterTuner` with `objective_metric_name="validation:auc"`, `objective_type="Maximize"`
3. Set `max_jobs=6`, `max_parallel_jobs=2`
4. Review the configuration before launching

---

## STEP 16 -- Launch HPO Job (10 minutes)

**Pacing: launch and monitor.**

> "We call `.fit()` on the tuner. SageMaker launches training jobs according to the strategy. Each job trains XGBoost with a different combination of hyperparameters."

While HPO runs, discuss:
- HPO dashboard in the SageMaker console
- Each training job is an independent experiment with its own metric
- Bayesian strategy: early jobs explore, later jobs exploit
- Connection to Wednesday: each HPO trial is essentially an experiment run

**Discussion Prompt:** "Wednesday you manually tracked experiments. HPO automates that process. What are the trade-offs between manual experimentation and automated HPO?" (Manual gives intuition and control; HPO is systematic but costs more compute. Best practice: manual exploration to narrow the space, then HPO to fine-tune.)

---

## STEP 17 -- Analyze HPO Results (12 minutes)

**Pacing: live code, step by step.**

> "The tuner provides analytics: the best trial, its hyperparameters, and the objective metric value. We extract these and compare to our Stage 1 baseline."

Walk through:
1. Get best training job name and hyperparameters
2. Extract the objective metric value
3. Compare to Stage 1 baseline (manual hyperparameters)
4. Show all trials: hyperparameters vs. objective metric
5. Visualize parameter importance (which hyperparameter had the most impact)

> "The best HPO trial will typically outperform our manually chosen hyperparameters. This is the value of systematic search -- it finds combinations we would not have tried."

---

## STEP 18 -- Connect to Wednesday's Experiments (5 minutes)

**Pacing: conceptual, notebook markdown.**

> "Wednesday you learned to track experiments with SageMaker Experiments. HPO is the ultimate experiment tracker -- it automatically logs every trial with its hyperparameters and metrics. In a production pipeline, you would run HPO within an experiment, so every tuning job is recorded alongside manual experiments."

Key connection points:
- HPO trials map to experiment runs
- Both capture hyperparameters, metrics, and artifacts
- HPO adds automated search strategy on top of manual tracking
- Production pattern: Experiment -> HPO Tuner -> Best model -> Model Registry

---

## STEP 19 -- Cleanup (5 minutes)

**Pacing: live code. Mandatory.**

> "Delete any remaining endpoints from HPO trials. Check billing. Verify no endpoints are running."

Cleanup checklist:
- Delete any endpoints created during HPO
- Verify no active endpoints in the SageMaker console
- Confirm S3 artifacts are expected (training outputs from HPO trials)

---

## Wrap-up & Q&A Buffer (10 minutes)

### Summary (4 minutes)

> "Today you accomplished three things. First, you trained XGBoost as a built-in algorithm -- no training script, just data and hyperparameters -- and compared it to Monday's Random Forest. You saw how built-in algorithms simplify the workflow while often delivering better performance. Second, you explored unsupervised algorithms: K-Means revealed natural clusters in the fraud data, and Random Cut Forest flagged anomalies without any labels. You also surveyed BlazingText for NLP and DeepAR for time series. Third, you used HPO to systematically search for the best XGBoost hyperparameters -- automating the experimentation process from Wednesday and finding better configurations than manual tuning."

### Friday Preview (2 minutes)

> "Friday wraps up the week with SageMaker Pipelines and end-to-end ML workflows. You will connect everything -- data preparation, training, evaluation, registration, and deployment -- into an automated pipeline. The manual steps you have been doing all week become reproducible, auditable DAG steps."

### Open Q&A (4 minutes)

---

## Instructor Notes -- Common Issues

| Issue | Resolution |
|-------|-----------|
| XGBoost data format error (target not first column) | Verify CSV has target as column 0, no header, no index. Use `to_csv(header=False, index=False)`. |
| `AlgorithmError` during XGBoost training | Check data format. Most common cause: header row present in CSV. |
| K-Means returns unexpected cluster count | Verify `k` hyperparameter matches expectation. Check that `feature_dim` matches actual column count. |
| RCF anomaly scores are all similar | Data may lack clear outliers. Use synthetic injection: add a few extreme rows and re-run. |
| HPO job stuck at 0 completed | Check CloudWatch logs for first training job. Common cause: data format or role permission issue. |
| HPO max_jobs budget exceeded | Reduce `max_jobs` or `max_parallel_jobs`. Each HPO trial is a full training job with its own cost. |
| `ResourceLimitExceeded` on HPO | Account-level limits on concurrent training jobs. Reduce `max_parallel_jobs` to 1. |
| XGBoost endpoint returns raw float instead of 0/1 | XGBoost with `binary:logistic` returns probabilities. Apply threshold (e.g., 0.5) in post-processing. |
| Student forgets to delete endpoints | Walk over immediately. HPO can create multiple endpoints. Check billing together. |
| Container image not found for region | Verify `image_uris.retrieve("xgboost", region, "1.5-1")` works. Some regions have limited algorithm support. |
