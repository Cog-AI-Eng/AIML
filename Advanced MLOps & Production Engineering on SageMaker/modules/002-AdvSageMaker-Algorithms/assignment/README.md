# Built-in Algorithms and Hyperparameter Optimization Lab

## Scenario
FraudShield Risk Analytics has prepared a curated feature set from e-commerce transaction data. The data engineering layer is in place, and the team is now ready to train models using SageMaker's built-in algorithms. The goal is threefold: build a supervised fraud classifier with XGBoost, segment customers into behavioral clusters with K-Means, and detect anomalous transaction patterns with Random Cut Forest.

Rather than writing custom training scripts, the team will leverage SageMaker's Algorithm Mode, which lets you configure and launch training jobs entirely from the console. After establishing baseline models, you will use Hyperparameter Optimization (HPO) to systematically search for the best XGBoost configuration across multiple trials, maximizing fraud detection performance while staying within Free Tier compute limits.

---

## Learning Objectives
By completing this lab you will demonstrate the ability to:
1. Launch an XGBoost training job in Algorithm Mode from the SageMaker console without writing a training script.
2. Train a K-Means clustering model on customer features and interpret cluster assignments.
3. Train a Random Cut Forest model for anomaly detection on time-series transaction data.
4. Configure an HPO tuning job with Bayesian strategy, parameter ranges, and an objective metric.
5. Analyze HPO results in the console to identify the best trial and compare hyperparameter values.
6. Clean up all training jobs, models, HPO jobs, and S3 artifacts.

---

## Prerequisites
- A SageMaker Studio Domain provisioned in us-east-1 (from Module 1 or equivalent).
- The FraudShield transactions CSV uploaded to S3 in CSV format with a header row.
- A separate time-series transaction file for anomaly detection (or a filtered subset of the main dataset with a timestamp column).
- An IAM execution role with S3 and SageMaker permissions.

---

## Milestones

| # | Guide | Estimated Time | What You Build |
|---|-------|---------------|----------------|
| 1 | [Train XGBoost in Algorithm Mode](console_guides/01_train_xgboost_algorithm_mode.md) | 25 min | A supervised XGBoost fraud classifier trained via the console |
| 2 | [Train K-Means Clustering](console_guides/02_train_kmeans_clustering.md) | 20 min | A K-Means model with customer segment assignments |
| 3 | [Train Random Cut Forest](console_guides/03_train_random_cut_forest.md) | 20 min | An RCF anomaly detection model on transaction data |
| 4 | [Configure an HPO Job](console_guides/04_configure_hpo_job.md) | 25 min | A Bayesian HPO tuning job with 10 trials for XGBoost |
| 5 | [Analyze HPO Results](console_guides/05_analyze_hpo_results.md) | 15 min | A comparative analysis of HPO trials with the best configuration identified |
| 6 | [Cleanup](console_guides/06_cleanup.md) | 10 min | All training jobs, models, HPO jobs, and S3 artifacts deleted |
| SDK | [SDK Algorithms Lab](notebooks/sdk_algorithms_lab.ipynb) | 50 min | Train XGBoost and K-Means using the SageMaker Python SDK Estimator API, then configure and run an HPO Tuner programmatically |

**Total estimated time:** ~165 minutes (console guides ~115 min + SDK notebook ~50 min)

---

## Presentation Deliverables
1. Show the completed XGBoost training job with its hyperparameters and validation metric.
2. Show the K-Means training job and explain the cluster assignments.
3. Show the Random Cut Forest training job and describe how anomaly scores are produced.
4. Walk through the HPO job configuration, including parameter ranges and strategy.
5. Present the HPO results chart, identify the best trial, and explain why those parameters performed best.
6. Confirm all resources have been deleted in the cleanup guide.

---

## Important Reminders
- **Free Tier:** Use ml.m5.xlarge or smaller. No GPU instances.
- **Region Consistency:** Stay in us-east-1.
- **Cleanup Is Mandatory:** Always complete the cleanup guide.
- **Do Not Skip Steps:** Each guide builds on the previous one.
