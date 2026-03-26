# XGBoost Architecture

**Estimated Time:** 10 Minutes

## Introduction

In the foundational AIML skill, you trained tree-based models using scikit-learn's `RandomForestClassifier` and `GradientBoostingClassifier`. XGBoost (Extreme Gradient Boosting) is the industrial-grade evolution of gradient boosting that dominates tabular ML competitions and production systems. SageMaker provides XGBoost as a **built-in algorithm** -- a pre-packaged, optimized container that you can train and deploy without writing a custom Docker image or installing dependencies.

Understanding XGBoost's architecture matters at the SageMaker level because the built-in XGBoost container supports both **Script Mode** (you provide a training script) and **Algorithm Mode** (you provide only data and hyperparameters). The choice affects how you configure training jobs, which hyperparameters you can tune, and how you integrate XGBoost into SageMaker Pipelines.

This reading covers XGBoost's algorithmic architecture, how it is packaged as a SageMaker built-in algorithm, and how to configure a training job using both modes from the console.

## Core Concepts

### How XGBoost works

XGBoost builds an ensemble of decision trees sequentially. Each tree is trained to correct the errors of the previous trees by fitting the *gradient of the loss function* (hence "gradient boosting"). The final prediction is the sum of all trees' outputs.

Key architectural elements:

- **Objective function:** A loss function (e.g., `binary:logistic` for binary classification, `reg:squarederror` for regression) plus a regularization term that penalizes complex trees (many leaves, large leaf weights). This regularization is what makes XGBoost more resistant to overfitting than basic gradient boosting.
- **Boosting rounds (`num_round`):** The number of trees to build. Each round adds one tree. More rounds improve training accuracy but risk overfitting.
- **Learning rate (`eta`):** Controls how much each new tree contributes to the ensemble. Smaller values (e.g., 0.1) require more rounds but produce more robust models.
- **Tree depth (`max_depth`):** Maximum depth of each decision tree. Deeper trees capture more complex interactions but increase overfitting risk and training time.
- **Column sampling (`colsample_bytree`):** Fraction of features randomly selected for each tree. Introduces diversity and reduces overfitting, similar to how Random Forest samples features.

### SageMaker's built-in XGBoost container

SageMaker maintains a pre-built Docker container with XGBoost installed and optimized for distributed training. You do not need to build a container or manage dependencies. The container supports:

- **Algorithm Mode:** You provide a CSV or LibSVM dataset in S3 and set hyperparameters in the training job config. No training script required. SageMaker handles data loading, training, and model serialization.
- **Script Mode:** You provide a Python training script (`train.py`) that uses the `xgboost` library. This gives you full control over data preprocessing, custom evaluation metrics, and model output. The SageMaker SDK's `XGBoost` estimator class manages the container and passes your script as the entry point.

### Configuring an XGBoost training job in the console

**Algorithm Mode:**

1. Navigate to **SageMaker > Training > Training jobs > Create training job**.
2. **Algorithm source:** Select **SageMaker built-in algorithm**.
3. **Algorithm:** Choose **XGBoost**.
4. **Hyperparameters:** Set `objective` (e.g., `binary:logistic`), `num_round` (e.g., `100`), `max_depth` (e.g., `6`), `eta` (e.g., `0.1`), and any others you want to configure.
5. **Input data configuration:** Specify the S3 path for training data (channel name: `train`) and optional validation data (channel name: `validation`). Data must be in CSV (with no header, target in the first column) or LibSVM format.
6. **Output data configuration:** Specify the S3 path for model artifacts.
7. **Resource configuration:** Select `ml.m5.xlarge` (Free Tier compatible) and set instance count to 1.
8. Click **Create training job**.

**Script Mode:**

1. Same console flow, but under **Algorithm source**, select **Your own algorithm container** and provide the SageMaker XGBoost container URI for your region.
2. Under **Hyperparameters**, set `sagemaker_program` to your script filename (e.g., `train.py`) and `sagemaker_submit_directory` to the S3 path of your zipped source code.
3. The rest of the configuration is the same.

In practice, most teams use Script Mode via the SDK rather than the console for reproducibility. The console walkthrough here builds your understanding of what the SDK call configures behind the scenes.

### Key hyperparameters for tuning

| Parameter | Range | Effect |
| :--- | :--- | :--- |
| `num_round` | 50-1000 | More rounds = higher capacity, risk of overfitting |
| `max_depth` | 3-10 | Deeper trees = more complex interactions |
| `eta` | 0.01-0.3 | Lower = more conservative learning, needs more rounds |
| `min_child_weight` | 1-10 | Higher = more conservative splits, reduces overfitting |
| `subsample` | 0.5-1.0 | Fraction of training samples per tree |
| `colsample_bytree` | 0.5-1.0 | Fraction of features per tree |
| `gamma` | 0-5 | Minimum loss reduction to make a split |
| `alpha` | 0-1 | L1 regularization on leaf weights |
| `lambda` | 0-1 | L2 regularization on leaf weights |

These parameters are the primary tuning knobs for the HPO jobs you will configure in later topics (*HPO Job Anatomy* and *Optimization Strategies*).

## Connecting to Practice

XGBoost is the workhorse algorithm you will use across multiple modules. The next topics in this module cover other built-in algorithms (K-Means, Random Cut Forest, BlazingText, DeepAR) for different problem types, followed by HPO topics that show you how to systematically tune XGBoost's hyperparameters at scale. In the module lecture, you will launch an XGBoost training job from the console and compare Algorithm Mode vs. Script Mode output. The assignment will require you to train an XGBoost model using Script Mode and evaluate it against an Autopilot baseline from Module 1.

## Further Learning & Resources

**Documentation and reading**

- **[SageMaker XGBoost Algorithm](https://docs.aws.amazon.com/sagemaker/latest/dg/xgboost.html)** - *Docs*: Complete reference for the built-in XGBoost container, hyperparameters, and input/output formats.
- **[XGBoost Documentation](https://xgboost.readthedocs.io/)** - *Docs*: Official XGBoost library documentation with detailed algorithm descriptions.

**Interactive practice**

- **[Train and Deploy XGBoost on SageMaker](https://github.com/aws/amazon-sagemaker-examples/tree/main/introduction_to_amazon_algorithms/xgboost_abalone)** - *Interactive*: AWS sample notebook demonstrating XGBoost training and deployment end to end.
