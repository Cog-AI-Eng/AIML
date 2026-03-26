# Random Cut Forest

**Estimated Time:** 10 Minutes

## Introduction

Anomaly detection -- identifying data points that deviate significantly from expected patterns -- is one of the most common production ML tasks. Fraudulent transactions, sensor malfunctions, infrastructure outages, and unusual user behavior all manifest as anomalies in data streams. SageMaker provides **Random Cut Forest (RCF)** as a built-in unsupervised algorithm specifically designed for anomaly detection on both tabular and time-series data.

RCF works differently from density-based approaches like Isolation Forest. It constructs a forest of random trees, where each tree is built by recursively cutting the data along random dimensions. Points that require fewer cuts to isolate are considered more anomalous. The algorithm outputs an **anomaly score** for each data point -- higher scores indicate greater abnormality.

## Core Concepts

### Algorithm mechanics

1. **Sampling:** RCF draws random subsamples from the training data. Each subsample feeds one tree in the forest.
2. **Tree construction:** For each subsample, the algorithm randomly selects a dimension and a cut value within that dimension's range. It splits the data and recurses on each partition until each leaf contains at most one point.
3. **Anomaly scoring:** To score a new data point, RCF inserts it into each tree and measures the *displacement* -- how much the tree structure changes to accommodate the point. Points that cause large displacements (they do not fit well into the existing structure) receive high anomaly scores.
4. **Ensemble averaging:** The final anomaly score is the average displacement across all trees in the forest.

### Why RCF over other anomaly detectors

| Approach | Strengths | Limitations |
| :--- | :--- | :--- |
| RCF (SageMaker) | No labels required, handles streaming data, built-in SageMaker integration, scales to high dimensions | Scores require threshold tuning for alerting |
| Isolation Forest (scikit-learn) | Similar principle, well-studied | Single-machine, no native streaming support |
| Statistical methods (z-score, IQR) | Simple, interpretable | Assume normal distribution, break on multivariate data |
| Autoencoders | Learn complex normal patterns | Require labeled data for threshold calibration, expensive to train |

RCF is particularly well-suited for SageMaker because it supports **streaming inference**: you can deploy an RCF model to a real-time endpoint and score incoming data points as they arrive, enabling near-real-time anomaly alerting.

### Configuring an RCF training job

1. Navigate to **SageMaker > Training > Training jobs > Create training job**.
2. **Algorithm:** Select **Random Cut Forest** from the built-in list.
3. **Key hyperparameters:**
   - `num_trees`: Number of trees in the forest (default: 50). More trees improve score stability at the cost of training time and model size.
   - `num_samples_per_tree`: Number of data points sampled for each tree (default: 256). Should be set relative to the expected proportion of anomalies.
   - `feature_dim`: Number of features in the input data.
4. **Input data:** S3 path in CSV or RecordIO-protobuf format. No target column -- RCF is unsupervised.
5. **Instance type:** `ml.m5.xlarge` is sufficient for most datasets. RCF training is fast even on large data because each tree processes only a subsample.

### Interpreting anomaly scores

RCF outputs a single numeric anomaly score per data point. The score is not a probability; it is a relative measure. To use it for alerting:

1. Score all training data (or a representative holdout) using batch transform or endpoint inference.
2. Examine the score distribution. Most points will have low scores. The distribution typically has a long right tail.
3. Set a threshold based on your tolerance: a common starting point is 3 standard deviations above the mean score, or the 99th percentile.
4. Points scoring above the threshold are flagged as anomalies.

There is no universal "correct" threshold -- it depends on the cost of false positives (investigating normal transactions) vs. false negatives (missing real anomalies) for your specific use case.

### Time-series anomaly detection

RCF supports time-series data by treating each time window as a data point. The approach:

1. **Shingling:** Convert the time series into fixed-width windows (shingles). For example, with a shingle size of 8, each data point is a vector of 8 consecutive values from the time series.
2. Set `feature_dim` to the shingle size.
3. Train RCF on the shingled data. Unusual temporal patterns (spikes, drops, level shifts) will produce high anomaly scores.

SageMaker's RCF container handles shingling automatically if you set the `shingle_size` hyperparameter and provide single-column time-series input.

## Connecting to Practice

RCF gives you an unsupervised anomaly detection tool that complements K-Means clustering. The next topic, *BlazingText & Word2Vec*, moves from numeric data to text. In the module lecture, you will train an RCF model on time-series sensor data and set up anomaly thresholds. The assignment will require you to deploy an RCF endpoint and score a stream of incoming data points.

## Further Learning & Resources

**Documentation and reading**

- **[SageMaker Random Cut Forest](https://docs.aws.amazon.com/sagemaker/latest/dg/randomcutforest.html)** - *Docs*: Built-in algorithm reference with hyperparameter details and input format specifications.
- **[Random Cut Forest Paper](https://proceedings.mlr.press/v48/guha16.html)** - *Reading*: The original research paper describing the RCF algorithm and its theoretical properties.

**Interactive practice**

- **[Anomaly Detection with RCF on SageMaker](https://github.com/aws/amazon-sagemaker-examples/tree/main/introduction_to_amazon_algorithms/random_cut_forest)** - *Interactive*: Sample notebook demonstrating RCF training and anomaly scoring on NYC taxi data.
