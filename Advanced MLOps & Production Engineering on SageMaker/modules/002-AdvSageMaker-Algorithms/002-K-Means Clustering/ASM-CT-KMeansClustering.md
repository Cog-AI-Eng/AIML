# K-Means Clustering

**Estimated Time:** 10 Minutes

## Introduction

The algorithms you have worked with so far -- logistic regression, random forests, XGBoost -- are all supervised: they require labeled training data with a known target column. K-Means is SageMaker's built-in **unsupervised** algorithm for clustering. Given a dataset with no labels, K-Means groups records into *k* clusters based on feature similarity. Each record is assigned to the cluster whose center (centroid) is nearest.

K-Means is the most commonly used clustering algorithm in production ML for customer segmentation, anomaly grouping, and data exploration. SageMaker's built-in K-Means implementation is optimized for large-scale datasets and supports distributed training across multiple instances, making it practical for datasets that would be too large for scikit-learn's in-memory implementation.

## Core Concepts

### Algorithm mechanics

1. **Initialization:** Select *k* initial centroids. SageMaker's implementation uses the `kmeans++` initialization strategy by default, which spreads initial centroids to reduce convergence time.
2. **Assignment:** Assign each data point to the nearest centroid (using Euclidean distance).
3. **Update:** Recalculate each centroid as the mean of all points assigned to its cluster.
4. **Repeat:** Iterate assignment and update until centroids stabilize (convergence) or the maximum number of iterations is reached.

### SageMaker K-Means vs. scikit-learn

| Feature | SageMaker K-Means | scikit-learn KMeans |
| :--- | :--- | :--- |
| Data scale | Distributed, handles datasets too large for memory | Single-machine, in-memory |
| Input format | RecordIO-protobuf or CSV in S3 | NumPy arrays or pandas DataFrames |
| GPU support | Yes (optional, via `ml.p2`/`ml.p3` instances) | No |
| Mini-batch mode | Built-in (`mini_batch_size` parameter) | Separate `MiniBatchKMeans` class |
| Integration | Direct Pipeline/endpoint deployment | Manual export and deployment |

For datasets under a few GB, scikit-learn is simpler. SageMaker K-Means shines when you need distributed processing, GPU acceleration, or direct integration with Pipelines and endpoints.

### Configuring a K-Means training job

1. Navigate to **SageMaker > Training > Training jobs > Create training job**.
2. **Algorithm:** Select **K-Means** from the built-in algorithm list.
3. **Key hyperparameters:**
   - `k`: Number of clusters. This is the primary parameter you must choose. Use domain knowledge or the elbow method (train for multiple values of *k* and plot the within-cluster sum of squares).
   - `feature_dim`: Number of features in the input data.
   - `mini_batch_size`: Number of records per mini-batch. Larger batches are faster but use more memory. Default: 5000.
   - `max_iterations`: Maximum assignment-update cycles. Default: 10 for the web-scale version.
   - `init_method`: Initialization strategy. Default: `random`. Use `kmeans++` for better convergence.
4. **Input data:** S3 path to training data in CSV format (no header, no target column) or RecordIO-protobuf.
5. **Instance type:** `ml.m5.xlarge` for CPU training on moderate datasets. `ml.p2.xlarge` for GPU-accelerated training on larger datasets (note: GPU instances incur higher costs).

### Interpreting K-Means output

After training, the model artifact in S3 contains the learned centroids. When deployed to an endpoint, the model accepts new data points and returns:

- **`closest_cluster`**: The cluster index (0 to k-1) assigned to each data point.
- **`distance_to_cluster`**: The Euclidean distance from the data point to its assigned centroid. Points with unusually high distances may be outliers.

You can use the centroid values to profile each cluster: compute the mean feature values per cluster and compare them to understand what makes each segment distinct. This profiling is typically done in a notebook after training.

### Choosing k

The most common challenge with K-Means is selecting the right number of clusters. Practical approaches:

- **Elbow method:** Train models for k = 2, 3, 4, ..., 20. Plot the average within-cluster distance (available in the training job metrics). The "elbow" -- where additional clusters stop providing significant improvement -- suggests the optimal k.
- **Silhouette score:** Compute the silhouette coefficient for each k. Higher scores indicate better-defined clusters.
- **Domain knowledge:** In customer segmentation, the number of clusters often maps to actionable business segments (e.g., "high-value frequent buyers," "bargain hunters," "churning customers").

In SageMaker, you can automate this search using an HPO job that treats k as a tunable parameter and optimizes for within-cluster distance.

## Connecting to Practice

K-Means introduces unsupervised learning on SageMaker. The next topic, *Random Cut Forest*, covers another unsupervised algorithm specialized for anomaly detection. In the module assignment, you will train a K-Means model for customer segmentation, profile the resulting clusters, and deploy the model to an endpoint for real-time cluster assignment.

## Further Learning & Resources

**Documentation and reading**

- **[SageMaker K-Means Algorithm](https://docs.aws.amazon.com/sagemaker/latest/dg/k-means.html)** - *Docs*: Built-in algorithm reference with hyperparameter definitions and input format requirements.

**Interactive practice**

- **[K-Means Clustering on SageMaker](https://github.com/aws/amazon-sagemaker-examples/tree/main/introduction_to_amazon_algorithms/k_means_mnist)** - *Interactive*: Sample notebook demonstrating K-Means on the MNIST dataset with visualization of clusters.
