# Inference Recommender

**Estimated Time:** 10 Minutes

## Introduction

Choosing the right instance type for inference is traditionally a trial-and-error process: deploy on `ml.m5.xlarge`, run a load test, check latency, try `ml.c5.xlarge`, repeat. SageMaker **Inference Recommender** automates this process by benchmarking your model across multiple instance types and configurations, then recommending the best option based on your latency and cost constraints.

## Core Concepts

### How Inference Recommender works

Inference Recommender runs load tests against your model on multiple instance types simultaneously. It measures latency, throughput, and cost for each configuration, then ranks the results by your optimization objective (lowest cost, lowest latency, or best cost-per-inference).

### Two job types

**Default job:** A quick benchmark (~45 minutes) that tests your model on a predefined set of instance types. SageMaker selects the instances based on your model's framework and size. This is the fastest way to get a recommendation.

**Advanced job:** A customized benchmark where you specify:
- Exact instance types to test (e.g., `ml.m5.xlarge`, `ml.c5.xlarge`, `ml.c5.2xlarge`).
- Traffic patterns (steady load, burst load, ramp-up).
- Duration per instance type.
- Custom latency and throughput constraints.

### Creating an Inference Recommender job in the console

1. Navigate to **SageMaker > Inference > Inference recommender**.
2. Click **Create recommendation job**.
3. **Model package:** Select a model registered in Model Registry (Inference Recommender requires Model Registry registration).
4. **Job type:** Default or Advanced.
5. **Sample payload:** Upload a representative inference request payload (e.g., a CSV row).
6. **For Advanced jobs:** Specify instance types, traffic patterns, and constraints.
7. Click **Create job**.

### Interpreting results

After the job completes, the console shows a ranked table of recommendations:

| Instance Type | Latency (p95) | Throughput (inferences/sec) | Cost/hr | Cost/inference |
| :--- | :--- | :--- | :--- | :--- |
| ml.c5.xlarge | 12ms | 450 | $0.204 | $0.000126 |
| ml.m5.xlarge | 15ms | 380 | $0.230 | $0.000168 |
| ml.c5.2xlarge | 8ms | 900 | $0.408 | $0.000126 |
| ml.inf1.xlarge | 10ms | 800 | $0.228 | $0.000079 |

The **Cost/inference** column is the most actionable metric: it accounts for both instance cost and throughput to show the true marginal cost of each prediction.

### When to use Inference Recommender

- **Before initial deployment:** Run a Default job to identify the most cost-effective instance type rather than guessing.
- **Before scaling up:** When an endpoint is approaching capacity limits, run an Advanced job to compare scaling out (more instances of the same type) vs. scaling up (fewer instances of a larger type).
- **After model changes:** If you retrain your model or change preprocessing, the optimal instance type may change. Re-run Inference Recommender with the new model.
- **For instance family migrations:** When AWS releases new instance types (e.g., `ml.m6i` replacing `ml.m5`), Inference Recommender can benchmark the new types against your current configuration.

### Requirements

- The model must be registered in **Model Registry** (Inference Recommender pulls the model from the registry).
- A **sample payload** must be provided (so the recommender can send realistic requests during benchmarking).
- The model container must support the SageMaker inference protocol (standard `model_fn`, `input_fn`, `predict_fn`, `output_fn` or the default serving stack).

### Limitations

- Inference Recommender tests instance types in isolation. It does not evaluate multi-instance deployments (auto-scaling behavior).
- Custom container images may require additional configuration (environment variables, model loading scripts) to work with the recommender.
- Benchmarks use synthetic traffic. Real production traffic patterns may differ from the test patterns.

## Connecting to Practice

Inference Recommender takes the guesswork out of instance selection. The next topic, *CloudWatch Dashboards*, covers building operational visibility for your deployed endpoints. The module assignment will require you to run an Inference Recommender job and use the results to select and justify an instance type for a production endpoint.

## Further Learning & Resources

**Documentation and reading**

- **[Inference Recommender](https://docs.aws.amazon.com/sagemaker/latest/dg/inference-recommender.html)** - *Docs*: Complete reference for job types, configuration, and result interpretation.

**Interactive practice**

- **[Inference Recommender Example](https://github.com/aws/amazon-sagemaker-examples/tree/main/sagemaker-inference-recommender)** - *Interactive*: Sample notebook demonstrating both Default and Advanced recommendation jobs.
