# Multi-Model Endpoints

**Estimated Time:** 10 Minutes

## Introduction

In the foundational SageMaker skill, each endpoint hosted a single model. This one-model-per-endpoint pattern is simple but expensive when you need to serve many models. Consider a SaaS platform that trains a personalized model for each customer, or a retailer with a separate demand forecasting model per store. Deploying 500 individual endpoints at $170/month each would cost $85,000/month in infrastructure alone.

SageMaker **Multi-Model Endpoints (MME)** solve this by hosting multiple models on the same endpoint infrastructure. Models are loaded into memory on demand, swapped in and out as requests arrive, and share the same pool of instances. This can reduce costs by 90% or more compared to dedicated endpoints.

## Core Concepts

### How Multi-Model Endpoints work

1. **Model artifacts** (`.tar.gz` files) are stored in a common S3 prefix (e.g., `s3://bucket/models/`).
2. **A single endpoint** is created pointing to this S3 prefix rather than a single model artifact.
3. When a request arrives, the client specifies **which model** to invoke via the `TargetModel` parameter.
4. SageMaker checks if the requested model is loaded in memory:
   - **If loaded:** Routes the request directly to the model. Latency is the same as a single-model endpoint.
   - **If not loaded:** Downloads the model artifact from S3, loads it into memory, and then processes the request. This adds a few seconds of latency (a "model loading" cold start).
5. When instance memory is full, SageMaker **evicts** the least-recently-used model to make room for new ones.

### Creating an MME in the console

1. Navigate to **SageMaker > Inference > Models > Create model**.
2. **Model name:** Enter a name (e.g., `customer-models-mme`).
3. **Container definition:** Select the inference container image URI.
4. **Model data URL:** Set to the S3 prefix containing all model artifacts (e.g., `s3://bucket/models/`). This is the key difference from single-model endpoints, where you specify a single `.tar.gz` URI.
5. **Enable Multi-Model:** Toggle the multi-model option on.
6. Create an endpoint configuration and endpoint as usual.

### Invoking a specific model

When calling the endpoint, include the `TargetModel` parameter specifying the model artifact filename:

```
invoke_endpoint(
    EndpointName='customer-models-mme',
    TargetModel='customer-123.tar.gz',
    ContentType='text/csv',
    Body=payload
)
```

SageMaker resolves `TargetModel` relative to the S3 prefix configured in the model definition. The model `customer-123.tar.gz` maps to `s3://bucket/models/customer-123.tar.gz`.

### Adding and removing models

To add a new model to the MME, simply upload a new `.tar.gz` file to the S3 prefix. No endpoint update or redeployment is needed. The new model becomes available immediately -- SageMaker will load it on the first request.

To remove a model, delete the `.tar.gz` file from S3. SageMaker will evict it from memory on the next cache cycle. There is no active deregistration step.

### Memory management and caching

MME performance depends on how many models fit in memory simultaneously:

- **Hot models:** Currently loaded in memory. Inference latency matches single-model endpoints.
- **Cold models:** Not in memory. First request incurs a loading delay (model download + deserialization).
- **Eviction policy:** Least Recently Used (LRU). Models that have not been invoked recently are evicted first when memory pressure increases.

To optimize:

- **Right-size instances:** Choose instance types with enough memory to hold your most frequently accessed models. If 80% of traffic goes to 50 out of 500 models, size instances to hold those 50 models comfortably.
- **Model compression:** Smaller model artifacts load faster and consume less memory. Quantize or prune models where accuracy impact is acceptable.
- **Auto-scaling:** Scale the instance count based on invocation rate to maintain enough aggregate memory for the working set.

### When to use MME vs. dedicated endpoints

| Scenario | MME | Dedicated endpoints |
| :--- | :--- | :--- |
| 100+ models with variable traffic | MME (massive cost savings) | Impractical at scale |
| All models use the same framework | MME (shared container) | N/A |
| Models have very different latency SLAs | Dedicated (guaranteed resources) | Appropriate |
| Models require different instance types | Dedicated (MME uses one type) | Appropriate |
| Model count changes frequently | MME (just upload to S3) | Requires redeployment |

### Limitations

- All models in an MME must use the **same inference container** (same framework and version).
- All models share the **same instance type** -- you cannot mix CPU and GPU models.
- There is no guaranteed memory reservation per model. High-traffic models may evict low-traffic models.
- Model loading latency on cold start depends on model artifact size and instance I/O performance.

## Connecting to Practice

MME is the cost-optimization pattern for multi-model serving. The next topic, *Multi-Container Endpoints*, covers a different pattern: serving a pipeline of different model containers (e.g., preprocessing + inference) on a single endpoint. The module assignment will require you to deploy an MME with at least 3 model artifacts, invoke different models, and measure cold vs. warm latency.

## Further Learning & Resources

**Documentation and reading**

- **[Multi-Model Endpoints](https://docs.aws.amazon.com/sagemaker/latest/dg/multi-model-endpoints.html)** - *Docs*: Complete reference for MME configuration, invocation, and model management.
- **[MME Best Practices](https://aws.amazon.com/blogs/machine-learning/best-practices-for-amazon-sagemaker-multi-model-endpoints/)** - *Blog*: AWS blog with performance tuning and cost optimization guidance.

**Interactive practice**

- **[Multi-Model Endpoint Example](https://github.com/aws/amazon-sagemaker-examples/tree/main/advanced_functionality/multi_model_xgboost_home_value)** - *Interactive*: Sample notebook demonstrating MME deployment with multiple XGBoost models.
