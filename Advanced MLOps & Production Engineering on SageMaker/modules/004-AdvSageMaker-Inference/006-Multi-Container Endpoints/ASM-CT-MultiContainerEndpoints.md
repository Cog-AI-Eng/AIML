# Multi-Container Endpoints

**Estimated Time:** 10 Minutes

## Introduction

Multi-Model Endpoints host many models of the same type on shared infrastructure. **Multi-Container Endpoints** solve a different problem: running a *pipeline* of different model containers on a single endpoint. Instead of sending data to one model, you chain multiple containers -- for example, a preprocessing container that transforms raw text into features, followed by an inference container that generates predictions.

Without Multi-Container Endpoints, implementing this pipeline requires either: (a) building a custom container that bundles all logic together, which creates a monolith that is hard to update, or (b) deploying separate endpoints and chaining API calls, which adds network latency and operational overhead. Multi-Container Endpoints let you compose independent containers into a single endpoint while keeping each container independently versioned and deployable.

## Core Concepts

### Two invocation modes

Multi-Container Endpoints support two modes:

**Direct invocation:** The client specifies which container to invoke using the `TargetContainerHostname` parameter. Each container operates independently. This mode is essentially hosting multiple unrelated models on one endpoint (different from MME because each container can use a different framework).

**Serial inference pipeline:** Containers are chained in sequence. The output of container 1 becomes the input of container 2, and so on. The client sends a single request, and it flows through the entire pipeline. The response comes from the last container in the chain.

### Serial inference pipeline architecture

A typical pipeline:

1. **Container A (Preprocessing):** Receives raw input (e.g., a JSON with text and metadata). Transforms it into model-ready features (e.g., tokenized text vectors). Output is passed to Container B.
2. **Container B (Inference):** Receives the preprocessed features. Runs the ML model. Generates predictions. Output is the final response.

Optionally, you can add more stages:

3. **Container C (Post-processing):** Receives raw predictions. Applies business rules (e.g., mapping probability scores to risk categories). Returns the final formatted response.

Each container runs its own Docker image and can use a different framework. Container A might use a custom Python image for feature engineering, Container B might use the SageMaker XGBoost container, and Container C might use a lightweight Flask-based transformer.

### Creating a Multi-Container Endpoint

1. Navigate to **SageMaker > Inference > Models > Create model**.
2. Under **Container definition**, select **Add container** multiple times.
3. For each container, specify:
   - **Container hostname:** A unique name (e.g., `preprocessor`, `predictor`, `postprocessor`).
   - **Image:** The Docker image URI for this container.
   - **Model data URL:** The S3 path to the model artifact for this container (if applicable; preprocessors may not have a model artifact).
   - **Environment variables:** Any configuration the container needs.
4. **Inference mode:** Select **Serial pipeline** (containers process in order) or **Direct** (client selects which container to invoke).
5. Create an endpoint configuration and endpoint as usual.

### Data flow in serial pipelines

In serial mode, SageMaker manages the inter-container communication:

- The client's request goes to Container 1.
- Container 1's response (stdout) is automatically piped as the request body to Container 2.
- Container 2's response goes to Container 3 (if present).
- The final container's response is returned to the client.

Each container must accept and produce data in a compatible format. For example, if Container A outputs JSON, Container B must accept JSON input. SageMaker does not transform data between containers.

### Advantages over monolithic containers

| Factor | Multi-Container Pipeline | Monolithic Container |
| :--- | :--- | :--- |
| Independence | Each container is versioned and updated independently | Single image; any change requires full rebuild |
| Reusability | Share a preprocessing container across multiple endpoints | Preprocessing logic duplicated per model |
| Debugging | Isolate issues to a specific container | All logic in one container complicates debugging |
| Framework mixing | Each container can use a different framework | All code must run in one environment |
| Scaling | All containers share the same instance(s) | Same |

### Limitations

- **Maximum containers:** Up to 15 containers per endpoint.
- **Shared resources:** All containers run on the same instance(s). Memory and CPU are shared, not partitioned. A memory-hungry preprocessor can starve the inference container.
- **No branching:** Serial pipelines are strictly linear (A -> B -> C). Conditional routing or fan-out patterns require external orchestration (e.g., Lambda or Step Functions).
- **Instance type:** All containers use the same instance type. You cannot assign GPU to the inference container and CPU to the preprocessor.

### Direct invocation use case

Direct invocation mode is useful when you want to host multiple unrelated models (potentially using different frameworks) on a single endpoint to reduce costs, but you do not want them chained. The client explicitly specifies which container to invoke per request.

This is conceptually similar to MME but with a key difference: each container in a Multi-Container Endpoint can use a different framework (e.g., one container for scikit-learn, another for TensorFlow). MME requires all models to share the same container image.

## Connecting to Practice

Multi-Container Endpoints complete the inference patterns for this module. You have now covered Serverless, Async, Batch Transform, Multi-Model, and Multi-Container architectures. The module lecture will walk through building a two-container serial pipeline (preprocessing + inference) and deploying it. The assignment will require you to implement a Multi-Container Endpoint in direct invocation mode hosting two models with different frameworks.

## Further Learning & Resources

**Documentation and reading**

- **[Multi-Container Endpoints](https://docs.aws.amazon.com/sagemaker/latest/dg/multi-container-endpoints.html)** - *Docs*: Complete reference for Multi-Container configuration, serial pipelines, and direct invocation.
- **[Inference Pipelines](https://docs.aws.amazon.com/sagemaker/latest/dg/inference-pipelines.html)** - *Docs*: Guide to serial inference pipeline architecture and data flow.

**Interactive practice**

- **[Inference Pipeline Example](https://github.com/aws/amazon-sagemaker-examples/tree/main/advanced_functionality/inference_pipeline_sparkml_xgboost_abalone)** - *Interactive*: Sample notebook demonstrating a serial inference pipeline with SparkML preprocessing and XGBoost inference.
