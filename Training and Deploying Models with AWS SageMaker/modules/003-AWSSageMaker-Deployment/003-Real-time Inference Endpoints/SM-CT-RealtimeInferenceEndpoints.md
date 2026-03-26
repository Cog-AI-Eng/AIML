# Real-time Inference Endpoints

**Estimated Time:** 10 Minutes

## Introduction

You have trained a model, saved its artifact to S3, registered it in the Model Registry, and approved it for deployment. Now what? A `model.tar.gz` file in a bucket does not answer questions, classify transactions, or generate predictions. To make the model useful, you need to put it behind a service that accepts requests and returns results. That service is a **SageMaker Real-time Inference Endpoint**.

In the AIML Foundations module, the deployment stage was described as the point where a trained model becomes integrated into applications or batch pipelines. In SageMaker, a real-time endpoint is the most common integration pattern: your application sends a data payload over HTTPS, the endpoint runs it through the model, and the response comes back in milliseconds. Think of it as opening a service window in the factory wall -- customers (or applications) walk up, hand over raw ingredients, and receive the finished prediction.

This reading walks you through the three-object deployment pattern in the SageMaker console, covers alternative inference modes, and ends with the mandatory cleanup steps that prevent billing surprises.

## Core Concepts

### The three-object deployment pattern

SageMaker deploys models through three objects that build on each other. Understanding this layered structure helps you debug deployment issues and manage endpoints over time.

**Model** is a pointer to your `model.tar.gz` artifact in S3 and the inference container image that knows how to load and serve it. Creating a Model object does not launch any compute -- it simply registers the artifact-container pair so SageMaker knows what to run.

**Endpoint Configuration** defines the infrastructure: which Model to deploy, on which instance type, how many instances, and optional settings like data capture for monitoring. Think of it as the blueprint for the service window.

**Endpoint** is the live, running service. When you create an Endpoint from an Endpoint Configuration, SageMaker provisions the instances, downloads the model artifact, loads it into the container, and starts accepting requests. The Endpoint has a unique URL that applications use to send predictions.

```
Model (artifact + container)
    |
    v
Endpoint Configuration (instance type, count, settings)
    |
    v
Endpoint (live HTTPS service accepting requests)
```

### Creating a Model in the console

1. **Navigate to Models.** In the SageMaker sidebar, click **Inference > Models**, then click **Create model**.
2. **Model name:** Enter a descriptive name (e.g., `fraud-rf-v3`).
3. **IAM role:** Select the execution role that has permissions to read from S3 and write to CloudWatch.
4. **Container definition:**
   - **Container image:** The ECR URI for the inference container. For scikit-learn Script Mode, use the same managed container image you used for training (e.g., the scikit-learn 1.2-1 image URI).
   - **Model data URL:** The S3 path to your `model.tar.gz` (e.g., `s3://my-bucket/training-output/rf-classifier/output/model.tar.gz`). This is the same artifact you located in the *Model Artifacts & S3 Storage* reading.
5. Click **Create model**. The Model appears in the Models list immediately. No compute is running yet.

> **Tip:** If you registered the model in the Model Registry, you can find the artifact S3 URI and container image on the version details page. Copy them from there to avoid typos.

### Creating an Endpoint Configuration in the console

1. **Navigate to Endpoint configurations.** In the sidebar, click **Inference > Endpoint configurations**, then click **Create endpoint configuration**.
2. **Configuration name:** Enter a name (e.g., `fraud-rf-v3-config`).
3. **Add model:**
   - Click **Add model** and select the Model you just created (`fraud-rf-v3`).
   - **Instance type:** Select `ml.m5.xlarge` to stay within Free Tier limits.
   - **Initial instance count:** Set to `1` for single-instance deployment.
4. **Data capture (optional):** Enable this to log a percentage of incoming requests and responses to S3. Data capture feeds SageMaker Model Monitor, which you learned about in the *SageMaker ML Lifecycle* reading. You can skip this for learning exercises.
5. Click **Create endpoint configuration**. Still no compute running -- you have only defined the blueprint.

### Creating an Endpoint in the console

1. **Navigate to Endpoints.** In the sidebar, click **Inference > Endpoints**, then click **Create endpoint**.
2. **Endpoint name:** Enter a name (e.g., `fraud-rf-v3-endpoint`).
3. **Endpoint configuration:** Select the configuration you just created (`fraud-rf-v3-config`). The form shows which Model, instance type, and instance count will be used.
4. Click **Create endpoint**. SageMaker now provisions compute. The status shows **Creating** and transitions to **InService** when the endpoint is ready to accept requests. This typically takes 5-10 minutes.
5. **Verify.** Once the status shows **InService**, the endpoint is live. The endpoint details page shows:
   - The endpoint ARN (used in SDK invocations).
   - The creation time and current status.
   - The endpoint configuration and model details.
   - CloudWatch monitoring metrics (invocations, latency, errors).

### Alternative inference modes

Real-time endpoints are the default, but SageMaker offers other modes for different use cases:

**Batch Transform** processes an entire dataset at once, reading from S3 and writing results back to S3. There is no persistent endpoint -- SageMaker provisions compute, processes all records, and tears down. Use batch transform when you need predictions on a large dataset and do not need sub-second latency. In the console, batch transform lives under **Inference > Batch transform jobs**.

**Serverless Inference** provisions compute on demand and scales to zero when idle. You do not choose an instance type; instead you configure maximum concurrency and memory size. SageMaker charges only for the compute time used per request. This is ideal for workloads with infrequent or unpredictable traffic where paying for an always-on endpoint is wasteful. In the console, you configure serverless inference within the Endpoint Configuration by selecting the **Serverless** option instead of specifying an instance type.

**Asynchronous Inference** handles large payloads (up to 1 GB) that take longer to process. Requests are queued, and the client receives a response location (an S3 path) where results will be written when processing completes. This is useful for workloads like document processing or large image analysis where synchronous responses would time out. In the console, async endpoints are configured through the Endpoint Configuration with an **Async inference config** section that specifies an S3 output path and optional SNS notification topics.

**Multi-Model Endpoints (MME)** host multiple models on a single instance. Instead of deploying one endpoint per model (which multiplies costs), you deploy one endpoint and tell it which model to load at invocation time. SageMaker dynamically loads and unloads models based on traffic patterns. MMEs are configured in the Endpoint Configuration by enabling the multi-model option and pointing to an S3 prefix that contains multiple `model.tar.gz` files.

| Mode | Latency | Cost Model | Best For |
| :--- | :--- | :--- | :--- |
| Real-time | Milliseconds | Per-instance-hour (always on) | Low-latency, steady traffic |
| Batch Transform | Minutes to hours | Per-instance-hour (job duration only) | Large datasets, offline predictions |
| Serverless | Seconds (cold start) | Per-request + compute time | Infrequent, unpredictable traffic |
| Asynchronous | Seconds to minutes | Per-instance-hour | Large payloads, long processing |
| Multi-Model | Milliseconds (warm) | Per-instance-hour (shared) | Many models, cost consolidation |

For this curriculum, you will primarily work with real-time endpoints. The alternative modes are introduced here so you can recognize when each is appropriate.

### Mandatory cleanup

Every real-time endpoint incurs charges for every second it runs. You must delete the endpoint, endpoint configuration, and model when you are done.

**Cleanup via the console:**

1. Navigate to **Inference > Endpoints**. Select your endpoint and click **Actions > Delete**. Confirm.
2. Navigate to **Inference > Endpoint configurations**. Select your configuration and click **Actions > Delete**. Confirm.
3. Navigate to **Inference > Models**. Select your model and click **Actions > Delete**. Confirm.

Delete in this order: endpoint first (stops billing immediately), then configuration, then model.

**Cleanup via code:**

```python
predictor.delete_endpoint()
predictor.delete_model()
```

Or using `boto3`:

```python
sm_client.delete_endpoint(EndpointName="fraud-rf-v3-endpoint")
sm_client.delete_endpoint_config(EndpointConfigName="fraud-rf-v3-config")
sm_client.delete_model(ModelName="fraud-rf-v3")
```

> **Critical reminder:** After cleanup, verify in the console that the endpoint is gone. Navigate to **Inference > Endpoints** and confirm the list no longer shows your endpoint. An endpoint in **InService** status is costing money. Check your *Billing & Cost Management* console to confirm no unexpected charges.

### SDK deployment shortcut

The SDK provides a one-line shortcut that creates all three objects (Model, Endpoint Configuration, Endpoint) at once:

```python
predictor = estimator.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.xlarge",
)
```

This is convenient, but understanding the three-object pattern helps you troubleshoot when things go wrong. If the endpoint fails to create, you can check each object independently in the console.

## Connecting to Practice

This reading teaches you how to deploy a model to a live endpoint. In the *Real-time Inference Endpoints Video*, you will see a live deployment and cleanup walkthrough. The next reading, *Invoking Endpoints*, shows how to send data to your endpoint and receive predictions. In the module assignment, you will deploy, invoke, and clean up an endpoint as part of a complete workflow.

The most useful thing you can do right now is open the SageMaker console and click through **Inference > Models**, **Endpoint configurations**, and **Endpoints** to see what exists in your account. If any endpoints show **InService** that you did not intend to keep, delete them immediately. Building the cleanup habit now prevents costly surprises.

---

## Further Learning & Resources

**Documentation and reading**

- **[Deploy Models for Real-time Inference](https://docs.aws.amazon.com/sagemaker/latest/dg/realtime-endpoints.html)** - *Docs*: The official guide covering the three-object deployment pattern, instance selection, and endpoint management.
- **[SageMaker Inference Options](https://docs.aws.amazon.com/sagemaker/latest/dg/deploy-model.html)** - *Docs*: A comparison of all inference modes (real-time, batch, serverless, async) with guidance on when to use each.

**Interactive practice**

- **[AWS Hands-On: Deploy a Model](https://aws.amazon.com/getting-started/hands-on/machine-learning-tutorial-deploy-model-to-real-time-inference-endpoint/)** - *Interactive*: A free guided lab walking through endpoint creation, testing, and cleanup in your own console.
- **[SageMaker Examples - Inference](https://github.com/aws/amazon-sagemaker-examples/tree/main/sagemaker-python-sdk)** - *Interactive*: Runnable notebooks demonstrating real-time, batch, and serverless deployment patterns.
