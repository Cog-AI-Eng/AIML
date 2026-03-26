# Inference Decision Matrix

**Estimated Time:** 10 Minutes

## Introduction

SageMaker offers four inference options -- real-time endpoints, Serverless Inference, Asynchronous Inference, and Batch Transform -- each optimized for different traffic patterns, latency requirements, and cost profiles. Choosing the wrong option leads to either unnecessary cost (over-provisioning for low traffic) or poor user experience (cold starts on a latency-sensitive API).

This reading provides a structured decision matrix that you can apply to any inference use case. Rather than memorizing when to use each option, you evaluate your workload against five criteria and the matrix points you to the right choice.

## Core Concepts

### The five decision criteria

1. **Latency requirement:** How fast must the response arrive? Sub-second? Seconds? Minutes?
2. **Traffic pattern:** Is traffic steady, bursty, sporadic, or one-time?
3. **Payload size:** How large is the input data? Kilobytes? Megabytes? Gigabytes?
4. **Processing time:** How long does a single inference call take? Milliseconds? Minutes?
5. **Cost sensitivity:** Is minimizing infrastructure cost a priority, or is performance the constraint?

### The decision matrix

| Criteria | Real-time Endpoint | Serverless | Async Inference | Batch Transform |
| :--- | :--- | :--- | :--- | :--- |
| **Latency** | < 1 second | 1-30 seconds (cold start) | Minutes | Hours (job-based) |
| **Traffic** | Steady or predictable | Sporadic, low volume | Variable, queued | One-time or scheduled |
| **Max payload** | 6 MB | 6 MB | 1 GB | Unlimited |
| **Max processing** | 60 seconds | 60 seconds | 60 minutes | No limit |
| **Scale to zero** | No | Yes | Yes | N/A |
| **GPU support** | Yes | No | Yes | Yes |
| **Billing model** | Instance-hours | Compute ms + data | Instance-hours (0 when idle) | Instance-hours (job duration) |
| **Infrastructure** | Always-on instances | Managed containers | On-demand instances | Ephemeral job cluster |

### Decision flowchart

Work through these questions in order:

**Q1: Is this a one-time or scheduled bulk scoring job?**
- Yes: **Batch Transform.** You have a dataset in S3 and want predictions for every record. No endpoint needed.

**Q2: Does the input payload exceed 6 MB, or does inference take more than 60 seconds?**
- Yes: **Async Inference.** It is the only endpoint option that supports large payloads and long processing times.

**Q3: Does the use case require sub-second response times?**
- Yes: **Real-time Endpoint.** Always-on instances eliminate cold start latency.
- No, seconds are acceptable: Continue to Q4.

**Q4: Is traffic predictable and steady (> 1,000 requests/hour consistently)?**
- Yes: **Real-time Endpoint.** Always-on instances are more cost-effective than Serverless at high volumes.
- No, traffic is sporadic or unpredictable: **Serverless Inference.** Scale-to-zero eliminates idle costs.

### Common patterns in practice

**Pattern: Real-time endpoint with auto-scaling.** For a user-facing prediction API with moderate traffic variability. Configure auto-scaling with a target tracking policy on `InvocationsPerInstance`. SageMaker adds or removes instances to maintain a target invocation rate.

**Pattern: Serverless for internal tools.** An internal dashboard that triggers a prediction when an analyst clicks a button. Traffic is unpredictable (0-50 requests/day). Serverless eliminates idle costs entirely. Analysts tolerate a few seconds of cold start.

**Pattern: Async for document processing.** A contract analysis model that processes 50-page PDFs, taking 2-3 minutes per document. Clients upload PDFs to S3, invoke the Async endpoint, and receive SNS notifications when results are ready.

**Pattern: Batch Transform for nightly scoring.** An e-commerce recommendation system that scores all customers nightly. A Batch Transform job reads customer features from S3, generates recommendations, and writes the output back to S3 for the application to serve the next day.

**Pattern: Hybrid architecture.** A system that uses a real-time endpoint for online predictions during business hours and an Async endpoint with scale-to-zero for overnight processing of large analytical requests.

### Cost optimization decision

When cost is the primary concern, work backwards from traffic volume:

- **< 100 requests/day:** Serverless (pennies/month vs. ~$170/month for an always-on endpoint)
- **100-10,000 requests/day:** Evaluate Serverless vs. Real-time; the crossover depends on average inference time and memory requirements
- **> 10,000 requests/day:** Real-time with auto-scaling (the per-invocation cost of Serverless exceeds instance-hour cost at high volumes)
- **Bulk scoring with no latency requirement:** Batch Transform (pay only for the job duration)

## Connecting to Practice

This decision matrix is a tool you will use every time you deploy a model on SageMaker. The remaining topics in this module -- *Batch Transform Architecture*, *Multi-Model Endpoints*, and *Multi-Container Endpoints* -- cover advanced deployment patterns that build on the real-time endpoint foundation. The module assignment will require you to evaluate a set of inference scenarios against this matrix and justify your architecture choice for each.

## Further Learning & Resources

**Documentation and reading**

- **[Choose an Inference Option](https://docs.aws.amazon.com/sagemaker/latest/dg/deploy-model.html)** - *Docs*: AWS overview comparing all inference deployment options with use case guidance.
- **[SageMaker Inference Pricing](https://aws.amazon.com/sagemaker/pricing/)** - *Docs*: Pricing details for all inference options to support cost-based decisions.

**Interactive practice**

- **[Inference Options Workshop](https://catalog.workshops.aws/sagemaker-inference/en-US)** - *Interactive*: Hands-on lab that deploys the same model using multiple inference options and compares behavior.
