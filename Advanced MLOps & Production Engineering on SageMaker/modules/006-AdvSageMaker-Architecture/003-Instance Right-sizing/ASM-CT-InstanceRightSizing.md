# Instance Right-sizing

**Estimated Time:** 10 Minutes

## Introduction

Choosing the right instance type is one of the highest-impact cost optimization decisions in SageMaker. An oversized instance wastes money on unused capacity. An undersized instance causes out-of-memory errors, slow training, or inference latency violations. This reading covers the instance type families available for SageMaker, how to evaluate your workload's resource requirements, and the tools SageMaker provides for data-driven instance selection.

## Core Concepts

### SageMaker instance type families

| Family | Prefix | Optimized For | Typical Use |
| :--- | :--- | :--- | :--- |
| General purpose | `ml.m5`, `ml.m6i` | Balanced CPU, memory, networking | Notebooks, scikit-learn/XGBoost training, light inference |
| Compute optimized | `ml.c5`, `ml.c6i` | High CPU throughput | CPU-intensive preprocessing, large-scale batch transform |
| Memory optimized | `ml.r5` | Large memory footprint | Large datasets, feature engineering, K-Means on high-dimensional data |
| Accelerated computing | `ml.p3`, `ml.p4d`, `ml.g5` | GPU (NVIDIA Tesla/A100) | Deep learning training and inference |
| Inference optimized | `ml.inf1`, `ml.inf2` | AWS Inferentia chips | High-throughput, cost-effective inference for deployed models |

### Right-sizing for training

To right-size a training instance:

1. **Start with `ml.m5.xlarge`:** This is the Free Tier-compatible general-purpose instance (4 vCPUs, 16 GB RAM). It handles most tabular ML training (XGBoost, linear models, small random forests) comfortably.
2. **Monitor CloudWatch metrics during training:** Check `CPUUtilization`, `MemoryUtilization`, and `DiskUtilization` in the CloudWatch console (metrics are published automatically by SageMaker training jobs).
3. **Identify the bottleneck:**
   - CPU > 90% but Memory < 50%: The job is CPU-bound. Move to a compute-optimized instance (`ml.c5.xlarge`) or increase instance size (`ml.m5.2xlarge`).
   - Memory > 90%: The job is memory-bound. Move to a memory-optimized instance (`ml.r5.xlarge`) or increase instance size.
   - CPU and Memory both low (< 30%): The instance is oversized. Move to a smaller instance.
   - GPU needed: Deep learning training (CNNs, transformers, DeepAR with large datasets) requires GPU instances. Start with `ml.g5.xlarge` (cost-effective) or `ml.p3.2xlarge` (high performance).

### Right-sizing for inference

Inference right-sizing is more critical because endpoints run continuously (unless using Serverless or Async with scale-to-zero):

1. **Start with `ml.m5.xlarge`:** Baseline for model serving.
2. **Measure inference latency and throughput:** Use `InvocationsPerInstance` and `ModelLatency` CloudWatch metrics.
3. **Load test:** Use SageMaker's built-in load testing or a tool like Locust to send realistic traffic volumes and measure p50, p95, and p99 latency.
4. **Identify overhead:**
   - If latency is acceptable at low traffic but degrades under load: scale out (more instances) rather than up (larger instances), unless the model itself is too large for the instance's memory.
   - If latency is consistently high even at low traffic: the model may benefit from a more powerful instance or inference optimization (model compilation with SageMaker Neo).

### Instance selection for cost optimization

| Decision | Recommendation |
| :--- | :--- |
| Training under 10 GB of data, tabular ML | `ml.m5.xlarge` (Free Tier) |
| Training with large datasets (10-100 GB) | `ml.m5.4xlarge` or `ml.c5.4xlarge` |
| Deep learning training | `ml.g5.xlarge` (cost-effective GPU) or `ml.p3.2xlarge` |
| Inference with steady traffic | `ml.m5.xlarge` with auto-scaling |
| Inference with bursty traffic | Serverless (no idle cost) or `ml.m5.xlarge` with aggressive auto-scaling |
| High-throughput inference | `ml.inf1.xlarge` (Inferentia, optimized for throughput/$ ) |
| HPO (many short trials) | `ml.m5.xlarge` with Spot (minimize per-trial cost) |

### Common mistakes

- **Using GPU instances for tabular ML:** XGBoost and scikit-learn models do not benefit from GPU acceleration on SageMaker (the built-in containers use CPU). GPU instances cost 5-10x more and provide no speedup.
- **Using the same instance for training and inference:** Training is a batch operation that benefits from large instances. Inference is a latency-sensitive operation that benefits from right-sized instances. A model trained on `ml.p3.2xlarge` can (and should) be deployed on `ml.m5.xlarge` if inference is CPU-only.
- **Over-provisioning for safety:** Starting with `ml.m5.4xlarge` "just in case" when `ml.m5.xlarge` would suffice wastes 4x the cost. Always start small and scale up based on metrics.

## Connecting to Practice

Instance right-sizing applies to every SageMaker workload. The next topic, *Auto-scaling Policies*, covers dynamic instance scaling for endpoints that need to handle variable traffic. The module assignment will require you to analyze CloudWatch metrics from a training job and an endpoint to recommend optimized instance types with cost justification.

## Further Learning & Resources

**Documentation and reading**

- **[SageMaker Instance Types](https://aws.amazon.com/sagemaker/pricing/)** - *Docs*: Complete list of available instance types with pricing per region.
- **[Choose the Best Instance Type](https://docs.aws.amazon.com/sagemaker/latest/dg/instance-types-az.html)** - *Docs*: AWS guidance on instance selection for different workload types.

**Interactive practice**

- **[Cost Optimization Workshop](https://catalog.workshops.aws/sagemaker-cost-optimization/en-US)** - *Interactive*: Hands-on lab covering instance selection, Spot Training, and auto-scaling configuration.
