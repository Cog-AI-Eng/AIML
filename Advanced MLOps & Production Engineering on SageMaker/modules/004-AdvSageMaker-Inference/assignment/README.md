# Advanced SageMaker Inference Lab

## Scenario
FraudShield Risk Analytics has moved beyond model development and now faces a critical production challenge: serving fraud predictions at scale across multiple channels. The payments team needs sub-second predictions for real-time card swipes, the compliance team needs nightly batch scoring of historical transactions, and the data science team wants to A/B test three regional fraud models behind a single endpoint. Your task is to deploy the FraudShield model using every inference pattern SageMaker offers.

As the MLOps engineer, you will deploy serverless and asynchronous endpoints for cost-efficient workloads, run batch transform jobs for offline scoring, consolidate multiple model variants behind a multi-model endpoint, and chain preprocessing with prediction in a serial inference pipeline. By the end of this lab, FraudShield will have a complete inference architecture capable of handling any serving pattern the business demands.

---

## Learning Objectives
By completing this lab you will demonstrate the ability to:
1. Deploy a trained model as a Serverless Inference endpoint and invoke it with test payloads
2. Configure an Async Inference endpoint with S3 output and optional SNS notification
3. Run a Batch Transform job on historical transaction data stored in S3
4. Deploy a Multi-Model Endpoint serving multiple model artifacts from a shared S3 prefix
5. Build a serial inference pipeline chaining a preprocessor container with a prediction container
6. Clean up all inference resources to avoid unnecessary charges

---

## Prerequisites
- AWS account with SageMaker full access and S3 read/write permissions
- A trained FraudShield model artifact (`model.tar.gz`) stored in S3
- Familiarity with SageMaker model creation and endpoint concepts
- Completed prior ASM labs (Modules 1-3) or equivalent experience
- An S3 bucket named `sagemaker-fraudshield-<account-id>` in us-east-1

---

## Milestones

| # | Guide | Estimated Time | What You Build |
|---|-------|---------------|----------------|
| 1 | [Deploy Serverless Endpoint](console_guides/01_deploy_serverless_endpoint.md) | 25 min | A Serverless Inference endpoint with configured memory and concurrency |
| 2 | [Deploy Async Endpoint](console_guides/02_deploy_async_endpoint.md) | 25 min | An Async Inference endpoint with S3 output and SNS notification |
| 3 | [Run Batch Transform](console_guides/03_run_batch_transform.md) | 20 min | A Batch Transform job scoring historical transactions |
| 4 | [Deploy Multi-Model Endpoint](console_guides/04_deploy_multi_model_endpoint.md) | 30 min | A Multi-Model Endpoint hosting 3 regional fraud models |
| 5 | [Build Serial Inference Pipeline](console_guides/05_build_serial_inference_pipeline.md) | 30 min | A 2-container inference pipeline with preprocessing and prediction |
| 6 | [Cleanup](console_guides/06_cleanup.md) | 15 min | All resources deleted |
| SDK | [SDK Inference Lab](notebooks/sdk_inference_lab.ipynb) | 50 min | Deploy Serverless, Batch Transform, and Multi-Model Endpoints programmatically using the SageMaker Python SDK and boto3 |

**Total estimated time:** ~195 minutes (console guides ~145 min + SDK notebook ~50 min)

---

## Presentation Deliverables
1. Screenshot of the Serverless Endpoint in "InService" status with memory and concurrency configuration visible
2. Screenshot of the Async Inference output file retrieved from S3
3. Screenshot of the completed Batch Transform job showing input/output S3 paths
4. Screenshot of the Multi-Model Endpoint successfully invoked with a TargetModel parameter
5. Screenshot of the serial inference pipeline endpoint returning an end-to-end prediction
6. Screenshot confirming all endpoints, models, and configurations have been deleted

---

## Important Reminders
- **Free Tier:** Use ml.m5.xlarge or smaller. No GPU instances.
- **Region Consistency:** Stay in us-east-1.
- **Cleanup Is Mandatory:** Always complete the cleanup guide.
- **Do Not Skip Steps:** Each guide builds on the previous one.
