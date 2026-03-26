# Data Preparation and Feature Engineering Lab

## Scenario
FraudShield Risk Analytics has outgrown its ad-hoc data pipelines. The data-science team ingests millions of e-commerce transactions daily, and the current workflow -- downloading CSVs, transforming them in local notebooks, and manually uploading features -- cannot scale. Leadership has approved a migration to Amazon SageMaker so that data preparation, feature storage, and model prototyping all live inside a governed, cloud-native platform.

Your task is to stand up the foundational data layer for FraudShield on SageMaker. You will provision a Studio Domain with production-grade networking, build a repeatable Data Wrangler transformation flow, publish curated features to a Feature Store group, automate the entire flow through a SageMaker Pipeline, and demonstrate rapid prototyping in SageMaker Canvas -- all without writing infrastructure code.

---

## Learning Objectives
By completing this lab you will demonstrate the ability to:
1. Provision a SageMaker Studio Domain using the Standard Setup with a custom VPC and execution role.
2. Build a Data Wrangler flow that profiles, transforms, and exports tabular data.
3. Create and populate a Feature Store feature group with online and offline stores.
4. Export a Data Wrangler flow into a SageMaker Pipeline and execute it end to end.
5. Use SageMaker Canvas to perform a no-code Quick Build model on FraudShield data.
6. Tear down all provisioned resources to avoid unnecessary charges.

---

## Prerequisites
- An AWS account with AdministratorAccess or equivalent permissions.
- The FraudShield e-commerce transactions CSV uploaded to an S3 bucket (e.g., `s3://fraudshield-data-<account-id>/raw/ecommerce_transactions.csv`).
- A VPC with at least two private subnets and a security group that allows internal traffic.
- Familiarity with basic S3, IAM, and VPC concepts.

---

## Milestones

| # | Guide | Estimated Time | What You Build |
|---|-------|---------------|----------------|
| 1 | [Configure a Custom Studio Domain](console_guides/01_configure_custom_domain.md) | 25 min | A Studio Domain with custom VPC, EFS, and execution role |
| 2 | [Create a Data Wrangler Flow](console_guides/02_create_data_wrangler_flow.md) | 30 min | A repeatable transformation flow with profiling and 3+ transforms |
| 3 | [Create a Feature Group](console_guides/03_create_feature_group.md) | 20 min | An online+offline Feature Store group with ingested records |
| 4 | [Export Flow to a Pipeline](console_guides/04_export_flow_to_pipeline.md) | 20 min | A SageMaker Pipeline that runs Data Wrangler processing |
| 5 | [Canvas Quick Build](console_guides/05_canvas_quick_build.md) | 25 min | A no-code Quick Build model with prediction results |
| 6 | [Cleanup](console_guides/06_cleanup.md) | 15 min | All lab resources deleted |

**Total estimated time:** ~135 minutes

---

## Presentation Deliverables
1. Show the Studio Domain summary page with VPC, subnet, and EFS details.
2. Walk through the Data Wrangler flow and explain each transformation step.
3. Open the Feature Group and demonstrate an online store query plus the offline store S3 path.
4. Show the completed Pipeline execution graph and the processing job output in S3.
5. Present the Canvas Quick Build model metrics and a sample prediction.
6. Confirm all resources have been deleted in the cleanup guide.

---

## Important Reminders
- **Free Tier:** Use ml.m5.xlarge or smaller. No GPU instances.
- **Region Consistency:** Stay in us-east-1.
- **Cleanup Is Mandatory:** Always complete the cleanup guide.
- **Do Not Skip Steps:** Each guide builds on the previous one.
