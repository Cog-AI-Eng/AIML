# Cross-account Sharing

**Estimated Time:** 10 Minutes

## Introduction

In enterprise environments, ML workloads rarely fit inside a single AWS account. A common architecture separates concerns across accounts: a **data account** owns the raw datasets and Feature Store, a **training account** runs experiments and training jobs, a **deployment account** hosts production endpoints, and a **governance account** provides centralized monitoring and audit. This separation enforces security boundaries and simplifies billing attribution.

SageMaker supports cross-account sharing for several key resources: Feature Store Feature Groups, Model Registry model packages, and S3 data assets. This reading covers the architectural patterns and AWS mechanisms (IAM, AWS RAM, S3 policies) that enable cross-account ML workflows while maintaining the reproducibility and governance practices from the previous topics.

## Core Concepts

### Why separate accounts?

| Account | Purpose | SageMaker Resources |
| :--- | :--- | :--- |
| Data | Owns raw data, Feature Store, Data Wrangler | Feature Groups, S3 buckets |
| Training | Runs experiments, training jobs, HPO | Experiments, Training Jobs, Processing Jobs |
| Deployment | Hosts production endpoints, monitors performance | Endpoints, Model Monitor |
| Governance | Centralized Model Registry, lineage queries, compliance | Model Registry, Lineage Graph |

Benefits of this architecture:
- **Blast radius control:** A misconfigured training job cannot affect production endpoints because they are in different accounts with separate IAM boundaries.
- **Cost isolation:** Each team's compute costs are attributed to their account, simplifying chargebacks.
- **Compliance:** The governance account provides a single pane of glass for audit, separate from operational accounts.

### Cross-account Feature Store with AWS RAM

AWS Resource Access Manager (RAM) enables sharing Feature Groups across accounts without copying data:

1. In the **data account**, navigate to **AWS RAM > Create resource share**.
2. Select the Feature Group ARN as the shared resource.
3. Specify the target account IDs (training account, governance account).
4. In the **target account**, accept the resource share in the RAM console.
5. The target account can now query the shared Feature Group's Offline Store using Athena and ingest records via the `PutRecord` API (if write access is granted).

Lineage entities created in the data account (e.g., Feature Group Artifacts) are visible in the governance account's lineage graph, enabling cross-account provenance queries.

### Cross-account Model Registry

The SageMaker Model Registry supports cross-account model package sharing:

1. In the **training account**, register a model in a Model Package Group and set its status to `PendingManualApproval`.
2. Apply a **resource policy** to the Model Package Group that grants the deployment account's role permission to describe and deploy model packages.
3. In the **deployment account**, reference the model package ARN from the training account when creating a model and deploying to an endpoint.
4. The **governance account** can be granted read access to all Model Package Groups across accounts for centralized approval workflows and audit.

### Cross-account S3 access

Training jobs in the training account frequently need to read data from S3 buckets in the data account:

1. In the **data account**, add a **bucket policy** that grants the training account's SageMaker execution role `s3:GetObject` and `s3:ListBucket` on the relevant prefixes.
2. In the **training account**, ensure the execution role's IAM policy includes `s3:GetObject` on the cross-account bucket ARN.
3. Reference the cross-account S3 URI directly in the training job's input data configuration.

For model artifacts, the reverse pattern applies: the training account writes model artifacts to its own S3 bucket, and the deployment account's execution role is granted `s3:GetObject` on that bucket.

### IAM role design for cross-account workflows

Each cross-account interaction requires careful IAM configuration:

- **Trust policies:** The execution role in the target account must trust the SageMaker service principal (`sagemaker.amazonaws.com`), not the source account's users. Cross-account access is controlled by resource policies (S3 bucket policies, Model Registry resource policies, RAM shares), not by assuming roles across accounts.
- **Least privilege:** Grant only the specific actions needed. A deployment account's role needs `sagemaker:CreateModel` and `sagemaker:CreateEndpoint` on Model Registry resources from the training account, but does not need `sagemaker:CreateTrainingJob`.
- **Condition keys:** Use `aws:PrincipalOrgId` conditions to restrict access to accounts within your AWS Organization, preventing accidental sharing with external accounts.

### Maintaining reproducibility across accounts

Cross-account workflows add complexity to the reproducibility patterns from the previous topic:

- **Data lineage:** Ensure the governance account has lineage read access to all operational accounts. Cross-account lineage queries work through RAM-shared Feature Groups and cross-account S3 access.
- **Code versioning:** Store pipeline definitions in a shared Git repository accessible to all accounts. Use the same commit hash across training and deployment.
- **Experiment sharing:** Experiments in the training account can be queried from the governance account through cross-account IAM. Export Experiment analytics to a shared S3 bucket for centralized reporting.

## Connecting to Practice

This topic completes the Tracking module by extending governance to multi-account architectures. You now have a full tracking toolkit: Experiments for operational comparison, Lineage for provenance, Feature Store for centralized features, Reproducibility Patterns for guarantees, and Cross-account Sharing for enterprise-scale governance. The module lecture will demonstrate a two-account setup (training + governance) with shared Feature Store and Model Registry. The assignment will require you to configure cross-account S3 access and Model Registry sharing using IAM policies.

## Further Learning & Resources

**Documentation and reading**

- **[Cross-account Model Registry](https://docs.aws.amazon.com/sagemaker/latest/dg/model-registry-cross-account.html)** - *Docs*: Step-by-step guide for sharing model packages across accounts.
- **[Cross-account Feature Store](https://docs.aws.amazon.com/sagemaker/latest/dg/feature-store-cross-account.html)** - *Docs*: Configuration for sharing Feature Groups using AWS RAM.
- **[Multi-account ML Best Practices](https://docs.aws.amazon.com/whitepapers/latest/organizing-your-aws-environment/ml-environments.html)** - *Whitepaper*: AWS Well-Architected guidance for multi-account ML architectures.

**Interactive practice**

- **[Multi-account SageMaker Workshop](https://catalog.workshops.aws/sagemaker-multi-account/en-US)** - *Interactive*: Hands-on lab demonstrating cross-account training, registry, and deployment workflows.
