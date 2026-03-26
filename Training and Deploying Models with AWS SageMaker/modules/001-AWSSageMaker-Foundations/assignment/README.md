# AWSSageMaker-Foundations Lab

## Scenario

You are a cloud ML engineer joining **FraudShield**, a fintech company building fraud detection systems. Before the data science team can train and deploy models, you need to set up the SageMaker environment: create a Studio workspace, configure secure IAM roles, and prove that the platform works by deploying a pre-built model from JumpStart.

This lab walks you through the AWS Management Console step by step. By the end, you will have created a functioning SageMaker environment, deployed your first model, and cleaned everything up to avoid billing surprises.

---

## Learning Objectives

By completing this lab you will demonstrate the ability to:

1. Create a SageMaker Studio Domain and launch the Studio IDE
2. Configure user profiles for different team members
3. Identify the differences between Studio, Studio Classic, and Canvas
4. Inspect IAM execution roles, trust policies, and permission boundaries
5. Create a least-privilege IAM role with scoped S3 access
6. Deploy a pre-built model from SageMaker JumpStart
7. Clean up all resources to prevent ongoing charges

---

## Prerequisites

- An AWS account with console access (Free Tier eligible)
- A modern web browser (Chrome, Firefox, or Edge recommended)
- Completion of the AWSSageMaker-Foundations readings and lecture

---

## Milestones

Work through these guides in order. Each guide includes checkpoints noting what you should be prepared to show or explain during your presentation.

| # | Guide | Estimated Time | What You Build |
|---|-------|---------------|----------------|
| 1 | [Create a SageMaker Studio Domain](console_guides/01_create_studio_domain.md) | 15 min | Studio Domain + default execution role |
| 2 | [Explore Studio and Create User Profiles](console_guides/02_explore_studio_and_profiles.md) | 15 min | Second user profile for the analyst team |
| 3 | [Inspect IAM Roles and Policies](console_guides/03_inspect_iam_roles.md) | 20 min | Understanding of trust policies and permissions |
| 4 | [Create a Least-Privilege IAM Role](console_guides/04_create_least_privilege_role.md) | 20 min | Custom IAM role with scoped S3 access |
| 5 | [Deploy a JumpStart Pre-built Model](console_guides/05_deploy_jumpstart_model.md) | 20 min | Live inference endpoint |
| 6 | [Clean Up All Resources](console_guides/06_cleanup.md) | 10 min | Verified clean environment |

**Total estimated time:** ~100 minutes

---

## Presentation Deliverables

When presenting your work, be prepared to show and explain:

1. Your Studio Domain details (name, region, status)
2. Both user profiles and which execution role each uses
3. The trust policy of the auto-created execution role and why it matters
4. Your custom least-privilege IAM role and how it differs from the auto-created one
5. The JumpStart model you deployed and a sample prediction result
6. Evidence that all resources have been cleaned up (no InService endpoints)

---

## Important Reminders

- **Free Tier:** Use `ml.m5.xlarge` or smaller for any endpoint deployments. Do not select GPU instances.
- **Region Consistency:** Stay in the same AWS region (e.g., `us-east-1`) for all steps.
- **Cleanup Is Mandatory:** You will be charged for running endpoints. Always complete the cleanup guide before stepping away.
- **Do Not Skip Steps:** Each guide builds on the previous one. Skipping ahead may leave you without required resources.
