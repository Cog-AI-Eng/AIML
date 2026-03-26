# IAM & Least-Privilege Practices

**Estimated Time:** 10 Minutes

## Introduction

In the *Ecosystem & Core Services* reading you learned that SageMaker never acts with your personal AWS credentials. Instead, every training job, endpoint, and notebook runs under an **IAM Execution Role** that you define. The *Studio Domains & Profiles* reading took this further: when you created a Domain, you assigned a default execution role, and each User Profile can optionally override it. But what exactly *is* a role, how does it get its permissions, and how do you make sure it has only what it needs?

That question is the core of **least-privilege access control**, and it is one of the most important operational habits in cloud ML. Think of it this way: if your ML lifecycle is a factory (data in, model out), IAM is the badge system. Every worker -- human or automated -- wears a badge that opens only the doors they need. A data scientist's badge opens the notebook room and the training floor but not the billing office. A deployment pipeline's badge opens the endpoint delivery dock but cannot touch raw customer data. If someone's badge is stolen or misconfigured, the blast radius is limited to the doors that badge can open.

In the Applied ML Foundations content, you saw that reproducibility requires consistent environments and version-controlled code. IAM adds a third pillar: **governed permissions**. A model that trains successfully today but fails tomorrow because someone widened a role and accidentally exposed a different S3 bucket is not reproducible in any meaningful sense. Least-privilege keeps the permission surface stable and auditable.

This reading walks you through the IAM console, explains the anatomy of a SageMaker Execution Role, and shows you how to build one with only the permissions your workloads need.

## Core Concepts

### IAM building blocks

Before you touch the console, here are the four objects you will work with:

**Users** represent individual people or applications that sign in to AWS. In a learning environment, your AWS account likely has one IAM user (you). In a team setting, each person has their own user with unique credentials.

**Roles** are identities that AWS *services* can assume. A role is not tied to a person; it is tied to a trust relationship that says "this service is allowed to wear this badge." SageMaker assumes a role every time it needs to interact with other AWS services on your behalf. This is the Execution Role you have seen in earlier readings.

**Policies** are JSON documents that define what actions are allowed or denied on which resources. A policy is attached to a user or a role. For example, a policy might say "allow `s3:GetObject` on `arn:aws:s3:::my-training-bucket/*`" -- meaning the bearer can read objects from that specific bucket, but nothing else.

**Trust policies** define *who* can assume a role. For a SageMaker Execution Role, the trust policy says "the SageMaker service (`sagemaker.amazonaws.com`) is allowed to assume this role." Without that trust relationship, SageMaker cannot use the role, no matter what permissions the role has.

### Finding IAM in the console

1. **Sign in** to the AWS Management Console at [console.aws.amazon.com](https://console.aws.amazon.com).
2. **Search for IAM.** Click the search bar at the top and type `IAM`. Click **IAM** under Services.
3. **The IAM Dashboard** appears. On the left sidebar you will see sections for **Users**, **User groups**, **Roles**, **Policies**, and **Identity providers**. For SageMaker work, you will spend most of your time in **Roles** and **Policies**.

> **Tip:** The IAM dashboard shows a "Security recommendations" panel at the top. Check it periodically -- it flags issues like unused credentials or overly permissive policies.

### Anatomy of a SageMaker Execution Role

When you created a Domain with Quick setup in the previous topic, AWS auto-generated a role for you. Let us find it and examine what it contains.

1. **Navigate to Roles.** In the IAM sidebar, click **Roles**. You will see a searchable list of all roles in your account.
2. **Search for SageMaker.** Type `SageMaker` in the search bar. You should see one or more roles with names like `AmazonSageMaker-ExecutionRole-YYYYMMDDTHHMMSS` (the timestamp from when it was created).
3. **Click the role name** to open its details page. You will see three tabs that matter:

**Summary tab** shows the role ARN (Amazon Resource Name), the creation date, and the trust relationship. The trust relationship for a SageMaker role looks like this:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "sagemaker.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
```

This says: "The SageMaker service is allowed to assume this role." If you ever need to let another service (like Lambda) also assume the role, you would add its service principal here.

**Permissions tab** lists the policies attached to the role. The auto-generated role typically has `AmazonSageMakerFullAccess`, a broad AWS-managed policy that grants permissions across S3, ECR, CloudWatch, and many other services. This is convenient for learning but violates least-privilege in production.

**Tags tab** shows any key-value metadata attached to the role. Tags become important for advanced access control patterns (covered briefly later in this reading).

### Creating a least-privilege role step by step

For production workloads (and as a best practice even in learning environments), you should create a custom role with only the permissions your specific workflow needs. Here is how to do it in the console.

1. **Go to Roles > Create role.** In the IAM sidebar, click **Roles**, then click the **Create role** button.

2. **Select the trusted entity.** Choose **AWS service** as the trusted entity type. In the "Use case" dropdown, search for and select **SageMaker**. This automatically generates the correct trust policy. Click **Next**.

3. **Attach permissions policies.** Instead of selecting `AmazonSageMakerFullAccess`, search for and attach narrower policies that match your workload. Common choices for a training-only role:
   - `AmazonS3ReadOnlyAccess` -- allows reading training data from S3 (you can further restrict this to a specific bucket using a custom policy).
   - `CloudWatchLogsFullAccess` -- allows writing training logs (SageMaker streams logs to CloudWatch during training).

   If the existing AWS-managed policies are still too broad, you can create a **custom policy** (see next section). For now, select the narrowest managed policies that cover your needs and click **Next**.

4. **Name and review.** Give the role a descriptive name like `SageMaker-Training-LeastPrivilege`. Add an optional description such as "Custom execution role for SageMaker training jobs with read-only S3 and CloudWatch logging." Review the trust policy and attached permissions, then click **Create role**.

5. **Copy the role ARN.** After creation, click into the role and copy its ARN from the Summary tab. You will paste this ARN into SageMaker when you configure training jobs or update your Domain's execution role.

### Writing a custom inline policy

When AWS-managed policies are too broad, you can write a custom policy that restricts access to specific resources. Here is an example that grants read access to a single S3 bucket and write access to a specific output prefix:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::my-training-data-bucket",
        "arn:aws:s3:::my-training-data-bucket/*"
      ]
    },
    {
      "Effect": "Allow",
      "Action": [
        "s3:PutObject"
      ],
      "Resource": "arn:aws:s3:::my-training-data-bucket/output/*"
    }
  ]
}
```

To attach this in the console:
1. Open the role you created.
2. On the **Permissions** tab, click **Add permissions > Create inline policy**.
3. Switch to the **JSON** tab and paste the policy.
4. Click **Review policy**, give it a name like `S3-TrainingBucket-Scoped`, and click **Create policy**.

The role now has precise access: it can read from the training bucket and write only to the `output/` prefix. Nothing else.

### The principle of least privilege in practice

Least-privilege is not about making things harder. It is about reducing the blast radius when something goes wrong. A few practical guidelines:

- **Start narrow, widen as needed.** Create roles with minimal permissions and add policies only when a specific action fails with an "Access Denied" error. CloudTrail logs (searchable in the console under **CloudTrail > Event history**) show exactly which action was denied and on which resource.
- **One role per workload type.** Do not reuse the same role for training, inference, and data processing. A training role needs S3 read and CloudWatch write. An inference role needs S3 read (for model artifacts) and endpoint invocation. Separating them limits what a compromised or misconfigured role can do.
- **Avoid `*` in resource ARNs.** Policies that say `"Resource": "*"` grant access to every resource of that type in your account. Always scope to specific bucket names, prefixes, or resource ARNs when possible.

### Tag-based access control (advanced pattern)

For teams managing many SageMaker resources, AWS supports **tag-based access control** using IAM condition keys. The idea is simple: tag your SageMaker resources (training jobs, endpoints, models) with key-value pairs like `project:fraud-detection` or `team:data-science`, then write IAM policies that restrict actions to resources carrying specific tags.

For example, a policy condition like `"Condition": {"StringEquals": {"sagemaker:ResourceTag/project": "fraud-detection"}}` would allow a role to manage only resources tagged with that project. This pattern is a stretch goal in this curriculum and is most valuable in multi-team environments where you need fine-grained resource isolation without creating separate AWS accounts.

### SDK and CLI equivalents

Everything you did in the console can be automated. The AWS CLI command `aws iam create-role` creates a role with a trust policy, and `aws iam attach-role-policy` attaches managed policies. The `boto3` library's `iam` client offers the same operations programmatically. Automation matters when you need to replicate role configurations across environments (development, staging, production), but always build and verify roles in the console first so you understand what each policy grants.

## Connecting to Practice

This reading gives you the conceptual and practical foundation for SageMaker IAM roles. In the module lecture, you will create a least-privilege role and use it to run a SageMaker training job end to end. In the module assignment, you will configure roles as part of a complete workflow.

The most useful thing you can do right now is open the IAM console, navigate to **Roles**, and search for any SageMaker execution roles in your account. Click into one and examine its trust policy and attached permissions. Ask yourself: does this role have more access than it needs? If it has `AmazonSageMakerFullAccess`, the answer is almost certainly yes. Understanding what that policy grants -- and how to replace it with something narrower -- is the foundation for every secure SageMaker deployment.

---

## Further Learning & Resources

**Documentation and reading**

- **[SageMaker Roles](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-roles.html)** - *Docs*: The official reference for SageMaker execution roles, covering required permissions for each SageMaker API action.
- **[IAM Best Practices](https://docs.aws.amazon.com/IAM/latest/UserGuide/best-practices.html)** - *Docs*: AWS's own guide to least-privilege design, role separation, and policy management across all services.

**Interactive practice**

- **[AWS IAM Policy Simulator](https://policysim.aws.amazon.com/)** - *Interactive*: A free browser tool that lets you test IAM policies against specific API actions without actually making calls, invaluable for verifying that your custom policies grant exactly what you intend.
- **[AWS Skill Builder - Security Fundamentals](https://explore.skillbuilder.aws/learn/course/internal/view/elearning/48/aws-security-fundamentals-second-edition)** - *Interactive*: Free self-paced course covering IAM concepts, policies, and access control patterns with hands-on exercises.
