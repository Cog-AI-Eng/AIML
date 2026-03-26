# Guide 3: Inspect IAM Roles and Policies

Every action SageMaker performs on your behalf -- reading training data from S3, writing model artifacts, creating CloudWatch logs -- is governed by the IAM **execution role** attached to your domain or user profile. Understanding what this role allows (and what it should not) is essential before running any workloads.

---

## Steps

### Step 1 -- Find the Auto-Created Execution Role

1. Open the **AWS Management Console** and navigate to **IAM** (search for "IAM" in the top search bar).
2. In the left navigation, click **Roles**.
3. In the search box, type `AmazonSageMaker-ExecutionRole` to filter.
4. You should see the role that was auto-created when you set up the Studio Domain. Click on it to open the role details.

### Step 2 -- Examine the Trust Policy

1. Click the **Trust relationships** tab.
2. Read the trust policy document. It should look similar to this:

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

3. Understand each part:
   - **Principal:** The SageMaker service (`sagemaker.amazonaws.com`) is the entity allowed to assume this role. Your notebook, training jobs, and endpoints all run as this identity.
   - **Action:** `sts:AssumeRole` means SageMaker can temporarily take on the permissions of this role.
   - **Effect:** `Allow` permits the action.

4. Note: This trust policy does NOT give any permissions to S3, ECR, or CloudWatch by itself. It only says *who can wear this hat*. The actual permissions come from the attached policies.

### Step 3 -- Examine the Attached Permissions

1. Click the **Permissions** tab.
2. You should see `AmazonSageMakerFullAccess` attached as a managed policy. Click on it to expand.
3. Scroll through the policy document. Notice:
   - It grants access to **many** AWS services: S3, ECR, CloudWatch, IAM (PassRole), Lambda, Glue, and more.
   - Many resources are set to `"Resource": "*"` -- meaning access to ALL resources of that type in the account.
4. This is the **problem**: the Quick Setup role is intentionally broad for convenience, but in production this violates the principle of **least privilege**.

### Step 4 -- Identify the Security Risks

Look for these specific patterns in the policy:

| Pattern | Risk |
|---------|------|
| `"s3:*"` with `"Resource": "*"` | Can read/write/delete ANY S3 bucket in the account |
| `"iam:PassRole"` | Can pass IAM roles to other services, potential privilege escalation |
| `"logs:*"` | Full CloudWatch access |
| `"ecr:*"` | Full container registry access |

In Guide 4, you will create a role that scopes these permissions down to only what a training job actually needs.

### Step 5 -- Check for Tags

1. Click the **Tags** tab on the role.
2. Note any tags that were auto-applied (e.g., `sagemaker:domain-arn`).
3. Tags are how organizations track which roles belong to which SageMaker domains, which is important for cost allocation and security auditing.

---

## Presentation Checkpoint

Be prepared to show:
- The **trust policy** of the auto-created execution role and explain all three components (Principal, Action, Effect)
- The **AmazonSageMakerFullAccess** policy and at least two examples of overly broad permissions
- Explain: What is the difference between a **trust policy** (who can assume the role) and a **permissions policy** (what the role can do)?
- Explain: Why is `"Resource": "*"` dangerous in a production environment?

---

## Key Concepts

- **Trust Policy:** Defines *who* can assume a role. For SageMaker roles, the trusted principal is always `sagemaker.amazonaws.com`.
- **Permissions Policy:** Defines *what* the role can do once assumed. This is where S3 access, CloudWatch logging, and SageMaker actions are granted.
- **Least Privilege:** The security principle that a role should have only the minimum permissions necessary to perform its task. The auto-created role violates this by using `Resource: *`.
- **`sts:AssumeRole`:** The API call that allows a service to temporarily take on the identity and permissions of an IAM role.

---

## AIML Connection

The *IAM & Least-Privilege Practices* reading covered why scoped access matters: in a real ML pipeline, training jobs should only access the specific S3 bucket containing training data and the specific output prefix for artifacts. Broad access creates risk -- a misconfigured training script could accidentally overwrite production data in another bucket.
