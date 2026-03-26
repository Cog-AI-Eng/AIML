# Guide 4: Create a Least-Privilege IAM Role

Now that you have seen the overly broad auto-created role, you will create a custom role that follows the principle of least privilege. This role will grant SageMaker only the permissions it needs to read training data from a specific S3 bucket and write model artifacts to a specific prefix.

---

## Steps

### Step 1 -- Navigate to IAM Role Creation

1. In the **IAM console**, click **Roles** in the left navigation.
2. Click **Create role**.
3. For **Trusted entity type**, select **AWS service**.
4. Under **Use case**, search for and select **SageMaker**.
5. Make sure **SageMaker - Execution** is selected (this sets up the correct trust policy for `sagemaker.amazonaws.com`).
6. Click **Next**.

### Step 2 -- Skip Managed Policies

1. On the "Add permissions" page, you will see AWS suggesting `AmazonSageMakerFullAccess`. **Do not select it.**
2. Instead, we will add a custom inline policy after the role is created. This gives us full control over the permissions.
3. Click **Next** without selecting any policies.

### Step 3 -- Name and Create the Role

1. **Role name:** `SageMaker-FraudShield-Training`
2. **Description:** `Least-privilege execution role for FraudShield SageMaker training jobs. Scoped to fraudshield-training-data bucket.`
3. Optionally add a tag:
   - **Key:** `project`
   - **Value:** `fraudshield`
4. Click **Create role**.

### Step 4 -- Add an Inline Policy for S3 Access

1. After the role is created, search for `SageMaker-FraudShield-Training` in the Roles list and click on it.
2. On the **Permissions** tab, click **Add permissions** -> **Create inline policy**.
3. Click the **JSON** tab and replace the contents with:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "ReadTrainingData",
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::fraudshield-training-data",
        "arn:aws:s3:::fraudshield-training-data/*"
      ]
    },
    {
      "Sid": "WriteModelArtifacts",
      "Effect": "Allow",
      "Action": "s3:PutObject",
      "Resource": "arn:aws:s3:::fraudshield-training-data/output/*"
    }
  ]
}
```

4. Review what this policy does:
   - **ReadTrainingData:** Allows reading objects and listing the bucket, but ONLY for the `fraudshield-training-data` bucket.
   - **WriteModelArtifacts:** Allows writing objects, but ONLY to the `output/` prefix within that bucket.
   - Notice: No `s3:DeleteObject`, no wildcard buckets, no `Resource: "*"`.
5. Click **Next**, name the policy `FraudShield-S3-Scoped-Access`, and click **Create policy**.

### Step 5 -- Add CloudWatch Logs Permissions

1. Still on the role's Permissions tab, click **Add permissions** -> **Create inline policy** again.
2. Switch to the **JSON** tab and enter:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "CloudWatchLogs",
      "Effect": "Allow",
      "Action": [
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents"
      ],
      "Resource": "arn:aws:logs:*:*:log-group:/aws/sagemaker/*"
    }
  ]
}
```

3. This allows SageMaker to write training logs to CloudWatch, but only to log groups under the `/aws/sagemaker/` prefix.
4. Name the policy `FraudShield-CloudWatch-Logs` and click **Create policy**.

### Step 6 -- Compare the Two Roles

Open both roles side by side (or switch between tabs) and compare:

| Aspect | Auto-Created Role | Custom Role |
|--------|-------------------|-------------|
| **S3 Access** | All buckets (`*`) | Only `fraudshield-training-data` |
| **S3 Actions** | All S3 actions (`s3:*`) | Read + scoped write only |
| **CloudWatch** | Full access (`logs:*`) | Create/write logs only, SageMaker prefix |
| **Other Services** | ECR, Lambda, Glue, etc. | None (not needed for training) |

### Step 7 -- Verify the Trust Policy

1. Click the **Trust relationships** tab on your new role.
2. Confirm the trust policy is identical to the auto-created role:

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

3. Both roles trust the same principal -- the difference is entirely in *what they are allowed to do* once assumed.

---

## Presentation Checkpoint

Be prepared to show:
- The **SageMaker-FraudShield-Training** role with both inline policies
- The S3 policy JSON and explain why each statement is scoped to specific ARNs
- The side-by-side comparison between the auto-created role and your custom role
- Explain: If a training job using this role tried to access a different S3 bucket, what would happen? (It would receive an `AccessDenied` error)
- Explain: Why do we separate read and write permissions? (A training job needs to read input data but should only write to a designated output location)

---

## Key Concepts

- **Inline Policy:** A policy embedded directly in a role, as opposed to a managed policy that can be shared across roles. Inline policies are useful for role-specific permissions that should not be reused elsewhere.
- **ARN (Amazon Resource Name):** The globally unique identifier for AWS resources. Scoping permissions to specific ARNs is the foundation of least privilege.
- **Sid (Statement ID):** A human-readable label for each statement in a policy. Not required, but makes policies easier to understand and audit.
- **Principle of Least Privilege:** Grant only the permissions required to perform the specific task. Start with zero permissions and add only what is needed.
