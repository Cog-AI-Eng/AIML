# Guide 1: Configure a Custom Studio Domain

This guide walks you through creating a SageMaker Studio Domain using the Standard Setup option. Unlike Quick Setup, Standard Setup gives you control over the VPC, subnets, EFS volume, and IAM execution role -- the foundation for a production-grade FraudShield environment.

---

## Steps

### Step 1 -- Navigate to SageMaker Studio
1. Sign in to the AWS Management Console and set your region to **us-east-1**.
2. In the top search bar, type **SageMaker** and select **Amazon SageMaker**.
3. In the left navigation pane, expand **Admin configurations** and choose **Domains**.
4. Click **Create domain**.

### Step 2 -- Select Standard Setup
1. On the setup screen, choose **Standard setup** (do not use Quick setup).
2. Enter a domain name: `fraudshield-studio-domain`.
3. Under **Authentication**, keep the default **IAM** authentication method selected.

### Step 3 -- Configure the Execution Role
1. Under **Default execution role**, click **Create a new role**.
2. In the dialog, select **Any S3 bucket** for S3 access.
3. Click **Create role** and note the generated role ARN.
4. Verify the role name appears in the dropdown (e.g., `AmazonSageMaker-ExecutionRole-...`).

### Step 4 -- Configure Networking
1. Under **Network**, choose **VPC only** for the network access type.
2. Select your existing VPC from the dropdown.
3. Select at least **two private subnets** in different Availability Zones.
4. Select a security group that allows inbound/outbound traffic within the VPC.
5. Leave **Encryption key** as the default AWS-managed key unless your organization requires a custom KMS key.

### Step 5 -- Review Storage Settings
1. Under **Storage**, confirm that a new EFS file system will be created for the domain.
2. Note the default home directory size per user (the default is typically 5 GB).
3. Leave the remaining storage defaults unchanged.

### Step 6 -- Create a User Profile
1. Scroll to **User profiles** and click **Add user**.
2. Enter the user profile name: `fraudshield-analyst`.
3. Keep the execution role set to the domain default you configured in Step 3.
4. Click **Next** through any remaining optional settings.

### Step 7 -- Review and Submit
1. Review all settings: domain name, VPC, subnets, security group, EFS, and execution role.
2. Click **Submit**.
3. Wait for the domain status to change from **Pending** to **InService** (this may take 5-10 minutes).
4. Once InService, click the domain name and verify the **Domain details** page shows the correct VPC, subnets, and EFS file system ID.

---

## Presentation Checkpoint
Be prepared to show:
- The Domain details page with domain name, VPC ID, subnet IDs, and EFS file system ID visible.
- The execution role ARN associated with the domain.
- The user profile `fraudshield-analyst` listed under the domain.

---

## Key Concepts
- **Studio Domain:** A logically isolated environment that bundles compute, storage (EFS), and identity for one or more users.
- **Standard Setup vs. Quick Setup:** Standard Setup exposes VPC, subnet, and security group configuration for network isolation; Quick Setup auto-creates a default VPC configuration.
- **Execution Role:** An IAM role assumed by SageMaker services to access AWS resources (S3, ECR, CloudWatch) on behalf of the user.
- **VPC Only Mode:** Forces all SageMaker traffic through your VPC, enabling private subnet routing and security group controls.
