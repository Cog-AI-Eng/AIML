# VPC, PrivateLink & KMS

**Estimated Time:** 10 Minutes

## Introduction

In the foundational SageMaker skill, you used default VPC settings and accepted the auto-created execution role's broad permissions. For production deployments, security requirements are stricter: training data must not traverse the public internet, model artifacts must be encrypted at rest and in transit, and SageMaker API calls must stay within the AWS private network. This reading covers the three core security services -- VPC configuration, PrivateLink, and KMS -- that secure a production SageMaker environment.

## Core Concepts

### VPC isolation for SageMaker

By default, SageMaker jobs (training, processing, inference) can access the public internet. In a production environment, you restrict this by placing SageMaker resources inside a **VPC** (Virtual Private Cloud) with private subnets:

1. **Create private subnets:** In the VPC console, create subnets with no internet gateway route. SageMaker resources in these subnets cannot reach the internet.
2. **Configure SageMaker to use the VPC:** When creating a training job, processing job, or endpoint, specify the VPC ID, subnet IDs, and security group IDs in the network configuration section.
3. **S3 access via VPC Endpoint:** SageMaker jobs in a private subnet cannot reach S3 over the internet. Create a **VPC Gateway Endpoint for S3** to allow private connectivity to S3 without internet access.

**Console steps for VPC-only training:**

1. Navigate to **SageMaker > Training > Training jobs > Create training job**.
2. Under **Network** (or **VPC**), select your VPC, private subnets, and security group.
3. Check **Enable network isolation** to prevent the training container from making any outbound network calls (strictest mode). This means all code and dependencies must be in the container or passed through input channels.

### Common VPC debugging issues

| Problem | Cause | Fix |
| :--- | :--- | :--- |
| Training job hangs or times out | No S3 VPC Endpoint; job cannot download data | Create a Gateway Endpoint for S3 in the VPC |
| Training job fails to install pip packages | Network isolation enabled; no internet access | Pre-install packages in the container image, or disable network isolation |
| Endpoint cannot be invoked | Security group inbound rules do not allow HTTPS (port 443) | Add inbound rule for HTTPS from the invoking source |
| Processing job cannot reach Athena | No Athena VPC Interface Endpoint | Create an Interface Endpoint for Athena |

### PrivateLink for SageMaker API calls

By default, when you call SageMaker APIs (e.g., `CreateTrainingJob`, `InvokeEndpoint`) from your VPC, the API call traverses the public internet to reach the SageMaker service endpoint. **PrivateLink** creates a private network connection between your VPC and the SageMaker service, so API calls never leave the AWS backbone.

**Setup:**

1. Navigate to **VPC > Endpoints > Create Endpoint**.
2. **Service category:** AWS services.
3. **Service:** Search for `com.amazonaws.<region>.sagemaker.api` (for SageMaker API calls) or `com.amazonaws.<region>.sagemaker.runtime` (for endpoint invocations).
4. **VPC:** Select your VPC.
5. **Subnets:** Select the subnets where the endpoint network interfaces should be created.
6. **Security group:** Attach a security group that allows inbound HTTPS (port 443).
7. **Policy:** Accept the default (full access) or create a custom policy to restrict which SageMaker API actions can be called through this endpoint.
8. Click **Create endpoint**.

After creation, SageMaker API calls from within the VPC automatically route through the PrivateLink connection instead of the public internet.

### KMS encryption

AWS Key Management Service (KMS) manages encryption keys. SageMaker uses KMS for:

**Encryption at rest:**
- **S3 data:** Training data and model artifacts in S3 can be encrypted with a KMS key (SSE-KMS). Specify the KMS key ARN in the S3 bucket policy or the SageMaker job's `OutputDataConfig`.
- **EFS volumes:** Studio Domain EFS storage can be encrypted with a customer-managed KMS key (configured during Domain creation).
- **Training job volumes:** The temporary EBS storage attached to training instances is encrypted with a KMS key specified in `ResourceConfig.VolumeKmsKeyId`.

**Encryption in transit:**
- SageMaker encrypts all data in transit with TLS by default. For inter-container communication in distributed training, enable **inter-container encryption** (`EnableInterContainerTrafficEncryption`) to encrypt data exchanged between instances.

### Configuring KMS in the console

1. Navigate to **KMS > Customer managed keys > Create key**.
2. **Key type:** Symmetric. **Key usage:** Encrypt and decrypt.
3. **Key alias:** Use a descriptive name (e.g., `sagemaker-training-key`).
4. **Key administrators:** Grant your admin role permission to manage the key.
5. **Key usage permissions:** Grant the SageMaker execution role permission to use the key for encryption/decryption.
6. When creating SageMaker resources, specify the KMS key ARN in the relevant configuration fields.

### Putting it together: production security architecture

A production SageMaker deployment typically combines all three:

1. **VPC with private subnets:** All SageMaker resources run in private subnets with no internet access.
2. **VPC Endpoints:** Gateway Endpoint for S3, Interface Endpoints for SageMaker API, SageMaker Runtime, CloudWatch Logs, and ECR (for pulling container images).
3. **KMS encryption:** Customer-managed keys for S3 data, EBS volumes, and EFS storage. Inter-container encryption enabled for distributed training.
4. **Security groups:** Minimal inbound/outbound rules scoped to required services only.

This architecture ensures that no ML data or API calls traverse the public internet, all data is encrypted at rest and in transit, and the encryption keys are under your control.

## Connecting to Practice

This topic completes the Architecture module by securing the entire SageMaker platform. You now have a full production architecture toolkit: Spot Training for cost, instance right-sizing for efficiency, auto-scaling for elasticity, Inference Recommender for data-driven decisions, CloudWatch for visibility, and VPC/PrivateLink/KMS for security. The module lecture will walk through configuring a VPC-isolated training job with KMS encryption and PrivateLink. The assignment will require you to deploy a model to a VPC-only endpoint with PrivateLink and demonstrate that API calls do not traverse the public internet.

## Further Learning & Resources

**Documentation and reading**

- **[SageMaker VPC Configuration](https://docs.aws.amazon.com/sagemaker/latest/dg/infrastructure-connect-to-resources.html)** - *Docs*: Reference for VPC networking, security groups, and network isolation.
- **[PrivateLink for SageMaker](https://docs.aws.amazon.com/sagemaker/latest/dg/interface-vpc-endpoint.html)** - *Docs*: Guide to creating PrivateLink endpoints for SageMaker services.
- **[KMS Encryption for SageMaker](https://docs.aws.amazon.com/sagemaker/latest/dg/encryption-at-rest.html)** - *Docs*: Reference for encryption configuration across all SageMaker resources.

**Interactive practice**

- **[SageMaker Security Workshop](https://catalog.workshops.aws/sagemaker-security/en-US)** - *Interactive*: Hands-on lab covering VPC configuration, PrivateLink, KMS, and network isolation for SageMaker.
