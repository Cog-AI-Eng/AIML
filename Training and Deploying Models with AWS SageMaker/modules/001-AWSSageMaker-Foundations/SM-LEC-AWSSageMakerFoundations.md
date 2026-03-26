# AWSSageMaker-Foundations Lecture - Instructor Guide

**Total Duration:** 180 Minutes (3 Stages)
**Consolidated Activities:** Ecosystem & Core Services (SM-CT-Ecosystem&CoreServices), Studio Domains & Profiles (SM-CT-StudioDomains&Profiles), IAM & Least-Privilege Practices (SM-CT-IAM&Least-PrivilegePractices), The SageMaker ML Lifecycle (SM-CT-TheSageMakerMLLifecycle), JumpStart Pre-built Models (SM-CT-JumpStartPre-builtModels)

| Block | Content | Minutes |
|-------|---------|---------|
| Stage 1 | Console Tour, Studio Domain Setup, and IAM Role Creation | 45 |
| Break 1 | Stretch / Questions | 10 |
| Stage 2 | The SageMaker ML Lifecycle in Action: JumpStart Deployment | 45 |
| Break 2 | Stretch / Questions | 10 |
| Stage 3 | Least-Privilege IAM, Cleanup Discipline, and Lifecycle Mapping | 45 |
| Buffer | Open Q&A, Console Exploration Activity, Wrap-Up | 25 |

---

## Lecture Overview

**Unified Scenario -- FraudShield Risk Analytics**

Associates will play the role of an ML engineer at a fictional fintech company called FraudShield that needs to deploy a transaction fraud detection model. The company has a small data science team that needs a governed, cost-conscious SageMaker environment. Today's session walks the team through their first day on AWS:

1. **"How do we set up our ML workspace?"** (Studio Domain and User Profiles)
2. **"How do we secure our environment?"** (IAM roles and least-privilege policies)
3. **"Can we get a fraud detection model running today?"** (JumpStart deployment)

This scenario threads through every Module 1 exit criterion: Associates will navigate the SageMaker console, set up a Studio Domain, create IAM execution roles, map the ML lifecycle to SageMaker services, deploy a JumpStart model, and clean up resources.

**Why console-first?** Every step in this lecture is performed in the AWS Management Console (browser). Associates follow along on their own screens. Code (SDK/CLI) is discussed but not executed until later modules. This builds the foundational understanding that all SDK operations map to console actions.

---

## Pre-Lecture Setup

### Instructor Checklist

- [ ] AWS account with SageMaker access verified (instructor and all Associates)
- [ ] No pre-existing SageMaker Domain in the account (or documented plan for multi-Domain setup)
- [ ] Browser open to [console.aws.amazon.com](https://console.aws.amazon.com) and signed in
- [ ] Screen sharing enabled with font/zoom large enough for projector readability
- [ ] SageMaker pinned to the console navigation bar
- [ ] Billing & Cost Management console bookmarked for cleanup verification
- [ ] This instructor guide open in a second tab or printed

### Student Prerequisites

- [ ] AWS account credentials (IAM user with console access)
- [ ] Completed readings: Ecosystem & Core Services, Studio Domains & Profiles, IAM & Least-Privilege Practices, The SageMaker ML Lifecycle, JumpStart Pre-built Models
- [ ] Browser open to AWS Console

---

## Stage 1: Console Tour, Studio Domain Setup, and IAM Role Creation

**Duration:** 45 minutes
**Exit Criteria Addressed:**
- Describe the core services in the AWS SageMaker ecosystem and their role in the ML lifecycle (Required)
- Identify the differences between SageMaker Studio, Studio Classic, and Canvas for different user personas (Required)
- Implement IAM least-privilege roles and policies for SageMaker Execution Roles (Required)

### Instructor Opening (5 minutes -- talk, no code)

> "You have read about the SageMaker ecosystem, Studio Domains, and IAM roles. Today we are going to set all of this up live. By the end of this lecture, every one of you will have a working SageMaker environment with proper security, and you will have deployed your first model. Everything we do is in the browser -- no CLI, no SDK, no code. We will get to code in Module 2. Today is about understanding what is happening before we automate it."

> "Our scenario: you just joined FraudShield, a fintech startup. Your team lead says: 'We need a governed ML workspace on AWS by end of day. Set up the environment, lock down permissions, and show me a working fraud detection model.' Let's go."

---

### STEP 1 -- Finding SageMaker in the Console (5 minutes)

**Pacing: live demonstration.** Navigate the console while Associates follow on their own screens.

1. Sign in to the AWS Console. Point out the Console Home page layout: recently visited services, search bar, region selector.
2. Click the search bar (or press `Alt+S`). Type "SageMaker." Click **Amazon SageMaker**.
3. Walk through the left sidebar: **Studio**, **Notebook instances**, **Training**, **Inference**, **Processing**, **Pipelines**, **Governance**.

> "Each of these sidebar sections maps to a stage of the ML lifecycle you read about. Studio is for Build. Training is for Train. Inference is for Deploy. Governance is for versioning and approval. By the end of this curriculum, you will have used every one of these sections."

4. Pin SageMaker to the navigation bar (click the star icon).

**Instructor Note:** If any student cannot find SageMaker, check their region. SageMaker availability varies by region. Recommend `us-east-1` for the broadest service coverage.

[PAUSE FOR Q&A - Ask: "Which sidebar section do you think we will use most today?"]

---

### STEP 2 -- Creating a Studio Domain (15 minutes)

**Pacing: live demonstration, step by step.** Wait for all Associates to complete each step before moving on.

1. In the SageMaker sidebar, click **Studio** (or **Admin configurations > Domains**).
2. If no Domain exists, click **Create Domain**.
3. Select **Quick setup**. Explain: "Quick setup uses your default VPC, creates a default execution role, and adds one User Profile. For production you would use Standard setup to customize the network and encryption. For learning, Quick setup is perfect."
4. Review the defaults:
   - **Domain name:** Change to `fraudshield-domain` (or leave default).
   - **Execution role:** Accept "Create a new role." Explain: "AWS is creating an IAM role right now that SageMaker will use to access S3 and other services on your behalf. We will examine this role in the next step."
   - **VPC:** Default VPC is fine.
5. Click **Submit**. Domain creation takes 3-5 minutes.

> "While we wait, let me explain what just happened behind the scenes. SageMaker is provisioning an EFS volume for shared storage, configuring network access within your VPC, and creating the execution role. Every User Profile in this Domain will share the same infrastructure but get their own home directory on EFS."

**Instructor Note:** Use the wait time to discuss Studio vs. Studio Classic vs. Canvas. Ask: "Who is Canvas designed for?" (Business analysts who do not write code.)

| Interface | Audience | Code? |
|-----------|----------|-------|
| Studio | ML engineers, data scientists | Yes |
| Studio Classic | Legacy users | Yes |
| Canvas | Business analysts | No |

6. When the Domain status shows **InService**, click the Domain name to view details.
7. Show the User Profiles list (one default profile should exist).
8. Click **Launch > Studio** next to the profile. Let Studio load in a new tab. Point out the JupyterLab interface.

> "You are now inside SageMaker Studio. This is the IDE where you will spend most of your time in later modules. Notice it runs entirely in the browser -- no local installation required."

**Teaching Note:** Some Associates may see Studio Classic instead of the new Studio. If so, explain the difference and show how to switch to the new experience in Domain settings.

[PAUSE FOR Q&A - Ask: "What three things does a Domain bundle together?" (VPC/network, default execution role, shared EFS storage)]

---

### STEP 3 -- Adding a Second User Profile (5 minutes)

**Pacing: live demonstration.**

1. Go back to the SageMaker console tab. Navigate to **Admin configurations > Domains**, click the Domain name.
2. Click **Add user**.
3. Set the name to `analyst-team`.
4. For execution role, select the same default role. Explain: "In a real team, you would create a separate role with narrower permissions for analysts who only need Canvas access. We will do that in the next step."
5. Click **Submit**. The profile appears in seconds.

> "Now we have two profiles sharing the same Domain. The data scientist profile can launch Studio and run notebooks. The analyst profile could be configured to only use Canvas. This is multi-user collaboration: shared infrastructure, individual workspaces."

---

### STEP 4 -- Examining the Auto-Generated IAM Role (15 minutes)

**Pacing: live demonstration with pauses for discussion.**

1. Open a new tab. Search for "IAM" in the console search bar. Click **IAM**.
2. In the IAM sidebar, click **Roles**.
3. Search for "SageMaker." Find the auto-generated role (e.g., `AmazonSageMaker-ExecutionRole-YYYYMMDDTHHMMSS`).
4. Click the role name. Walk through each section:

**Trust relationships tab:**

> "This JSON says the SageMaker service is allowed to assume this role. Without this trust relationship, SageMaker cannot use the role, no matter what permissions it has."

Show the trust policy JSON. Point out the `sagemaker.amazonaws.com` service principal.

**Permissions tab:**

> "Look at the attached policy: `AmazonSageMakerFullAccess`. This is a managed policy from AWS that grants broad permissions -- S3, ECR, CloudWatch, and more. It is convenient for learning but violates least-privilege for production. If this role were compromised, the attacker could access every S3 bucket in your account and create endpoints that cost hundreds of dollars per hour."

5. Click on `AmazonSageMakerFullAccess` to show its policy document. Scroll through the permissions. Point out the `"Resource": "*"` patterns.

> "See these wildcard resources? That means this role can access ANY S3 bucket, ANY ECR repository, ANY CloudWatch log group. In production, we would replace this with a custom policy scoped to specific resources."

[PAUSE FOR Q&A - Ask: "Why is 'Resource: *' dangerous in a production environment?"]

**Tags tab:**

> "Tags are key-value metadata. We will use them later for tag-based access control -- a pattern where you restrict role permissions to only resources carrying specific tags."

[PAUSE FOR BREAK - 10 MINS]

---

## Stage 2: The SageMaker ML Lifecycle in Action -- JumpStart Deployment

**Duration:** 45 minutes
**Exit Criteria Addressed:**
- Describe the five stages of the SageMaker ML Lifecycle (Prep, Build, Train, Deploy, Monitor) (Required)
- Deploy a pre-built model from SageMaker JumpStart using the AWS Management Console (Required)
- Implement resource cleanup steps (deleting endpoints and models) to avoid unnecessary billing (Required)

### Instructor Opening (3 minutes)

> "Now that we have a working environment, let's map it to the ML lifecycle. You read about five stages: Prepare, Build, Train & Tune, Deploy, and Monitor. Today we are going to skip straight to Deploy using JumpStart -- think of it as a fast-forward button that lets us see a model serving predictions within the hour. In later modules, you will build the middle stages yourself."

Draw or display the lifecycle diagram on screen:

```
Prepare --> Build --> Train & Tune --> Deploy --> Monitor
                                        ^
                                        |
                                   (We are here)
```

---

### STEP 5 -- Lifecycle-to-Console Mapping Exercise (7 minutes)

**Pacing: interactive discussion.** Have Associates navigate the console while you quiz them.

> "I want everyone to open the SageMaker console sidebar. I am going to name a lifecycle stage, and you tell me which sidebar section it maps to."

| Stage | Sidebar Section |
|-------|----------------|
| Prepare | Processing > Processing jobs; also S3 console |
| Build | Studio |
| Train & Tune | Training > Training jobs; Training > Hyperparameter tuning jobs |
| Deploy | Inference > Models, Endpoint configurations, Endpoints |
| Monitor | Inference > Model monitoring |

> "Notice that Governance (Model Registry) spans multiple stages -- it is the versioning layer that connects Training to Deployment. We will use it heavily in Module 3."

[PAUSE FOR Q&A - Ask: "Which lifecycle stage did we just complete in Stage 1 of this lecture?" (Build -- we set up our workspace)]

---

### STEP 6 -- Navigating to JumpStart (5 minutes)

**Pacing: live demonstration.**

1. Switch to the Studio browser tab (launched in Step 2).
2. In Studio, locate **JumpStart** in the left sidebar (rocket icon or labeled link).
3. Click JumpStart. The model catalog loads.
4. Browse the categories: Text Generation, Image Classification, Tabular, Sentence Embeddings.

> "JumpStart is a curated library of pre-trained models. For our FraudShield scenario, we want a tabular classification model. But for this demo, let's pick a lightweight text classification model that deploys quickly and stays within Free Tier limits."

5. Search for a lightweight model (e.g., a text classification or sentiment model that supports `ml.m5.xlarge`).
6. Click the model card. Walk through the detail page:
   - Model description and provider
   - Default instance type (check it is `ml.m5.xlarge` or change it)
   - Sample input/output
   - License

**Instructor Note:** Model availability in JumpStart changes over time. Before the lecture, identify a specific model that deploys on `ml.m5.xlarge` within 5-10 minutes. Test the deployment yourself the day before. Have a backup model in case the first choice is unavailable.

---

### STEP 7 -- Deploying a JumpStart Model (15 minutes)

**Pacing: live demonstration. All Associates deploy simultaneously.**

1. On the model detail page, review the deployment configuration:
   - **Endpoint name:** Change to `fraudshield-demo-endpoint`.
   - **Instance type:** Verify `ml.m5.xlarge`. If the default is a GPU instance, change it.

> "We are deliberately choosing `ml.m5.xlarge` because it stays within Free Tier limits. A `ml.p3.2xlarge` GPU instance costs roughly 10x more per hour. Cost awareness is a required skill in cloud ML."

2. Click **Deploy**. The progress indicator shows status updates.
3. While waiting (5-10 minutes), discuss:

> "What is happening right now? SageMaker is provisioning an `ml.m5.xlarge` instance, pulling the model's Docker container from ECR, downloading the model weights, loading them into memory, and starting an HTTPS server. When it finishes, we will have a live endpoint that any application can call."

> "This is exactly the Deploy stage of the lifecycle. In Module 3, you will do this manually -- creating a Model object, an Endpoint Configuration, and an Endpoint as three separate steps. JumpStart bundles all three for you."

4. When the status shows **InService**, verify:
   - On the JumpStart model page, the endpoint status shows InService.
   - Navigate to **Inference > Endpoints** in the SageMaker sidebar. The endpoint appears in the list.

> "Your model is now live. It is accepting HTTPS requests and returning predictions. Every second it runs, it costs money. That is why cleanup is mandatory."

[PAUSE FOR Q&A - Ask: "If you walked away and forgot about this endpoint, what would happen to your AWS bill?"]

---

### STEP 8 -- Testing the Endpoint (5 minutes)

**Pacing: live demonstration.**

1. Back on the JumpStart model page, click **Open Notebook** (if available) to generate a sample invocation notebook.
2. Run the sample code in Studio. Show the prediction result.
3. If no sample notebook is available, show the endpoint details page in the console and explain that we will write invocation code in Module 3.

> "We just completed the full Deploy and first invocation. In Module 3 you will learn to write your own `invoke_endpoint` calls with `boto3`. For now, the important thing is that you know: model in S3, container in ECR, endpoint serving predictions, all visible in the console."

---

### STEP 9 -- Mandatory Cleanup (10 minutes)

**Pacing: live demonstration. EVERY student must complete this step.**

> "This is the most important step of the entire lecture. If you skip this, you will be charged for every minute the endpoint runs. Let's clean up together."

**Console cleanup:**

1. Navigate to **Inference > Endpoints**. Select `fraudshield-demo-endpoint`. Click **Actions > Delete**. Confirm.
2. Navigate to **Inference > Models**. Select the associated model. Click **Actions > Delete**. Confirm.
3. Navigate to **Inference > Endpoint configurations**. Select the associated config. Click **Actions > Delete**. Confirm.

> "Delete in this order: endpoint first (stops billing immediately), then model, then configuration. Always verify the endpoint is gone from the list."

4. Search "Billing" in the console. Open **Billing & Cost Management**. Show the current charges.

> "Make this a habit: every time you deploy anything, check billing after cleanup. If you see unexpected charges, investigate immediately."

**Teaching Note:** Walk around the room (or monitor screen shares) to verify every student has deleted their endpoint. This is non-negotiable.

[PAUSE FOR BREAK - 10 MINS]

---

## Stage 3: Least-Privilege IAM, Cleanup Discipline, and Lifecycle Mapping

**Duration:** 45 minutes
**Exit Criteria Addressed:**
- Implement IAM least-privilege roles and policies for SageMaker Execution Roles (Required)
- Describe the five stages of the SageMaker ML Lifecycle (Required)
- Demonstrate the use of SageMaker Studio Domains for multi-user collaboration (Preferred)

### STEP 10 -- Creating a Least-Privilege Role (20 minutes)

**Pacing: live demonstration, step by step.**

> "In Stage 1 we saw that the auto-generated role has `AmazonSageMakerFullAccess` -- way too broad. Let's build a custom role from scratch with only the permissions a training job needs."

1. Open the IAM console. Navigate to **Roles > Create role**.
2. **Trusted entity:** Select **AWS service**. Use case: **SageMaker**. Click **Next**.
3. **Permissions:** Instead of `AmazonSageMakerFullAccess`, search for:
   - `AmazonS3ReadOnlyAccess` -- select it.
   - `CloudWatchLogsFullAccess` -- select it.
   Click **Next**.
4. **Name:** `SageMaker-Training-LeastPrivilege`.
5. **Description:** "Custom execution role for SageMaker training jobs with read-only S3 and CloudWatch logging."
6. Click **Create role**.

> "This role can read from S3 and write logs. It cannot create endpoints, delete buckets, or access ECR. If this role were compromised, the blast radius is limited to reading data and writing logs."

7. Click into the new role. Show the trust policy and attached policies.
8. Now create a custom inline policy for scoped S3 access:
   - On the Permissions tab, click **Add permissions > Create inline policy**.
   - Switch to JSON tab. Paste:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:ListBucket"],
      "Resource": [
        "arn:aws:s3:::fraudshield-training-data",
        "arn:aws:s3:::fraudshield-training-data/*"
      ]
    },
    {
      "Effect": "Allow",
      "Action": ["s3:PutObject"],
      "Resource": "arn:aws:s3:::fraudshield-training-data/output/*"
    }
  ]
}
```

9. Name it `S3-FraudShield-Scoped`. Click **Create policy**.

> "Now this role can only read from one specific bucket and write to one specific prefix. Compare that to the auto-generated role that could access every bucket in the account. This is least-privilege in practice."

[PAUSE FOR Q&A - Ask: "If a training job using this role tried to write to a different S3 bucket, what would happen?" (Access Denied error)]

---

### STEP 11 -- Multi-User Domain Configuration Discussion (10 minutes)

**Pacing: interactive discussion with console demonstration.**

1. Navigate back to **SageMaker > Admin configurations > Domains**. Click the Domain.
2. Show the two User Profiles: the default one and `analyst-team`.
3. Discuss:

> "Right now both profiles use the same default execution role. In a real FraudShield setup, we would assign our new least-privilege role to the `analyst-team` profile. The analyst can use Canvas for no-code model building but cannot launch training jobs or create endpoints -- because their role does not have those permissions."

> "This is how Domains enable multi-user collaboration with proper governance. Same infrastructure, different permission boundaries."

4. (Optional live demo) Edit the `analyst-team` profile and change its execution role to `SageMaker-Training-LeastPrivilege`. Show how the role selection works in the console.

---

### STEP 12 -- Lifecycle Mapping Consolidation (10 minutes)

**Pacing: interactive whiteboard/discussion exercise.**

> "Let's consolidate everything we did today against the five lifecycle stages."

| Lifecycle Stage | What We Did Today | Console Location |
|----------------|-------------------|-----------------|
| **Prepare** | (Not covered -- Module 2) | Processing, S3 |
| **Build** | Created Studio Domain, launched Studio, browsed JumpStart | Studio, JumpStart |
| **Train & Tune** | (Not covered -- Module 2) | Training jobs |
| **Deploy** | Deployed JumpStart model to endpoint, tested it | Inference > Endpoints |
| **Monitor** | (Not covered -- Module 4) | Model monitoring |
| **Govern** | Created IAM roles, configured User Profiles | IAM console, Domain settings |

> "We touched Build, Deploy, and Govern today. Modules 2, 3, and 4 will fill in Prepare, Train, and Monitor. By the end of the curriculum, you will have hands-on experience with every cell in this table."

---

### STEP 13 -- Final Cleanup Verification (5 minutes)

**Pacing: everyone together.**

1. Navigate to **Inference > Endpoints**. Verify the list is empty (or shows no InService endpoints from today).
2. Navigate to **Inference > Models**. Verify no lingering models from today's demo.
3. Check **Billing & Cost Management** one more time.

> "I want everyone to confirm out loud: is your endpoint deleted? Do you see any InService resources? Good. This cleanup discipline will save you real money throughout this curriculum."

**Teaching Note:** If any student still has a running endpoint, stop the lecture and help them delete it immediately.

[PAUSE FOR Q&A]

---

## Wrap-up & Console Exploration Activity

**Duration:** 25 minutes

### Summary (5 minutes)

> "Today you set up a complete SageMaker environment from scratch: a Studio Domain with two User Profiles, a least-privilege IAM role, and a deployed JumpStart model. You navigated the console end to end and cleaned up your resources. In Module 2, you will write your own training scripts and run them on SageMaker infrastructure using the environment you built today."

### Console Exploration Activity (20 minutes)

> "For the remainder of this session, I want you to explore the SageMaker console on your own. Here is your challenge:"

**Activity Instructions:**

1. Navigate to every sidebar section in the SageMaker console (Studio, Notebook instances, Processing, Training, Inference, Pipelines, Governance).
2. For each section, write down which ML lifecycle stage it belongs to (Prepare, Build, Train, Deploy, Monitor, or Govern).
3. Open the IAM console and find the two roles in your account: the auto-generated `AmazonSageMaker-ExecutionRole-*` and the `SageMaker-Training-LeastPrivilege` role you created.
4. Compare their permissions. Write down three specific actions the auto-generated role can perform that the least-privilege role cannot.
5. (Stretch) Navigate to the S3 console. Find the SageMaker-generated bucket (if one exists). Explore its folder structure.

> "Submit your lifecycle mapping and permission comparison as a text file or screenshot. We will review these at the start of the Module 2 lecture."

---

## Instructor Notes -- Common Issues

| Issue | Resolution |
|-------|-----------|
| Student cannot find SageMaker | Check the region selector (top-right). Switch to `us-east-1`. |
| Domain creation fails | Usually a VPC/subnet issue. Verify the default VPC exists. If deleted, create a new default VPC in the VPC console. |
| JumpStart model unavailable | Model catalog changes. Have 2-3 backup models identified that deploy on `ml.m5.xlarge`. |
| Studio takes too long to load | First load can take 3-5 minutes. This is normal. Have Associates wait. |
| Student forgets to delete endpoint | Walk over and help immediately. Check billing together. |
| IAM role creation permission denied | Student's IAM user may lack `iam:CreateRole` permission. The account admin needs to grant this. |
