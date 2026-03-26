# Guide 1: Create a SageMaker Studio Domain

A Studio Domain is your team's shared workspace in SageMaker. It provisions the IDE, manages user access, and connects to shared storage (EFS). Every SageMaker project starts here.

---

## Steps

### Step 1 -- Navigate to SageMaker

1. Sign in to the **AWS Management Console**.
2. In the top search bar, type **SageMaker** and select **Amazon SageMaker** from the results.
3. Confirm your region in the top-right corner. Use **US East (N. Virginia) / us-east-1** for this lab.
4. Pin SageMaker to your navigation bar by clicking the pin icon next to the service name.

### Step 2 -- Open the Studio Setup

1. In the left navigation panel, click **Domains** (under "Admin configurations").
2. Click **Create domain**.
3. Select **Quick setup** -- this is the fastest way to get started and is appropriate for learning environments.

### Step 3 -- Configure the Domain

1. **Domain name:** Enter `fraudshield-domain`.
2. **Default execution role:** Select **Create a new role**.
   - In the dialog that appears, under "S3 buckets you specify," select **Any S3 bucket** (we will tighten this later with a custom role in Guide 4).
   - Click **Create role**.
3. Leave VPC settings as default (SageMaker will use the default VPC).
4. Click **Submit**.

### Step 4 -- Wait for Domain Creation

1. The domain status will show **Pending**. This typically takes 3-5 minutes.
2. While waiting, observe the domain details page. Note the:
   - **Domain ID** (e.g., `d-xxxxxxxxxxxx`)
   - **Domain ARN**
   - **Execution Role ARN** that was auto-created
   - **VPC** and **Subnets** being used
3. Refresh the page until the status changes to **InService**.

### Step 5 -- Launch Studio

1. Once the domain is **InService**, you will see a default user profile was created automatically.
2. Click **Open Studio** next to the default user profile.
3. Studio will open in a new browser tab. This may take 1-2 minutes on first launch.
4. Once Studio loads, take a moment to observe the home page -- you will explore it in Guide 2.
5. Close the Studio tab for now and return to the SageMaker console.

---

## Presentation Checkpoint

Be prepared to show:
- The **fraudshield-domain** in the Domains list with **InService** status
- The **Domain ID** and **region**
- The **Execution Role ARN** that was auto-created during setup
- Explain: What does "Quick setup" do behind the scenes? (Creates a domain, default user profile, execution role, and connects to default VPC)

---

## Key Concepts

- **Domain:** A shared workspace that manages Studio access, user profiles, and storage for a team.
- **Execution Role:** An IAM role that SageMaker assumes when performing actions on your behalf (training, deploying, accessing S3). The Quick Setup creates one with broad permissions -- we will address this in Guide 4.
- **Default VPC:** The pre-configured virtual network in your AWS account. For production, you would use a custom VPC with private subnets.

---

## AIML Connection

Recall from the *ML Lifecycle & Reproducibility* reading: environment setup is the first stage of any ML project. A SageMaker Domain is the cloud equivalent of setting up your local Python virtual environment -- but it provisions managed compute, shared storage, and IAM-controlled access for an entire team.
