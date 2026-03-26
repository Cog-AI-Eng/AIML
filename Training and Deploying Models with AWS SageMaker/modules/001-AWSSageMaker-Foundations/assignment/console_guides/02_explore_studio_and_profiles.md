# Guide 2: Explore Studio and Create User Profiles

SageMaker supports multiple users within a single Domain through **user profiles**. Each profile gets its own home directory on EFS and can be assigned a different execution role. This is how teams with different responsibilities (data engineers, data scientists, business analysts) share the same workspace with appropriate access levels.

---

## Steps

### Step 1 -- Launch Studio and Explore the Home Page

1. In the SageMaker console, go to **Domains** and click on **fraudshield-domain**.
2. Click **Open Studio** next to the default user profile.
3. Once Studio loads, explore the **home page**. Note the sections:
   - **Quick start actions** (Create notebook, Open JumpStart, etc.)
   - **Recent activity**
   - Left sidebar with icons for Home, File Browser, Running Terminals/Kernels, etc.
4. Click the **SageMaker Home** icon (house icon) in the left sidebar. You should see links for:
   - **Studio** (full IDE for data scientists)
   - **JumpStart** (pre-built models and solutions)
   - Other tools (Data Wrangler, Feature Store, Pipelines, etc.)

### Step 2 -- Understand the Three Studio Interfaces

From the SageMaker console (not inside Studio), observe the domain details. SageMaker offers three interfaces for different personas:

| Interface | Target User | Key Capability |
|-----------|------------|----------------|
| **Studio** | Data scientists, ML engineers | Full IDE with notebooks, terminal, Git, experiments |
| **Studio Classic** | Existing users, backward compatibility | Legacy Studio experience (being deprecated) |
| **Canvas** | Business analysts, non-coders | No-code ML model building with visual interface |

Note which interfaces are enabled for your domain. For this lab, we will use **Studio**.

### Step 3 -- Create a Second User Profile

1. Close the Studio tab and return to the **SageMaker console**.
2. Go to **Domains** -> click on **fraudshield-domain**.
3. Under the **User profiles** tab, click **Add user**.
4. Configure the new profile:
   - **Name:** `analyst-team`
   - **Execution role:** Select the same role that was auto-created during domain setup (you can find it in the dropdown -- it starts with `AmazonSageMaker-ExecutionRole-`).
5. Click **Next** through any additional settings, then **Submit**.
6. Wait for the profile to be created (this should take under a minute).

### Step 4 -- Verify Both Profiles

1. On the domain details page, you should now see two user profiles:
   - The **default profile** (created during Quick Setup)
   - **analyst-team** (just created)
2. Note that both currently use the **same execution role**. In a real environment, you would assign the analyst team a more restricted role (e.g., read-only S3 access, no training job permissions). We will create a custom role in Guide 4 that could be assigned here.

### Step 5 -- Launch Studio as the New User

1. Click **Open Studio** next to the **analyst-team** profile.
2. Observe that the Studio environment looks the same but is running under a different user context.
3. Click the **File Browser** icon in the left sidebar. Note that this user has their own home directory -- separate from the default user's files.
4. Close this Studio tab.

---

## Presentation Checkpoint

Be prepared to show:
- Both user profiles listed under **fraudshield-domain**
- The execution role assigned to each profile
- Explain: Why would you use different execution roles for different user profiles? (Least privilege -- analysts might only need read access to S3, while ML engineers need full training and deployment permissions)
- Explain: What is the difference between Studio, Studio Classic, and Canvas? (Studio is the full IDE; Canvas is no-code for business users; Studio Classic is the legacy interface)

---

## Key Concepts

- **User Profile:** An individual identity within a Domain. Each profile gets isolated EFS storage but shares the Domain's networking and security settings.
- **EFS (Elastic File System):** The shared storage backend for Studio. Each user profile gets a home directory on EFS that persists across sessions.
- **Multi-Tenancy:** A single Domain can support multiple users (profiles) with different roles and permissions, enabling team collaboration with appropriate access control.

---

## AIML Connection

The *ML Lifecycle & Reproducibility* reading emphasized consistent, reproducible environments. SageMaker Studio provides this at the team level -- every team member accesses the same managed environment with version-controlled notebooks and shared storage, eliminating "works on my machine" problems.
