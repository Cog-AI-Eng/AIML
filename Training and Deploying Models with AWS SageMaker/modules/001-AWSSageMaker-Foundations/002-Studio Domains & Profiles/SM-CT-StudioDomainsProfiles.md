# Studio Domains & Profiles

**Estimated Time:** 10 Minutes

## Introduction

In the *Ecosystem & Core Services* reading, you saw that SageMaker Studio sits at the top of the console sidebar and serves as the integrated development environment where most hands-on ML work happens. But Studio is not a single-user tool you simply open and start typing in. Before anyone on your team can launch a notebook, AWS needs to know *how* the workspace is organized, *who* is allowed in, and *what* resources each person can reach. That is the job of **Domains** and **User Profiles**.

Think of a Domain as a shared office floor plan. The floor plan defines the building (your VPC and network), the security desk (the default IAM execution role), and the shared storage closet (an Amazon EFS volume). A User Profile is like an assigned desk on that floor: it gives an individual a personal home directory within the shared storage and can optionally override the default security settings with tighter or broader permissions. Everyone on the floor shares the same building infrastructure, but each person's desk is theirs alone.

If you recall the *ML Lifecycle & Reproducibility* reading from the Applied ML Foundations module, a key reproducibility practice is ensuring that every team member works in a consistent, governed environment rather than ad-hoc local setups. Domains are how SageMaker enforces that idea at the infrastructure level. Instead of each data scientist installing different library versions on different laptops, the Domain provides a uniform workspace where compute, storage, and permissions are centrally managed.

This reading explains what Domains and User Profiles are, walks you through creating them in the AWS Management Console, and clarifies the differences between Studio, Studio Classic, and Canvas so you can choose the right interface for different user personas.

## Core Concepts

### Studio vs. Studio Classic vs. Canvas

Before you create a Domain, it helps to understand the three interfaces AWS offers under the SageMaker umbrella, because the Domain configuration determines which ones your team can access.

**SageMaker Studio** is the current-generation IDE. It provides a JupyterLab-based environment with integrated access to experiments, training jobs, model registry, pipelines, and debugging tools. Studio is where ML engineers and data scientists spend most of their time. It supports multiple open applications (notebooks, terminals, code editors) in a single browser tab and ties directly into the Domain's shared storage and execution roles.

**SageMaker Studio Classic** is the previous-generation Studio experience. It used a different application architecture (one JupyterServer per user with separate kernel gateway apps). AWS has been migrating users to the new Studio, but you may encounter Studio Classic in older accounts or documentation. The key difference is architectural: Classic ran each application as a separate SageMaker app instance, while the new Studio uses a more streamlined JupyterLab-based interface that is faster to start and easier to manage.

**SageMaker Canvas** is a no-code, visual interface designed for business analysts and domain experts who need to build and evaluate models without writing Python. Canvas provides a point-and-click workflow for importing data, selecting a problem type (classification, regression, forecasting), training models, and generating predictions. It runs within the same Domain, so it shares the same governance and storage infrastructure, but it is aimed at users who are not comfortable in a code environment.

| Interface | Target User | Code Required | Primary Use Case |
| :--- | :--- | :--- | :--- |
| SageMaker Studio | Data scientists, ML engineers | Yes (Python, notebooks) | Full ML lifecycle: EDA, training, deployment, pipelines |
| Studio Classic | Legacy users, existing setups | Yes (Python, notebooks) | Same as Studio; older architecture |
| SageMaker Canvas | Business analysts, domain experts | No (visual, point-and-click) | Quick model building and predictions without code |

All three interfaces share the same underlying Domain. The Domain is the organizational wrapper; the interface is how a user interacts with it.

### What a Domain contains

A Domain bundles several configuration decisions together:

**VPC and network settings.** The Domain is associated with a Virtual Private Cloud (VPC) and specific subnets. This determines the network boundary for all Studio activity. Data traffic between Studio notebooks and S3, for example, stays within this network configuration. You do not need to be a networking expert to create a Domain for learning purposes -- the console offers a default VPC option -- but in production environments, teams use custom VPCs and private subnets to enforce data isolation.

**Default execution role.** Every Domain has a default IAM execution role that determines what AWS resources Studio users can access (S3 buckets, training APIs, endpoint creation). Individual User Profiles can override this role if certain team members need broader or narrower permissions. The *IAM & Least-Privilege Practices* topic later in this module covers execution roles in detail; for now, know that the Domain's default role is the baseline.

**Shared storage (Amazon EFS).** When you create a Domain, SageMaker provisions an Amazon Elastic File System (EFS) volume. Every User Profile in the Domain gets a home directory on this shared file system. Files you save in Studio persist across sessions because they live on EFS, not on the ephemeral compute instance running your notebook. This is similar to the version-controlled project directory concept from the AIML Foundations module, except that EFS handles the persistence automatically at the infrastructure level.

### Creating a Domain in the console

Here is the step-by-step walkthrough for creating a Studio Domain through the AWS Management Console.

1. **Navigate to SageMaker.** Sign in to the AWS Console and search for "SageMaker" in the top search bar. Click **Amazon SageMaker** to open the SageMaker console.

2. **Open the Studio section.** In the left-hand sidebar, click **Studio** (or **Admin configurations > Domains** depending on your console version). If no Domain exists yet, you will see a setup prompt or a **Create Domain** button.

3. **Choose the setup mode.** AWS offers two paths:
   - **Quick setup** creates a Domain with default VPC settings, a default execution role, and a single User Profile. This is the fastest path for learning and individual accounts.
   - **Standard setup** lets you customize the VPC, subnets, encryption, and role configurations. Use this when you need specific network isolation or compliance controls.

   For this curriculum, **Quick setup** is sufficient. Click it to proceed.

4. **Review the default configuration.** The Quick setup screen shows:
   - **Domain name:** A default name (you can change it to something recognizable like `ml-training-domain`).
   - **Execution role:** AWS will create a new role or let you select an existing one. The auto-created role typically has permissions for S3 access and SageMaker API calls. Accept the default for now; you will refine roles in the IAM topic.
   - **VPC and subnet:** Defaults to your account's default VPC. This is fine for learning.

5. **Click Submit.** Domain creation takes a few minutes. The console shows a status indicator. When the status changes to **InService**, your Domain is ready.

6. **Verify the Domain.** After creation, the console shows your Domain details page with the Domain ID, status, VPC information, and a list of User Profiles. You should see one default profile (created during Quick setup).

> **Tip:** If you need to revisit Domain settings later, navigate to **SageMaker > Admin configurations > Domains** in the sidebar. You can view, edit, or delete Domains from this page.

### Adding User Profiles

A Domain starts with one User Profile from Quick setup, but teams typically create a profile for each team member. Each profile gets its own EFS home directory and can be assigned a specific execution role.

1. **Open the Domain details page.** In the SageMaker console sidebar, go to **Admin configurations > Domains** and click your Domain name.

2. **Click Add user.** You will see a form asking for:
   - **User profile name:** A unique identifier (e.g., `data-scientist-01`, `analyst-team`).
   - **Execution role:** Choose the Domain default or select a different role for this user. For example, an analyst who only needs Canvas access might have a role with read-only S3 permissions, while an ML engineer might have a role that can launch training jobs and create endpoints.

3. **Click Submit.** The new profile appears in the User Profiles list within seconds.

This structure is what makes Domains useful for multi-user collaboration. Everyone shares the same governed infrastructure (VPC, network, storage), but each person has their own workspace and permission boundary. When a new team member joins, you add a User Profile rather than spinning up an entirely separate environment.

### Launching Studio from a User Profile

Once a Domain and User Profile exist, launching Studio takes two clicks:

1. On the **Domains** page, find your Domain and expand the User Profiles list.
2. Next to the target profile, click **Launch > Studio**.

The browser opens a JupyterLab interface. You are now inside SageMaker Studio, connected to your EFS home directory, running under your profile's execution role, and ready to create notebooks, launch terminals, or start training jobs. Everything you save here persists on EFS until you explicitly delete it.

### SDK and CLI equivalents

After you are comfortable creating Domains through the console, you can automate the process with code. The AWS CLI command `aws sagemaker create-domain` and the SageMaker Python SDK both accept the same parameters (Domain name, VPC, execution role, subnets) as the console form. Automation becomes valuable when you need to replicate Domain configurations across multiple AWS accounts or regions, but always understand the console workflow first so you know what each parameter controls.

## Connecting to Practice

This reading gives you the conceptual foundation for Domains and User Profiles. In the *Studio Domains & Profiles Video*, you will see a recorded walkthrough of the console steps described above. In the module lecture, you will set up your own Domain and explore Studio firsthand. And in the module assignment, you will work inside Studio to complete hands-on tasks.

The most useful thing you can do right now is confirm that you can navigate to **SageMaker > Admin configurations > Domains** in your console and identify whether a Domain already exists in your account. If one does, check how many User Profiles it has and which execution role it uses. If one does not, follow the Quick setup steps above to create one -- you will need it for every lab in this curriculum.

---

## Further Learning & Resources

**Documentation and reading**

- **[Amazon SageMaker Studio](https://docs.aws.amazon.com/sagemaker/latest/dg/studio.html)** - *Docs*: The official documentation covering Studio features, Domain configuration, and User Profile management in detail.
- **[SageMaker Canvas - No-Code ML](https://docs.aws.amazon.com/sagemaker/latest/dg/canvas.html)** - *Docs*: Overview of the Canvas no-code interface, useful for understanding how non-technical team members interact with the same Domain infrastructure.

**Interactive practice**

- **[AWS Hands-On: Onboard to Amazon SageMaker Studio](https://aws.amazon.com/getting-started/hands-on/machine-learning-tutorial-set-up-sagemaker-studio-account-permissions/)** - *Interactive*: A free guided walkthrough from AWS that takes you through Domain and User Profile creation step by step in your own console.
- **[AWS Skill Builder - Getting Started with SageMaker Studio](https://explore.skillbuilder.aws/learn/course/internal/view/elearning/16357/getting-started-with-amazon-sagemaker-studio)** - *Interactive*: Self-paced digital training with hands-on exercises covering Studio setup and navigation.
