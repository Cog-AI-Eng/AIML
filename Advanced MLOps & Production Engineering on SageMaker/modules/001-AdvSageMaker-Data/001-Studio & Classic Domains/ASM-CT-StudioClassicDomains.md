# Studio & Classic Domains

**Estimated Time:** 10 Minutes

## Introduction

In the foundational SageMaker skill, you created a Studio Domain using Quick Setup and launched the IDE for the first time. That was enough to run notebooks and deploy a JumpStart model. In a production environment, however, the choice between Studio and Studio Classic -- and how you configure Domains -- has real consequences for team collaboration, compute governance, and cost control.

AWS currently offers two IDE experiences under the SageMaker umbrella: **SageMaker Studio** (the newer, JupyterLab-based workspace) and **Studio Classic** (the older, custom-built IDE that many existing tutorials and documentation still reference). A third surface, **Canvas**, provides a no-code option and is covered in a later topic. Understanding the architectural differences between these environments is critical because it affects which features you can access, how compute is provisioned, and how user isolation works.

This reading covers the domain architecture at a deeper level than the foundations module, focusing on multi-user configuration, execution role scoping, and the practical trade-offs that determine which experience your team should standardize on.

## Core Concepts

### Domains as governance boundaries

A SageMaker Domain is not just a container for notebooks. It is the administrative boundary that controls:

- **User access:** Each user gets a profile with its own execution role, home directory on EFS, and compute settings.
- **Network isolation:** The Domain is bound to a VPC and specific subnets. All Studio compute (notebooks, training jobs, endpoints) operates within this network boundary.
- **Shared storage:** An Amazon EFS volume is automatically provisioned for the Domain. Every user profile mounts this volume, enabling shared datasets and artifacts within the team.

In the foundations module, Quick Setup created a Domain with defaults: the default VPC, broad S3 permissions, and a single user profile. For production teams, you would use **Custom Setup** to specify private subnets, a custom KMS key for EFS encryption, and tightly scoped execution roles per user.

To view Domain configuration after creation, navigate to **SageMaker > Admin configurations > Domains** in the console, click your Domain name, and examine the **Domain settings** and **User profiles** tabs.

### Studio vs. Studio Classic: architectural differences

**SageMaker Studio** (current) runs on standard JupyterLab. When you open Studio, SageMaker launches a JupyterLab server backed by a compute instance that you select (called a *Space*). Key characteristics:

- JupyterLab 3+ interface with standard extensions and terminal access.
- Each notebook runs in a **Space** that you create with a specific instance type and image. You can run multiple Spaces simultaneously with different instance sizes.
- Supports SageMaker-managed JupyterLab apps, Code Editor (VS Code in browser), and Canvas within the same Domain.
- Lifecycle configurations (startup scripts) can be attached to Spaces for automated environment setup.

**Studio Classic** uses a custom-built IDE that predates the JupyterLab migration. Key characteristics:

- Single application per user profile (the "default app") running on a fixed instance type.
- KernelGateway apps provision separate compute for each notebook kernel. Switching instance types requires stopping and restarting the kernel.
- Many older AWS tutorials and sample notebooks were written for this interface.
- AWS has announced that Studio Classic is in maintenance mode; no new features are being added.

### When each experience matters

| Factor | Studio (Current) | Studio Classic |
| :--- | :--- | :--- |
| Interface | Standard JupyterLab | Custom IDE |
| Compute model | Spaces (explicit, user-created) | KernelGateway apps (implicit) |
| Multi-instance | Multiple Spaces per user | One default app + KernelGateway apps |
| Extension support | Standard JupyterLab extensions | Limited, custom extension model |
| New feature availability | Active development | Maintenance mode |
| Existing tutorial compatibility | Growing | Extensive (legacy docs) |

For new projects, AWS recommends Studio. The only practical reason to use Studio Classic today is if your team relies on a specific legacy extension or internal tool that has not been ported to the current Studio interface.

### Configuring user profiles for team collaboration

Each Domain supports multiple user profiles. In a team setting, you create one profile per engineer. Each profile gets:

- Its own **execution role** (the IAM role SageMaker assumes for that user's jobs). This lets you scope permissions per person -- for example, a junior analyst might have read-only S3 access while a senior engineer has full training and deployment permissions.
- Its own **EFS home directory** (a subdirectory on the shared EFS volume, isolated by UID/GID).
- Configurable **default Spaces settings** that control which instance types the user can select.

To create a user profile in the console: **SageMaker > Domains > [your domain] > User profiles > Add user**. You assign a name and select or create an execution role. This is the same flow you saw in the foundations lab, but in a production setting you would create the execution role in IAM first with least-privilege policies and then attach it here.

### Cost considerations

Studio compute is billed per instance-hour while instances are running. The most common cost mistake in multi-user Domains is leaving Spaces or KernelGateway apps running overnight. To mitigate this:

- **Lifecycle configurations** can include auto-shutdown scripts that stop idle instances after a configurable timeout.
- **Domain-level settings** can restrict the instance types available to users, preventing anyone from accidentally selecting a `ml.p3.8xlarge` for a notebook session.
- In the console, check **SageMaker > Domains > [your domain] > Domain settings > Space default settings** to see and edit instance type restrictions.

## Connecting to Practice

This reading establishes the Domain architecture you will work within for every lab in this skill unit. The next topic, *Data Wrangler Flows*, introduces one of the Studio-native tools for data preparation. The module lecture will have you configure a Domain with custom settings (private subnets, scoped roles) rather than Quick Setup, and the assignment will require you to demonstrate multi-user profile management.

## Further Learning & Resources

**Documentation and reading**

- **[SageMaker Studio Overview](https://docs.aws.amazon.com/sagemaker/latest/dg/studio.html)** - *Docs*: Official documentation covering Studio architecture, Spaces, and lifecycle configurations.
- **[Migrate from Studio Classic to Studio](https://docs.aws.amazon.com/sagemaker/latest/dg/studio-classic-migration.html)** - *Docs*: AWS migration guide for teams transitioning from Studio Classic to the current Studio experience.

**Interactive practice**

- **[SageMaker Studio Workshop](https://catalog.workshops.aws/sagemaker-studio/en-US)** - *Interactive*: AWS-hosted workshop that walks through Studio Domain setup with custom networking and multi-user configuration.
