# Canvas vs. Autopilot

**Estimated Time:** 10 Minutes

## Introduction

The previous two topics covered Canvas and Autopilot individually. Both use automated ML under the hood, and Canvas's Standard Build literally runs an Autopilot job. So why do both exist, and when should you recommend one over the other?

The answer is audience. Canvas and Autopilot serve different users within the same organization, and understanding the boundary between them is essential for designing an enterprise ML strategy that includes both technical and non-technical contributors. This reading provides a direct comparison across every dimension that matters when advising teams.

## Core Concepts

### Side-by-side comparison

| Dimension | Canvas | Autopilot |
| :--- | :--- | :--- |
| **Primary user** | Business analysts, domain experts, PMs | Data scientists, ML engineers |
| **Interface** | Visual, no-code (point and click) | Console form + generated Jupyter notebooks |
| **Code access** | None (code is hidden) | Full: generated notebooks are editable Python |
| **Data import** | GUI-based: S3, Redshift, Athena, Snowflake, local upload | S3 URI only (specified in the job config) |
| **Data transforms** | Built-in visual transforms (rename, filter, join) | Automatic (Autopilot decides preprocessing) |
| **Training modes** | Quick Build (fast sample) and Standard Build (full Autopilot) | Ensembling and HPO |
| **Model output** | Deployed endpoint or batch prediction CSV | Model artifact in S3, optional endpoint deployment |
| **Explainability** | Column impact ranking (simplified) | SHAP summary plots + full explainability report |
| **Customization** | None -- accept the model as built | Full -- download generated notebooks, modify, retrain |
| **Sharing** | Share model to Studio user profile for code-level access | Models are SDK objects, shareable via Model Registry |
| **Deployment** | One-click from Canvas UI | Console or SDK deployment |
| **Cost** | Canvas session charges + training compute | Training compute only |

### Decision framework

Use Canvas when:

- The person building the model does not write code.
- The goal is rapid validation ("can this data predict this outcome?") before investing engineering effort.
- The use case is well-scoped tabular prediction (classification or regression) with clean data.
- You want to empower domain experts to answer their own ML questions without filing tickets with the data science team.

Use Autopilot directly when:

- The user is comfortable reading and modifying Python code.
- You need full control over preprocessing, algorithm selection, or training configuration after the initial search.
- You want to integrate the model into a SageMaker Pipeline or custom MLOps workflow.
- You need the generated Candidate Notebook as a learning tool or starting point for custom model development.
- The dataset requires complex preprocessing that Canvas's built-in transforms cannot handle.

### The handoff pattern

In practice, Canvas and Autopilot work together in a handoff workflow:

1. **Business analyst in Canvas:** Imports data, explores it visually, builds a Quick Build model to validate that the prediction task is feasible. Reviews column impact to confirm the features make business sense.
2. **Analyst shares model:** Uses Canvas's Share feature to send the model to a data scientist's user profile.
3. **Data scientist in Studio:** Opens the shared model, examines the Autopilot-generated notebooks, and either deploys the model as-is or refines it with custom code.
4. **Engineer deploys to production:** Takes the final model, registers it in Model Registry, deploys it to an endpoint, and integrates it into the application.

This pattern respects each team member's expertise: the business analyst defines *what* to predict and validates that the data supports it, while the engineer decides *how* to productionize and govern it.

### Common misconceptions

**"Canvas is just Autopilot with a GUI."** Partially true. Canvas uses Autopilot for Standard Build, but it also provides its own data exploration, transformation, and batch prediction features that do not exist in the Autopilot console flow. Canvas is a complete end-to-end experience; Autopilot is a training service.

**"Autopilot always produces better models than Canvas."** Not necessarily. Standard Build in Canvas runs the same Autopilot job. Quick Build produces a weaker model because it uses a sample and fewer algorithm trials, but Standard Build output is identical to running Autopilot directly.

**"Canvas cannot be used for production."** Canvas can deploy models to real-time endpoints and run batch predictions. For simple use cases with stable data, a Canvas-deployed model can serve production traffic. The limitation is operational: Canvas does not support Pipeline integration, monitoring, or automated retraining. For production systems that need MLOps capabilities, the model should be transitioned to a code-managed workflow.

### Governance implications

Both Canvas and Autopilot operate within the SageMaker Domain's governance boundary:

- IAM execution roles control data access and resource creation.
- VPC settings control network isolation.
- CloudTrail logs capture all API calls for auditability.
- Model artifacts are stored in S3 and can be registered in Model Registry regardless of whether they originated from Canvas or Autopilot.

The key governance difference is visibility: Autopilot models have full code lineage (you can trace every preprocessing step and hyperparameter), while Canvas models are opaque unless shared to Studio for inspection. For regulated industries, this transparency requirement often means Autopilot is preferred for production models even if Canvas was used for initial validation.

## Connecting to Practice

This topic completes the data and AutoML section of Module 1. You now have a full picture of the SageMaker data preparation and automated model building ecosystem: Data Wrangler for visual data transforms, Feature Store for centralized feature management, Canvas for no-code model building, and Autopilot for transparent AutoML. The module lecture ties these together in a hands-on session, and the assignment will require you to demonstrate both a Canvas Quick Build and an Autopilot HPO job on the same dataset, comparing the results.

## Further Learning & Resources

**Documentation and reading**

- **[Choosing Between Canvas and Autopilot](https://docs.aws.amazon.com/sagemaker/latest/dg/autopilot-automate-model-development.html)** - *Docs*: AWS guidance on when to use each service.
- **[SageMaker for Business Analysts](https://aws.amazon.com/sagemaker/canvas/)** - *Docs*: Canvas product page with use case examples and customer stories.

**Interactive practice**

- **[Canvas and Autopilot Comparison Lab](https://catalog.workshops.aws/canvas-immersion-day/en-US)** - *Interactive*: Workshop that walks through both Canvas and Autopilot on the same dataset to compare workflows and results.
