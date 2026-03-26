# Feature Store Lineage Integration

**Estimated Time:** 10 Minutes

## Introduction

In Module 1, you learned that Feature Store provides a centralized repository for ML features with Online and Offline stores. In the previous topics in this module, you learned that Lineage Tracking records the provenance chain from data to model. Feature Store Lineage Integration connects these two systems: when a training job reads features from the Offline Store, SageMaker automatically creates lineage associations that link the Feature Group to the model, establishing a traceable path from feature definitions to deployed predictions.

This integration answers questions that neither system can answer alone: "Which features were used to train this model?" "If we change the computation for feature X, which models need retraining?" "Which Feature Groups contributed to the model currently serving in production?"

## Core Concepts

### Automatic lineage for Feature Store

When SageMaker services interact with Feature Store, lineage entities are created automatically:

1. **Feature Group as Artifact:** Each Feature Group is represented as a lineage Artifact. The artifact's properties include the Feature Group name, ARN, and feature definitions.
2. **Ingestion as Action:** Batch ingestion jobs that write to Feature Store are recorded as Actions, linked to the source data Artifacts and the Feature Group Artifact.
3. **Training data from Offline Store:** When a training job reads data from the Offline Store's S3 location (registered as a Glue table), SageMaker creates an association from the Feature Group Artifact to the Training Job Action.

The resulting lineage chain: `Raw Data --> Ingestion Job --> Feature Group --> Training Job --> Model Artifact --> Endpoint`.

### Querying feature-to-model lineage

To find which Feature Groups contributed to a specific model:

1. Start at the model artifact in the lineage graph.
2. Traverse upstream through the Training Job Action.
3. The input Artifacts will include S3 paths pointing to the Offline Store's Glue table partitions. These Artifacts are associated with the Feature Group Artifact.

In the console, navigate to the model's Lineage tab and follow the upstream chain. The Feature Group will appear as a node in the graph.

Programmatically, use the SDK's `lineage.query()` method with an upstream direction from the model artifact ARN, filtering for Artifacts of type `FeatureGroup`.

### Querying model impact for feature changes

The reverse query -- finding all models affected by a Feature Group change -- is equally important:

1. Start at the Feature Group lineage Artifact.
2. Traverse downstream through all associated Training Job Actions.
3. Each Training Job's output Model Artifact represents an affected model.
4. Continue downstream to find associated Endpoints.

This query is critical for change management: before modifying a feature computation, you can identify every model and endpoint that depends on that feature and plan retraining accordingly.

### Point-in-time correctness

Feature Store's event timestamp mechanism and Lineage Tracking work together to ensure temporal correctness:

- Each feature record in the Offline Store includes an event timestamp.
- When constructing training data, you use point-in-time joins to ensure each training example sees only features that were available at the time of the event (preventing data leakage from future feature values).
- The lineage graph records which time range of the Offline Store was used for training, enabling you to reproduce the exact training dataset later.

This combination of temporal feature access and lineage tracking is what makes Feature Store suitable for regulated environments where you must demonstrate that training data was constructed without lookahead bias.

### Console walkthrough

To view Feature Store lineage in the console:

1. Navigate to **SageMaker > Feature Store > Feature groups > [your group]**.
2. Click the **Lineage** tab.
3. The graph shows downstream entities: Processing Jobs that ingested data into the group, Training Jobs that read from it, and Models produced by those jobs.
4. Click any node to see its details, including timestamps, ARNs, and parameters.

### Cross-account lineage

In enterprise environments, Feature Groups may be shared across AWS accounts using AWS Resource Access Manager (RAM). When a training job in Account B reads from a Feature Group owned by Account A, SageMaker creates lineage associations that cross the account boundary. This enables centralized lineage queries from a governance account that can trace any model back to its feature sources regardless of which account trained it.

## Connecting to Practice

This topic ties Feature Store (Module 1) to Lineage Tracking (this module) into a unified provenance system. The next topic, *Reproducibility Patterns*, covers how to use Experiments, Lineage, and Feature Store together to guarantee that any past model can be fully reproduced. The module assignment will require you to demonstrate a feature-to-model lineage query and a model-impact-for-feature-change query.

## Further Learning & Resources

**Documentation and reading**

- **[Feature Store and Lineage](https://docs.aws.amazon.com/sagemaker/latest/dg/feature-store-lineage.html)** - *Docs*: Guide to Feature Store lineage integration and query patterns.
- **[Cross-account Feature Store Access](https://docs.aws.amazon.com/sagemaker/latest/dg/feature-store-cross-account.html)** - *Docs*: Configuration for sharing Feature Groups across AWS accounts.

**Interactive practice**

- **[Feature Store Lineage Notebook](https://github.com/aws/amazon-sagemaker-examples/tree/main/sagemaker-featurestore)** - *Interactive*: Sample notebook demonstrating Feature Group lineage queries and provenance reporting.
