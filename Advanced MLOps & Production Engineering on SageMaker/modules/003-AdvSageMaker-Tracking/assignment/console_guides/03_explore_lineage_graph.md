# Guide 3: Explore the Lineage Graph

This guide walks you through navigating the SageMaker Lineage Graph for a trained FraudShield model. You will trace the full provenance chain from the model artifact back to the original training dataset, identifying each lineage entity type along the way.

---

## Steps

### Step 1 -- Navigate to the Model in Studio
1. In SageMaker Studio, click **Home** in the left sidebar.
2. Click **Models** (or navigate via **Deployments > Models** depending on Studio version).
3. Alternatively, go to **Experiments**, open `fraudshield-fraud-detection`, and click the best-performing run from Guide 2.
4. From the run detail page, locate the associated model artifact or training job link.

### Step 2 -- Open the Lineage Graph
1. From the training job or model detail page, look for a **Lineage** tab or a **View lineage** link.
2. Click it to open the Lineage Graph visualization. The graph renders as an interactive node-and-edge diagram.
3. If the Lineage tab is not directly visible, navigate to **Experiments > [your experiment] > [best run]** and look for the lineage option in the run's detail panel.

### Step 3 -- Identify the Model Artifact Node
1. In the lineage graph, locate the **Model** node. It is typically on the right side of the graph.
2. Click the model node. A detail panel opens showing:
   - Artifact type: **Model**
   - S3 URI: the path to `model.tar.gz`
   - Creation time
3. This is the endpoint of the lineage chain -- the final output you want to trace backward.

### Step 4 -- Trace Backward Through the Training Action
1. Follow the edge from the model node backward (left) to the **Training Job** node.
2. Click the training job node. The detail panel shows:
   - Entity type: **Action** (the training job is an action that produced the model)
   - Training job name and ARN
   - Hyperparameters used
   - Instance type and training duration
3. Note how the training job connects the input data to the output model.

### Step 5 -- Identify the Input Data Artifact
1. Follow the edges further backward from the training job node to the **Dataset** nodes.
2. You should see one or two dataset artifact nodes (train channel and validation channel).
3. Click each dataset node. The detail panel shows:
   - Entity type: **Artifact** (type: DataSet)
   - S3 URI of the training/validation data
4. This completes the backward trace: Dataset --> Training Job --> Model.

### Step 6 -- Identify Context and Association Entities
1. Look for any **Context** nodes in the graph. Contexts represent grouping entities such as the Experiment or the Experiment Run (Trial).
2. Click a context node and note its type (e.g., Experiment, ExperimentTrialComponent).
3. Examine the edges (lines) connecting nodes. Each edge represents an **Association** with a type:
   - **Produced**: the training job produced the model artifact.
   - **ContributedTo**: the dataset contributed to the training job.
   - **AssociatedWith**: the training job is associated with the experiment context.
4. Document each entity type you observe: Context, Action, Artifact, and the Association types that connect them.

### Step 7 -- Summarize the Full Lineage Chain
1. Write down the complete lineage path:
   - **Artifact (DataSet):** S3 path to training data
   - **Action (Training Job):** Job name, hyperparameters, instance type
   - **Artifact (Model):** S3 path to model.tar.gz
   - **Context (Experiment):** Experiment name and run name
2. Confirm that every step in the chain is traceable and that no information is missing.

---

## Presentation Checkpoint
Be prepared to show:
- The Lineage Graph visualization with all nodes and edges visible.
- The detail panel for each node type: Artifact (Dataset), Action (Training Job), Artifact (Model), and Context (Experiment).
- A verbal walk-through of the full lineage chain from dataset to model.

---

## Key Concepts
- **Lineage Graph:** A directed acyclic graph that records the provenance of ML artifacts, showing how datasets, processing steps, training jobs, and models are connected.
- **Artifact:** A lineage entity representing a data object, such as a dataset in S3 or a trained model file.
- **Action:** A lineage entity representing a computation, such as a training job or a processing job that transforms inputs into outputs.
- **Context:** A lineage entity representing a grouping concept, such as an experiment or a project, that organizes related actions and artifacts.
- **Association:** An edge in the lineage graph that defines the relationship between two entities (e.g., "Produced", "ContributedTo", "AssociatedWith").
