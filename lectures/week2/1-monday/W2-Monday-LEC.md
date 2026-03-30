# Week 2 Monday -- Deploy, Evaluate, and Sequence Models

**Total Duration:** 185 Minutes (3 Stages)
**Consolidated Activities:**
- SM Deployment: Model Registry & Versioning, Approval Workflows, Real-time Inference Endpoints, Invoking Endpoints
- SM MLOps: MLOps & CI/CD Principles, Pipelines, DAGs & Versioning
- Deep Learning: RNNs for Sequence Data, LSTMs & GRUs (completes DeepLearning module)

| Block | Content | Minutes |
|-------|---------|---------|
| Stage 1 | Model Registry, Deployment, and Evaluation | 60 |
| Break 1 | Stretch / Questions | 5 |
| Stage 2 | RNNs, LSTMs, and GRUs | 55 |
| Break 2 | Stretch / Questions | 5 |
| Stage 3 | CNN Evaluation + MLOps Concepts | 45 |
| Buffer | Open Q&A, Summary, Tuesday Preview | 15 |

---

## Lecture Overview

**Unified Scenario -- FraudShield Risk Analytics**

Associates continue as ML engineers at FraudShield. Friday they set up SageMaker, trained a Random Forest on tabular fraud data via Script Mode, trained a CNN image classifier via JumpStart, and explored the encoder-decoder pattern. Today is the payoff: they deploy those models, invoke them with held-out data, and evaluate performance with real metrics. In the middle, they open the LSTM black box from Friday's encoder-decoder. The day closes with MLOps concepts that frame how this manual workflow becomes automated.

1. **"How do we get our trained model into production?"** (Model Registry, real-time endpoints)
2. **"How well does the model actually perform?"** (Precision, recall, F1, confusion matrix on validation data)
3. **"How does the LSTM actually work?"** (Gates, cell state, GRU alternative)
4. **"How do we automate the train-deploy cycle?"** (MLOps, pipelines, DAGs)

Each stage pairs a SageMaker block with a Deep Learning or Evaluation block.

---

## Pre-Lecture Setup

### Instructor Checklist

- [ ] Friday's model artifacts confirmed in S3 (both RF and CNN `model.tar.gz`)
- [ ] `validation.csv` available locally or in S3 from Friday
- [ ] SageMaker execution role ARN ready
- [ ] Companion lecture notebook (`W2-Monday-notebook.ipynb`) open and tested
- [ ] AWS account with SageMaker access verified
- [ ] Budget from Friday still active
- [ ] This instructor guide open in a second tab

### Student Prerequisites

- [ ] Completed readings: RNNs for Sequence Data CT, LSTMs & GRUs CT, Model Registry & Versioning CT, Approval Workflows CT, Real-time Inference Endpoints CT, Invoking Endpoints CT, MLOps & CI/CD Principles CT, Pipelines/DAGs & Versioning CT
- [ ] Friday's notebook completed (models trained, artifacts in S3)
- [ ] AWS credentials configured, SageMaker SDK installed

---

# STAGE 1 -- Model Registry, Deployment, and Evaluation (60 min)

> **Goal:** Register Friday's trained model, deploy it to a real-time endpoint, invoke it with validation data, and compute performance metrics. Associates see their model predict for the first time.

**Exit Criteria Addressed:**
- Register a model version in the SageMaker Model Registry with metadata and approval status (Required)
- Describe the approval workflow and governance gates for model deployment (Required)
- Deploy a model to a real-time inference endpoint using the SageMaker Python SDK (Required)
- Invoke an endpoint and interpret predictions (Required)

### Instructor Opening (3 minutes -- talk, no code)

> "Friday you trained two models: a Random Forest for fraud detection and a CNN for image classification. Both produced a `model.tar.gz` artifact in S3. But a model sitting in S3 is useless to the business. Today we bridge that gap. By the end of Stage 1, you will have a live endpoint serving predictions, and you will know exactly how well your model performs on data it has never seen."

---

## STEP 1 -- Verify Friday's Artifacts (5 minutes)

**Pacing: live code in notebook.** Associates run the setup cells and the artifact verification cell.

Narrate: "We reconnect to our SageMaker session and verify that the model artifacts from Friday still exist in S3. If anyone's artifacts are missing, the notebook includes a fallback cell that retrains quickly."

[PAUSE -- Verify every student has at least the RF model artifact before proceeding.]

---

## STEP 2 -- Model Registry Concepts (8 minutes)

**Pacing: conceptual with notebook markdown, then code.**

> "Think of Model Registry as Git for models. A Model Package Group is like a repository. Each version is a commit -- it captures the artifact, the container image, and metadata. The approval status acts as a merge gate: nothing goes to production without explicit approval."

Key points:
- Model Package Group = logical model repository
- Model Package = specific version (artifact + container + metadata)
- Approval status: PendingManualApproval -> Approved -> deployed
- Tags capture lineage (training job, accuracy, framework)

Run the registration cell. Walk through each parameter.

---

## STEP 3 -- Approve and Deploy (10 minutes)

**Pacing: live code, step by step.**

> "We approve the model version -- in production this would follow a review. Then we deploy. SageMaker's three-object pattern: Model + Endpoint Configuration + Endpoint. The SDK's `.deploy()` method handles all three."

While deployment runs (3-5 minutes), discuss:
- Instance types: `ml.m5.xlarge` (general purpose, Free Tier eligible)
- Billing starts when the endpoint is InService
- In production: auto-scaling, multi-variant endpoints

---

## STEP 4 -- Invoke and Evaluate (15 minutes)

**Pacing: live code. This is the payoff moment.**

> "Now we send the validation data through the endpoint. This is the first time the model predicts on data it was not trained on."

Walk through:
1. Load `validation.csv`, separate features from target
2. Send CSV rows to the endpoint via `invoke_endpoint`
3. Collect predictions
4. Compute metrics: accuracy, precision, recall, F1
5. Generate and visualize confusion matrix

> "The confusion matrix shows where the model succeeds and fails. For fraud detection, false negatives are expensive -- a missed fraud costs real money. That is why precision and recall matter more than raw accuracy."

**Discussion Prompt:** "If precision is 0.95 and recall is 0.60, what does that mean for FraudShield's business? Would you deploy this model?" (High precision = few false alarms, but low recall = 40% of fraud gets through.)

---

## STEP 5 -- Reading the Classification Report (7 minutes)

**Pacing: reference notebook output, discuss.**

Show `classification_report` output. Connect to the Evaluation module concepts (precision, recall, F1) they read earlier in the week.

---

## STEP 6 -- Cleanup RF Endpoint (5 minutes)

**Pacing: live code. Mandatory.**

> "Delete the endpoint immediately. You will deploy the CNN model in Stage 3, so we clean up the RF endpoint now to minimize costs."

Delete order: endpoint -> endpoint config -> model.

[PAUSE FOR BREAK - 5 MINS]

---

# STAGE 2 -- RNNs, LSTMs, and GRUs (55 min)

> **Goal:** Understand how recurrent architectures process sequential data, why vanilla RNNs fail on long sequences, and how LSTM/GRU gating solves the problem.

**Exit Criteria Addressed:**
- Describe how RNNs process sequential data with shared weights across time steps (Required)
- Explain the vanishing gradient problem and its impact on long-range dependencies (Required)
- Diagram the LSTM gated memory cell including forget, input, cell state, and output gates (Required)
- Compare GRU simplified gating to LSTM (Required)

### Instructor Opening (2 minutes)

> "Friday we used an LSTM inside our encoder-decoder, but we treated it as a black box. Today we open that box. You will understand exactly why LSTMs exist, what problem they solve, and how their gates work. This completes the Deep Learning module."

---

## STEP 7 -- From Images to Sequences (5 minutes)

**Pacing: conceptual, notebook markdown.**

> "CNNs exploit spatial structure in images. But many data types have temporal structure: transaction sequences, text, time series. We need architectures that process data step by step, maintaining a memory of what came before."

---

## STEP 8 -- Synthetic Sentiment Data (5 minutes)

**Pacing: run code cell.**

Generate synthetic sentiment data. Explain:
- Positive sequences use higher token values, negative use lower
- Noise tokens make the task non-trivial
- This isolates the architecture from real NLP complexity

---

## STEP 9 -- Vanilla RNN Forward Pass (10 minutes)

**Pacing: code walkthrough, line by line.**

NumPy implementation of a single-layer RNN. Key teaching points:
- Same weight matrices at every time step (weight sharing)
- Hidden state carries information forward
- tanh activation bounds the hidden state

> "The RNN reads the sequence one token at a time, updating its hidden state. At the end, the final hidden state summarizes the entire sequence. We pass it through a linear layer to predict sentiment."

---

## STEP 10 -- Vanishing Gradient Problem (8 minutes)

**Pacing: run visualization cell, discuss.**

> "The gradient at time step 0 is 0.5% of the gradient at step 49. The network cannot learn from early tokens. This is not a bug -- it is a mathematical property of repeated matrix multiplication through tanh."

**Discussion Prompt:** "What happens if the singular value is 1.1 instead of 0.9?" (Exploding gradients -- gradient norms grow exponentially.)

---

## STEP 11 -- LSTM Gated Memory (12 minutes)

**Pacing: code walkthrough with markdown diagrams.**

Walk through the LSTM cell step by step:
1. Forget gate: what to erase from cell state
2. Input gate: what new information to write
3. Cell state update: addition (not multiplication) -- gradients flow
4. Output gate: what to expose as hidden state

> "The key insight is the cell state highway. Information flows through addition, not multiplication. Gradients do not decay because they are not repeatedly multiplied by the weight matrix."

---

## STEP 12 -- GRU Simplified Gating (5 minutes)

**Pacing: code cell + comparison markdown.**

> "GRU combines forget and input gates into a single update gate. Fewer parameters, often comparable performance. Choose GRU when you need faster training; choose LSTM when you need maximum capacity."

---

## STEP 13 -- Architecture Comparison (8 minutes)

**Pacing: run comparison visualization + summary table.**

Show the representative training curves and the summary comparison table. Tie back to Friday's encoder-decoder:

> "Now you know what is inside the LSTM boxes in the encoder and decoder. The encoder LSTM reads the source sequence, using its cell state to preserve long-range information. The decoder LSTM generates the target sequence, initialized with the encoder's final states."

[PAUSE FOR BREAK - 5 MINS]

---

# STAGE 3 -- CNN Evaluation + MLOps Concepts (45 min)

> **Goal:** Deploy and evaluate the CNN model from Friday, then transition to MLOps concepts that frame how the manual workflow becomes automated.

**Exit Criteria Addressed:**
- Deploy a JumpStart model to an endpoint and evaluate its predictions (Required)
- Define MLOps and describe its three pillars: automation, versioning/governance, and monitoring/feedback (Required)
- Describe SageMaker Pipelines as DAGs with conditional logic (Required)

### Instructor Opening (2 minutes)

> "We evaluated the tabular fraud model in Stage 1. Now let us evaluate the CNN image classifier. Then we step back and ask: how do we stop doing all of this manually?"

---

## STEP 14 -- Deploy and Evaluate CNN (15 minutes)

**Pacing: live code. Deploying the JumpStart model takes a few minutes.**

Deploy the CNN model from Friday. While it deploys, review:
- JumpStart vs Script Mode deployment differences
- Image input format (JPEG/PNG bytes)

Once live, invoke with CIFAR-10 test images. Compute per-class accuracy. Visualize the confusion matrix.

> "This confusion matrix tells us which classes the model confuses. If 'cat' and 'dog' are often swapped, that suggests the model needs more diverse training data for those classes."

---

## STEP 15 -- MLOps: From Manual to Automated (10 minutes)

**Pacing: conceptual with notebook diagrams.**

Three pillars of MLOps:
1. Automation (pipelines, triggers)
2. Versioning & Governance (Model Registry, approval gates)
3. Monitoring & Feedback (drift detection, retraining)

Compare to traditional CI/CD:

| CI/CD | ML CI/CD |
|-------|----------|
| Code commit triggers build | Data change or code change triggers retrain |
| Unit tests gate merge | Model evaluation gates registration |
| Binary artifact | Model artifact |
| App monitoring | Model + data monitoring |

---

## STEP 16 -- SageMaker Pipelines Concepts (10 minutes)

**Pacing: conceptual with diagrams.**

> "A pipeline is a DAG -- directed acyclic graph. Each node is a step: preprocess, train, evaluate, register. Edges are data dependencies. SageMaker infers the graph from how steps reference each other's outputs."

Show the conceptual pipeline for FraudShield:
```
Preprocess -> Train -> Evaluate -> [F1 >= 0.85?] -> Register
```

Key concepts: ConditionStep, PropertyFile, JsonGet, ParameterString.

---

## STEP 17 -- Mandatory Cleanup (5 minutes)

**Pacing: live code. EVERY student must complete this.**

> "Delete the CNN endpoint. Check billing. Make this a habit."

Delete: endpoint -> endpoint config -> model. Verify both endpoints (RF and CNN) are gone.

---

## Wrap-up & Q&A Buffer (15 minutes)

### Summary (5 minutes)

> "Today you accomplished three things. First, you deployed your Friday model to a live endpoint, invoked it with validation data, and computed real performance metrics -- precision, recall, F1, confusion matrix. You now know exactly how well your model performs. Second, you opened the LSTM black box: forget gate, input gate, cell state highway, output gate. You understand why LSTMs solve the vanishing gradient problem that cripples vanilla RNNs. Third, you saw how MLOps automates the manual workflow we did today -- pipelines replace your click-by-click process with reproducible DAGs."

### Tuesday Preview (2 minutes)

> "Tuesday covers data engineering: Feature Store, Data Wrangler, Canvas, and Autopilot. The question shifts from 'how do we deploy a model' to 'how do we manage the features that feed it.' Read the Advanced SageMaker Data CTs before Tuesday."

### Open Q&A (8 minutes)

---

## Instructor Notes -- Common Issues

| Issue | Resolution |
|-------|-----------|
| Friday's model artifact not in S3 | Use the fallback retrain cell in the notebook. Takes ~5 minutes. |
| `invoke_endpoint` returns `ModelError` | Check that the CSV payload matches the training schema (6 features, no header). |
| Endpoint stuck on "Creating" | Normal for first deployment in a session. Container pull takes 3-5 min. |
| Predictions are all the same class | Likely the model was trained on unbalanced data without class weights. Discuss as a teaching moment. |
| Student forgets to delete endpoint | Walk over immediately. Check billing together. |
| `ValidationError` on invoke | Content type mismatch. Verify `text/csv` for sklearn model. |
| JumpStart CNN deployment fails | Check that the model ID and version match. JumpStart catalog changes. |
| LSTM/GRU code cells slow | Pure NumPy implementations -- expected. Emphasize this is pedagogical, not production. |
