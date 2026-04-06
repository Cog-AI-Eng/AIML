# Week 2 -- Interview-Ready Cheat Sheet

> Covers Monday through Friday: Deployment and Model Registry, Feature Store and AutoML, Transformers and Experiments, Built-in Algorithms and HPO, and Advanced Inference plus Model Monitoring. Aligns with the **Advanced MLOps and Production Engineering on SageMaker** skill unit for production patterns.
> Every concept follows the same template: **What** / **Why** / **How** / **Where** / **Gotcha**.

---

## Table of Contents

- [Monday -- Deploy, Evaluate, Sequences, and MLOps Framing](#monday----deploy-evaluate-sequences-and-mlops-framing)
- [Tuesday -- Advanced SageMaker Data Engineering](#tuesday----advanced-sagemaker-data-engineering)
- [Wednesday -- Transformers, Pre-trained Models, Experiments, and Lineage](#wednesday----transformers-pre-trained-models-experiments-and-lineage)
- [Thursday -- Built-in Algorithms and Hyperparameter Optimization](#thursday----built-in-algorithms-and-hyperparameter-optimization)
- [Friday -- Inference Patterns and Model Monitoring](#friday----inference-patterns-and-model-monitoring)
- [Advanced MLOps on SageMaker (Evolv skill -- production and cost)](#advanced-mlops-on-sagemaker-evolv-skill----production-and-cost)
- [Supplemental Deep Dives (SDK Patterns and Interview Sound Bites)](#supplemental-deep-dives-sdk-patterns-and-interview-sound-bites)
- [Quick-Reference Formula Card](#quick-reference-formula-card)

---

# Monday -- Deploy, Evaluate, Sequences, and MLOps Framing

| # | Topic | One-Liner |
|---|-------|-----------|
| 1 | Model Registry | Git for models: package groups, versions, metadata, and approval gates |
| 2 | Real-Time Deployment and Invocation | Model plus Endpoint Config plus Endpoint; HTTPS with correct content type |
| 3 | Endpoint Evaluation | Precision, recall, F1, confusion matrix on validation data |
| 4 | Vanilla RNNs | Shared weights over time; hidden state carries the sequence summary |
| 5 | Vanishing Gradients in RNNs | Repeated tanh/sigmoid through time shrinks gradients on long sequences |
| 6 | LSTM | Forget, input, cell-state, output gates; additive cell state preserves gradients |
| 7 | GRU | Fewer gates than LSTM; update and reset; often faster with similar quality |
| 8 | CNN Deployment and Evaluation | JumpStart image models; per-class metrics and confusion matrix |
| 9 | MLOps Three Pillars | Automation; versioning and governance; monitoring and feedback |
| 10 | SageMaker Pipelines as DAGs | Steps with dependencies; conditions; register-or-fail |

---

## 1. Model Registry

**What it is:**
A managed catalog for deployable models. A **Model Package Group** names a family of models (like a repository). Each **Model Package** is one immutable version: **S3 model artifact** (`model.tar.gz`), **inference image URI**, and **metadata** (training job name, metrics, framework, who approved).

**Approval workflow:**
Statuses such as **PendingManualApproval** to **Approved** gate production deployment. In class you approve after review; in enterprises this maps to change control.

**Why it matters:**
Regulated teams need **who approved what, when, and from which artifact**. Registry is the system of record between training and endpoints.

**How you use it (conceptual SDK flow):**
1. Train a model; note `estimator.model_data` (S3 URI of `model.tar.gz`).
2. Create a **Model** or register a **Model Package** in a **Model Package Group**.
3. Set approval status after validation.
4. Deploy from an **Approved** package for production endpoints.

**Where it shows up:**
Finance model risk management, healthcare ML governance, any org with audit requirements.

**Gotcha:**
Registry tracks the **container plus artifact** pair. If you change the inference container without re-registering, behavior can change even with the same `.pkl` file.

> **Interview Tip:** Say explicitly: "Model Package = artifact + container + metadata + approval state." That is the full object you audit.

---

## 2. Real-Time Deployment and Invocation

**What it is:**
SageMaker composes three resources: **Model** (artifact + image), **Endpoint Configuration** (instance type, instance count, optional production variants), **Endpoint** (DNS name, HTTPS). Calling `estimator.deploy(...)` on a training estimator builds all three.

**Why it matters:**
Endpoints bill **per instance-hour** while **InService**. Orphan endpoints are a top cause of surprise SageMaker bills.

**How you use it:**

```python
# After training (Script Mode sklearn example)
predictor = estimator.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.xlarge",
)

# Default serializer/deserializer often handle CSV/numpy; set explicitly if needed
from sagemaker.serializers import CSVSerializer
from sagemaker.deserializers import JSONDeserializer

predictor.serializer = CSVSerializer()
# predictor.predict(X_batch)  # rows must match training feature order
```

**Low-level invoke (same contract):**

```python
import boto3

sm = boto3.client("sagemaker-runtime")
response = sm.invoke_endpoint(
    EndpointName="your-endpoint",
    ContentType="text/csv",
    Body=payload_csv_bytes,  # no header if that is how the model was trained
)
```

**Where it shows up:**
Any online scoring: fraud, recommendations, real-time risk.

**Gotcha:**
**Training-serving skew** (Tuesday): if feature order, scaling, or definitions differ from training, accuracy drops with no stack trace.

> **Interview Tip:** Mention **delete order** when cleaning up: endpoint first, then endpoint config, then model -- mirrors dependency creation.

---

## 3. Evaluating at the Endpoint

**What it is:**
Treat the endpoint like a black-box classifier: send **validation** rows it never trained on, collect predictions, compute **accuracy**, **precision**, **recall**, **F1**, **confusion matrix**, and optionally **ROC-AUC** if you have probabilities.

**Why it matters:**
For **imbalanced fraud**, accuracy is misleading. You optimize the metric that matches **business cost** (missed fraud vs false alarms).

**How you use it:**

```python
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# y_true from validation labels; y_pred from endpoint outputs
print("Accuracy:", accuracy_score(y_true, y_pred))
print("Precision:", precision_score(y_true, y_pred))
print("Recall:", recall_score(y_true, y_pred))
print("F1:", f1_score(y_true, y_pred))
print(confusion_matrix(y_true, y_pred))
print(classification_report(y_true, y_pred))
```

**Interesting implementation note:**
If precision is **0.95** and recall is **0.60**, you catch only **60%** of fraud -- the rest is false negatives. Stakeholders may still accept that if false positives are very expensive, but you must quantify **dollar loss** from missed fraud.

> **Interview Tip:** Always connect metrics to **threshold**: production uses one operating point; AUC measures ranking across all thresholds.

---

## 4. Vanilla RNNs

**What it is:**
A recurrent network processes a sequence **one time step at a time** with **shared weights** across steps. At each step it updates a **hidden state** that summarizes what it has seen so far.

**Core recurrence (conceptually):**

```
h_t = tanh(W_xh @ x_t + W_hh @ h_{t-1} + b_h)
```

**Why it matters:**
RNNs are the baseline for **ordered data**: transactions, text characters, sensor traces. They are **not** the state of the art for long text (transformers won), but the **weight sharing** and **hidden state** ideas carry forward.

**Where it shows up:**
Time series classification, simple sequence tagging, and historically seq2seq before attention dominated.

**Gotcha:**
**Backpropagation through time** multiplies Jacobian terms across many steps. With **tanh/sigmoid**, gradients often **vanish** (or rarely **explode**), limiting long-range learning.

---

## 5. Vanishing Gradient Problem (in RNNs)

**What it is:**
During backprop through time, gradients from a late time step propagate to early steps through repeated matrix multiplications and derivative chains. If the recurrent Jacobian’s effective scale is **less than 1** at many steps, early layers get **near-zero** updates.

**Why it matters:**
The model cannot learn to depend on **distant tokens** -- "the subject at the start of the sentence" does not influence the loss signal at the end.

**Diagnostic intuition:**
Gradient magnitude at step 0 may be a tiny fraction of the magnitude at the last step.

**The fix (before Transformers):**
- **LSTM/GRU** (additive paths for memory)
- Careful **initialization**
- **Gradient clipping** for explosion
- **Transformers** sidestep recurrence entirely for many NLP tasks

> **Interview Tip:** Contrast with **Week 1** vanishing gradients in **deep sigmoid MLPs** -- same mathematical issue (many small derivatives multiplied), different architecture.

---

## 6. LSTM (Long Short-Term Memory)

**What it is:**
An RNN cell with **explicit memory** (**cell state** `c_t`) and **gates** that control read, write, and forget. The cell state update uses **addition**, creating a **gradient highway** similar in spirit to ResNet skip connections.

**Gates (standard formulation):**

```
f_t = sigmoid(W_f @ [h_{t-1}, x_t] + b_f)   # forget gate
i_t = sigmoid(W_i @ [h_{t-1}, x_t] + b_i)   # input gate
o_t = sigmoid(W_o @ [h_{t-1}, x_t] + b_o)   # output gate
c_tilde_t = tanh(W_c @ [h_{t-1}, x_t] + b_c)

c_t = f_t * c_{t-1} + i_t * c_tilde_t       # additive update to cell state
h_t = o_t * tanh(c_t)
```

**Why it matters:**
LSTMs were the default for **encoder-decoder** machine translation before Transformers. Week 1’s encoder-decoder lecture used LSTM blocks; Week 2 opens the **gates** and **cell state** intuition.

**Where it shows up:**
Legacy NLP, time series with long dependencies, edge devices when small RNNs suffice.

**Gotcha:**
LSTMs are **sequential** -- hard to parallelize across time compared to self-attention.

> **Interview Tip:** Emphasize **additive cell state** vs vanilla RNN's purely multiplicative path through `h`.

---

## 7. GRU (Gated Recurrent Unit)

**What it is:**
A simplified gated RNN: combines **forget** and **input** into an **update gate**, uses a **reset gate** to control past hidden influence. Fewer parameters than LSTM.

**Why it matters:**
Often **comparable accuracy** to LSTM with **faster training** and smaller models -- good default when you still need recurrence.

**LSTM vs GRU (decision sketch):**

| | LSTM | GRU |
|--|------|-----|
| Parameters | More (separate cell state) | Fewer |
| Capacity | Often slightly higher | Slightly less |
| When to try | Maximum sequence modeling capacity | Speed / smaller footprint |

---

## 8. CNN Deployment and Evaluation (JumpStart)

**What it is:**
Deploy an image model (for example from **JumpStart**), send **image bytes** (JPEG/PNG), receive class logits or labels. Evaluate with **per-class accuracy** and a **confusion matrix** on a held-out set (for example CIFAR-10).

**Why it matters:**
**Aggregate accuracy** hides weak classes. Confusion pairs (cat vs dog) tell you **which data to collect** or **how to augment**.

**How you read results:**
Inspect off-diagonal mass -- classes with frequent mislabels need more diverse training examples or label cleaning.

> **Interview Tip:** For vision, mention **data augmentation** only on training, never on validation/test (Week 1 CNN section).

---

## 9. MLOps Three Pillars

**What it is:**
A framing for production ML systems:

| Pillar | What it covers |
|--------|----------------|
| **Automation** | Pipelines, scheduled jobs, CI triggers, infrastructure as code |
| **Versioning and governance** | Data, code, features, model registry, approvals |
| **Monitoring and feedback** | Data and model quality, drift, alerts, retraining loops |

**Why it matters:**
**Traditional CI/CD** compares to **MLOps** like this:

| Traditional CI/CD | ML CI/CD |
|-------------------|----------|
| Commit triggers build | Data or code change triggers retrain |
| Unit tests gate merge | **Evaluation metrics** gate model promotion |
| Single binary artifact | **Model artifact** plus **data lineage** |
| App uptime monitoring | **Input distribution** and **prediction quality** monitoring |

**Where it shows up:**
Any team running models beyond a notebook.

---

## 10. SageMaker Pipelines as DAGs

**What it is:**
**SageMaker Pipelines** define a **directed acyclic graph (DAG)**. Each **Step** is a node: **ProcessingStep**, **TrainingStep**, **Evaluation step** (often a ProcessingStep that writes metrics), **ConditionStep**, **RegisterModel**.

**Why it matters:**
Manual "notebook order" does not scale. Pipelines give **repeatable**, **auditable** automation with **parameterized** runs.

**Conceptual FraudShield flow:**

```
Processing --> Training --> Evaluation --> Condition (F1 >= threshold?)
                                              |
                    Yes --> RegisterModel (PendingManualApproval)
                    No  --> Fail / notify
```

**Key SDK ideas:**
- **Pipeline parameters** (`ParameterString`, `ParameterInteger`) for environment portability.
- **PropertyFile** + **JsonGet** to read metric JSON from evaluation for **ConditionStep**.
- **RegisterModel** step targets a **Model Package Group** from the Model Registry.

**Gotcha:**
Downstream steps run **only if upstream succeeds** -- failures stop the DAG unless you add **catch** or branching (depending on SDK version and pattern).

> **Interview Tip:** Say "DAG inferred from step dependencies" -- edges are data lineage between steps, not arbitrary ordering.

---

# Tuesday -- Advanced SageMaker Data Engineering

| # | Topic | One-Liner |
|---|-------|-----------|
| 1 | Training-Serving Skew | Same features must be built the same way in train and serve |
| 2 | Feature Store Overview | Central feature definitions with online and offline access |
| 3 | Feature Group Schema | Record identifier, event time, typed features |
| 4 | Online Store | Low-latency get_record by entity id |
| 5 | Offline Store | Historical Parquet in S3; Athena SQL for training sets |
| 6 | Data Wrangler | Visual ETL; export to jobs, pipelines, Feature Store |
| 7 | Canvas | No-code tabular, time series, vision, text experiments |
| 8 | Autopilot | AutoML with objective metric, candidates, generated artifacts |
| 9 | Canvas vs Autopilot | Audience, control, and integration depth differ |
| 10 | Pipelines plus Feature Store | Offline query in ProcessingStep; train; register |

---

## 1. Training-Serving Skew

**What it is:**
**Training** used feature logic **A** (for example `transaction_count_24h` from a rolling window). **Serving** accidentally uses logic **B** (calendar day count). The model receives **different distributions** than it was fit on.

**Why it matters:**
Performance drops **silently** -- no Python exception -- until business KPIs move.

**How you fix it:**
- Single **feature definition** in **Feature Store**
- Same **ingestion** path for batch and online updates where possible
- Integration tests comparing **train vs serve** feature distributions

> **Interview Tip:** Skew is one of the top **production ML** failures; Feature Store is the AWS-native mitigation.

---

## 2. Feature Store (Architecture)

**What it is:**
A managed feature catalog. You define **Feature Groups** with a schema. Ingest writes rows; **online** store serves **latest** features by key; **offline** store keeps **history** for training queries.

**High-level diagram:**

```
Ingestion --> Feature Group --> Online Store (latest per record id)
                            --> Offline Store (S3 Parquet, full history)
                                        |
                                        v
                                  Athena / Spark joins
```

**Why it matters:**
**One source of truth** for features consumed by **batch training** and **real-time inference**.

---

## 3. Feature Group Schema

**What it is:**
Every group defines:
- **Record identifier** (for example `record_id` or `customer_id` + transaction id pattern)
- **Event time** (when the feature row is valid)
- **Feature** columns with types (Fractional, Integral, String, ...)

**Why it matters:**
The offline store partitions by time; the online store resolves **latest** as of lookups.

**Gotcha:**
Schema mismatches on ingest cause **ValidationException** -- names and types must match exactly.

---

## 4. Online Store

**What it is:**
A low-latency key-value style API: given **record identifier**, return the **current** feature vector for real-time models.

**Why it matters:**
Fraud scoring at payment time needs **milliseconds**, not a batch SQL job.

**How you use it (conceptual):**
Call **`get_record`** (Feature Store runtime client) with feature group name and record id; receive feature values your model expects.

---

## 5. Offline Store

**What it is:**
Append-only **historical** feature values stored as **Parquet** in **S3**, cataloged for **Athena** (or Spark) queries.

**Why it matters:**
Training needs **point-in-correct** history -- "what did we know at decision time?" -- not only **latest** snapshots.

**How you use it:**
SQL over the Glue/Athena table for a date range to materialize a **training CSV** or Parquet in a **ProcessingStep**.

**Gotcha:**
New partitions can lag slightly; **`MSCK REPAIR TABLE`** or partition refresh may be needed after backfills.

---

## 6. Data Wrangler

**What it is:**
A **visual** data prep tool in **SageMaker Studio**: import from S3, Athena, Redshift, Snowflake, Databricks, etc.; apply **300+** transforms; analyze **data quality** and leakage; **export** to:
- SageMaker **Processing** job
- **Pipeline**
- **Feature Store** ingestion
- **Notebook**

**Why it matters:**
Accelerates standard prep while emitting **the same SDK artifacts** you would hand-code.

> **Interview Tip:** Wrangler is an **accelerator**, not a separate runtime -- exports are normal SageMaker jobs.

---

## 7. SageMaker Canvas

**What it is:**
A **no-code** Studio app for **Quick Build** models: upload CSV or connect data, pick a target, train and analyze **without** writing Python.

**Why it matters:**
**Business analysts** can validate signal before engineering invests in pipelines.

**Limitations:**
Less control over algorithms, preprocessing, and custom containers than full SDK workflows.

---

## 8. Autopilot

**What it is:**
**Automated ML**: point at **S3** tabular data, specify **target**, **problem type**, and **objective metric** (for example **F1** for fraud). Autopilot explores **feature pipelines** and **algorithms**, trains **candidates**, and ranks them.

**Important knobs (conceptual):**
- **`AutoMLJobObjective`**: metric to optimize
- **`CompletionCriteria`**: **max candidates**, **max runtime** per job, total runtime
- **Problem type**: binary/multiclass classification, regression

**Why it matters:**
Strong **baseline** fast; surfaces **leaderboard** and sometimes **notebooks** for transparency.

**Gotcha:**
Each candidate is a **full training job** -- cost scales with **max candidates**.

---

## 9. Canvas vs Autopilot

| Dimension | Canvas | Autopilot |
|-----------|--------|-----------|
| **Audience** | Business analysts, citizen DS | ML engineers, data scientists |
| **Interface** | Point-and-click in Canvas | Studio UI or **Python SDK** |
| **Control** | Minimal | Moderate (objective, candidates, runtime) |
| **Artifacts** | In-app analysis, deploy from UI | Models in S3, logs, candidate notebooks |
| **Best for** | Quick validation, self-service | Production-grade AutoML baselines |

> **Interview Tip:** Canvas often uses **Autopilot** under the hood for model training -- relationship between products.

---

## 10. Pipelines plus Feature Store

**What it is:**
Wire **offline Feature Store** into **ProcessingStep**: Athena (or Spark) query for a training window writes **train/validation** to **S3**. **TrainingStep** consumes that output. **RegisterModel** lands a new version in **Model Registry** with **PendingManualApproval**.

**Why it matters:**
End-to-end reproducibility: **features**, **training**, and **registration** are one automated path.

**Gotcha:**
IAM permissions must allow **Athena**, **Glue**, **S3**, **SageMaker** execution on the same role.

---

# Wednesday -- Transformers, Pre-trained Models, Experiments, and Lineage

| # | Topic | One-Liner |
|---|-------|-----------|
| 1 | Encoder-Decoder Bottleneck | One context vector loses long-sequence detail |
| 2 | Attention (seq2seq) | Decoder attends to all encoder states with weights |
| 3 | Scaled Dot-Product Attention | Softmax(Q K^T / sqrt(d_k)) V |
| 4 | Multi-Head Attention | Parallel heads; concatenate; linear projection |
| 5 | Positional Encoding | Inject order; sinusoidal or learned embeddings |
| 6 | Transformer Encoder Block | Self-attention, FFN, residuals, layer norm |
| 7 | BERT vs GPT vs T5 | Encoder MLM; decoder causal LM; text-to-text |
| 8 | Transfer Learning for Text | Pre-train on large corpus; fine-tune on task |
| 9 | Hugging Face Pipelines | High-level API for classify, generate, summarize |
| 10 | SageMaker Experiments | Experiments, runs, logged parameters and metrics |
| 11 | Compare Runs | Sort and filter trials by metric |
| 12 | Lineage Entities | Artifacts, actions, contexts, associations |
| 13 | Feature Store Lineage | Feature groups linked to training jobs |
| 14 | Reproducibility Patterns | Seeds, data versions, images, commits |

---

## 1. Encoder-Decoder Bottleneck

**What it is:**
A classic seq2seq model encodes the whole input into **one vector**, then decodes. Long inputs must compress into **fixed size** -- information loss.

**Why it matters:**
Motivates **attention**: the decoder can **peek** at all encoder positions instead of one summary vector.

---

## 2. Attention (in seq2seq)

**What it is:**
At each decoder step, compute a **weighted sum** of **encoder hidden states**. Weights come from **compatibility** between decoder state and each encoder position.

**Why it matters:**
Direct **long-range** dependencies from decoder to relevant source tokens -- the foundation of **Transformers**, which generalize attention.

---

## 3. Scaled Dot-Product Attention

**What it is:**
Given **Q**, **K**, **V** matrices (queries, keys, values):

```
Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
```

**Why scale by sqrt(d_k):**
For large key dimension, dot products grow in magnitude; **softmax** saturates and gradients shrink (**Week 1 softmax saturation** theme).

**How you use it (NumPy sketch):**

```python
import numpy as np

def scaled_dot_product_attention(Q, K, V):
    d_k = Q.shape[-1]
    scores = (Q @ K.T) / np.sqrt(d_k)
    weights = softmax(scores, axis=-1)  # row-wise over keys
    return weights @ V, weights

def softmax(x, axis=-1):
    exp = np.exp(x - x.max(axis=axis, keepdims=True))
    return exp / exp.sum(axis=axis, keepdims=True)
```

> **Interview Tip:** One row of **weights** shows "what this query looked at" -- interpretable as an attention map.

---

## 4. Multi-Head Attention

**What it is:**
Split **d_model** into **h** heads: each head runs **scaled dot-product attention** on projected **Q_i, K_i, V_i**. Concatenate outputs and apply **W_O**.

**Why it matters:**
Different heads can specialize (**syntax vs semantics**, **local vs global** patterns).

**Parameter note:**
Total parameters are similar to one big projection -- dimensions are **split**, not duplicated for free.

---

## 5. Positional Encoding

**What it is:**
Self-attention is **permutation invariant** without position info. **Positional encodings** are added to token embeddings.

**Sinusoidal (original Transformer):**

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

**Why it matters:**
Unique position signatures; relative position relationships emerge in the embedding space.

**Alternative:** **Learned** positional embeddings (common in BERT-like models with fixed max length).

---

## 6. Transformer Encoder Block

**What it is:**
One layer typically contains:
1. **Multi-head self-attention** (sub-layer)
2. **Residual** + **LayerNorm**
3. **Position-wise feed-forward network** (two linear layers with activation)
4. **Residual** + **LayerNorm**

**Why it matters:**
**Residuals** help optimization; **layer norm** stabilizes activations across features.

**Complexity note:**
Self-attention is **O(n^2)** in sequence length **n** -- long documents need **chunking**, **sparse attention**, or **linear attention** variants (outside Week 2 scope).

---

## 7. BERT vs GPT vs T5

| Model | Stack | Pre-training objective | Typical fine-tune |
|-------|-------|------------------------|-------------------|
| **BERT** | Encoder only | **Masked LM** (+ next sentence) | Classification head on **[CLS]** |
| **GPT** | Decoder only | **Causal** next-token prediction | Prompting or fine-tune for task |
| **T5** | Encoder-decoder | **Text-to-text** span corruption | Prefix tasks as "translate English to French: ..." |

**Why it matters:**
Architecture choice matches **task**: understanding (encoder) vs **generation** (decoder) vs **format conversion** (seq2seq).

---

## 8. Transfer Learning for Text

**What it is:**
Same pattern as **Week 1** image fine-tuning: start from weights trained on **large generic text**, add a **task head**, fine-tune on **labeled in-domain data**.

| Image (Week 1) | Text (Week 2) |
|----------------|---------------|
| MobileNet on ImageNet | BERT on Wikipedia + BookCorpus |
| Freeze conv blocks | Freeze lower transformer layers (optional) |
| Train classifier head | Train classification head / soft prompts |

---

## 9. Hugging Face Transformers (Pipelines)

**What it is:**
The **transformers** library provides **pre-trained weights** and **tokenizers**. **`pipeline`** wraps end-to-end inference.

**How you use it:**

```python
from transformers import pipeline

clf = pipeline("sentiment-analysis")
clf("This movie was surprisingly good.")

gen = pipeline("text-generation", model="gpt2")
gen("The future of MLOps is", max_new_tokens=40, do_sample=True, temperature=0.8)

summ = pipeline("summarization")
summ(long_article_text, max_length=60, min_length=20, do_sample=False)
```

**Why it matters:**
Fast **baseline**; for production, pin **model versions**, handle **batching**, **GPU**, and **latency SLOs**.

---

## 10. SageMaker Experiments

**What it is:**
Organize work into **Experiment** (project) and **Run** (one training attempt). Log **parameters**, **metrics**, and **artifacts**.

**Conceptual logging:**

```python
# Pseudocode pattern -- actual imports vary by SDK version
# from sagemaker.experiments import Run

# with Run(experiment_name="fraud-study", run_name="rf-100trees") as run:
#     run.log_parameter("n_estimators", 100)
#     run.log_metric("validation_f1", 0.87)
#     run.log_file("confusion_matrix.png")
```

**Why it matters:**
Replaces **spreadsheet chaos** when you have dozens of trials.

---

## 11. Comparing Experiment Runs

**What it is:**
List runs under an experiment; filter and sort by **metric** (for example highest **validation F1**); compare **parameters** side by side.

**Why it matters:**
Scientific iteration: you need **which change caused uplift**.

---

## 12. Lineage Tracking

**What it is:**
A **graph** connecting **datasets**, **processing jobs**, **training jobs**, **models**, **endpoints**.

**Core entities:**
- **Artifact** -- S3 objects, Feature Groups, models
- **Action** -- Training, processing
- **Context** -- Higher-level grouping (experiment, pipeline)
- **Association** -- edges (produced, consumed, contributed-to)

**Why it matters:**
**Root-cause** analysis when production predictions degrade: trace back to **data snapshot** and **code**.

---

## 13. Feature Store plus Lineage

**What it is:**
**Feature groups** register as **artifacts**. Training jobs that **read** offline or online features create **associations**. You can ask: "Which models depend on this feature definition?"

**Why it matters:**
**Impact analysis** before changing a feature transform.

---

## 14. Reproducibility Patterns

| Pattern | What to record | Where |
|---------|----------------|--------|
| Data | S3 URI + version / FS time range | Run params, lineage |
| Code | Git commit hash | Run params |
| Environment | Container image URI, framework version | Training job |
| Randomness | Seeds | Hyperparameters |
| Metrics | All evaluation numbers | Run metrics |

**Gotcha:**
**Non-determinism** on GPU (especially with multiple streams) can still cause small drift even with seeds.

> **Interview Tip:** Minimum viable reproducibility: **seed + data hash + image + commit + hyperparams**.

---

# Thursday -- Built-in Algorithms and Hyperparameter Optimization

| # | Topic | One-Liner |
|---|-------|-----------|
| 1 | Built-in XGBoost | Managed container; no custom train.py |
| 2 | Tabular Data Contract | Target first column; no header |
| 3 | Boosting vs Bagging | Sequential correction vs parallel bagging |
| 4 | K-Means | Unsupervised clusters; choose k |
| 5 | Random Cut Forest | Anomaly scores without labels |
| 6 | BlazingText | Word2Vec and text classification at scale |
| 7 | DeepAR | Probabilistic forecasting with RNNs |
| 8 | HPO Anatomy | Objective, ranges, strategy, budget |
| 9 | HyperparameterTuner | Wrap estimator; search; best job |

---

## 1--2. Built-in XGBoost and Data Format

**What it is:**
SageMaker provides a **prebuilt XGBoost container**. You configure **hyperparameters**; you do **not** supply a training script for the built-in path.

**CSV contract (typical):**
- **Label in the first column**
- **No header row**
- **No index column**

```python
# Pandas: fraud label column must be moved to position 0 before export
df = df[["target", "feat1", "feat2", ...]]
df.to_csv("train.csv", header=False, index=False)
```

**Why it matters:**
Violating the contract yields **AlgorithmError** or nonsense models -- always verify with a **head** of the uploaded file.

---

## 3. Boosting vs Bagging (Random Forest)

**What it is:**
- **Bagging (Random Forest):** train many trees on **bootstrap** samples; **average** or vote -- **reduces variance**.
- **Boosting (XGBoost):** trees are **sequential**; each fits **residuals** / gradients of the ensemble -- **reduces bias**, focuses on **hard examples**.

**Why XGBoost often wins on tabular:**
Strong **nonlinear** interactions and **ordered splits** with regularization.

> **Interview Tip:** "Random Forest parallelizes independent trees; XGBoost chains trees to fix remaining error."

---

## 4. K-Means (Built-in)

**What it is:**
Partition **n** points into **k** clusters by minimizing within-cluster distance to centroids.

**Use on fraud features (unsupervised):**
Discover **segments** (for example high-amount international) then **analyze fraud rate per cluster** post hoc.

**Gotcha:**
**Scale features** when magnitudes differ wildly; distance metrics are not scale-invariant.

---

## 5. Random Cut Forest (RCF)

**What it is:**
An **ensemble of random trees** over random cuts; **anomaly score** reflects how easily a point is **isolated** from the bulk.

**Why it matters:**
**No labels** required -- useful as a **first-line** alert layer with supervised models.

---

## 6--7. BlazingText and DeepAR (Survey)

**BlazingText:** **Word2Vec** (cbow/skip-gram) and **text classification** optimized for throughput. Older than Transformers but still relevant for **embedding** pipelines at scale.

**DeepAR:** **Probabilistic** forecasts for **many related time series**; uses **RNN**-style recurrence internally; outputs **quantiles**.

---

## 8. HPO Job Anatomy

**Components:**
- **Objective metric name** -- must match a **logged** metric from training (for example `validation:auc`)
- **Objective type** -- **Maximize** or **Minimize**
- **Parameter ranges** -- continuous, integer, categorical
- **Search strategy** -- Bayesian or Random
- **Resource limits** -- `max_jobs`, `max_parallel_jobs`, timeouts

**Why Bayesian search:**
Uses results from prior trials to pick the **next** promising configuration -- often **sample-efficient** vs random.

**Why random search:**
Embarrassingly **parallel**; no sequential dependency between suggestions.

---

## 9. HyperparameterTuner (Sketch)

**What it is:**
Wrap your **Estimator** (here, built-in XGBoost) in a **`HyperparameterTuner`**, define **ranges**, call **`.fit()`** on the tuner.

```python
from sagemaker.tuner import HyperparameterTuner, IntegerParameter, ContinuousParameter

hyperparameter_ranges = {
    "max_depth": IntegerParameter(3, 10),
    "eta": ContinuousParameter(0.01, 0.3),
    "num_round": IntegerParameter(50, 300),
}

tuner = HyperparameterTuner(
    estimator=xgb_estimator,
    objective_metric_name="validation:auc",
    objective_type="Maximize",
    hyperparameter_ranges=hyperparameter_ranges,
    max_jobs=20,
    max_parallel_jobs=3,
)

tuner.fit({"train": train_s3, "validation": val_s3})
```

**After tuning:**
`best_training_job()` yields the **winning** hyperparameters and model artifact for **deployment** or **registry**.

**Gotcha:**
**Objective metric** string must **exactly** match what the algorithm **logs** to CloudWatch.

> **Interview Tip:** Map HPO trials to **Experiment runs** -- both are structured iteration; HPO **automates** the search policy.

---

# Friday -- Inference Patterns and Model Monitoring

| # | Topic | One-Liner |
|---|-------|-----------|
| 1 | Inference Decision Matrix | Choose pattern from latency, payload, traffic, cost |
| 2 | Real-Time Endpoint | ms latency; 6 MB payload; always-on instances |
| 3 | Serverless Inference | Scale to zero; cold start; CPU only; 4 MB payload |
| 4 | Asynchronous Inference | Queue; large payloads; results in S3 |
| 5 | Batch Transform | Ephemeral job; S3 input/output |
| 6 | Multi-Model Endpoint | Shared instance; dynamic model load |
| 7 | Multi-Container Endpoint | Preprocess + predict chains |
| 8 | Data Capture | Sampled request/response logs to S3 |
| 9 | Baselines | statistics.json and constraints.json |
| 10 | Data vs Model Quality Monitoring | Inputs vs predictions vs labels |
| 11 | Drift Types | Data, concept, prediction |
| 12 | Statistical Tests | KS, chi-squared, L-infinity |
| 13 | Bias and Attribution Drift | Clarify monitoring extensions |
| 14 | EventBridge Automation | Alarm to pipeline/Lambda |

---

## 1. Inference Decision Matrix

**Full comparison (typical SageMaker docs):**

| Dimension | Real-Time | Serverless | Async | Batch Transform | Multi-Model |
|-----------|-----------|------------|-------|-----------------|-------------|
| **Latency** | Low ms | Variable (cold start) | Minutes queue | Job runtime | Low ms after load |
| **Payload** | 6 MB | 4 MB | Up to ~1 GB class | Unlimited via S3 | 6 MB per call |
| **Traffic** | Steady | Bursty / sporadic | Large slow requests | Offline bulk | Many models, moderate each |
| **Scale to zero** | No | Yes | No | N/A | No |
| **GPU** | Yes | No | Yes | Yes | Yes |

**Decision flow (text):**
- **One-time bulk score** on S3 data --> **Batch Transform**
- **Huge payload** or **long inference** --> **Async**
- **Bursty** traffic, **cost-sensitive**, OK with **cold start** --> **Serverless**
- **Many similar models** on one box --> **Multi-Model Endpoint**
- Else default **Real-Time**

---

## 2. Real-Time Endpoint

**What it is:**
Always-on **instances** behind an Application Load Balancer-style HTTPS endpoint. Lowest **steady-state** latency when provisioned correctly.

**When to pick:**
Stable **QPS**, strict **p99 latency**, **GPU** models.

---

## 3. Serverless Inference

**What it is:**
Configure **`ServerlessInferenceConfig`** with **memory** (1024--6144 MB steps) and **max concurrency**. SageMaker scales **in** and **out**; can **scale to zero**.

```python
from sagemaker.serverless import ServerlessInferenceConfig

serverless_config = ServerlessInferenceConfig(
    memory_size_in_mb=2048,
    max_concurrency=5,
)
# predictor = model.deploy(..., serverless_inference_config=serverless_config)
```

**Gotcha:**
**Cold start** (often **1--2 s** first request after idle). **No GPU**. **4 MB** payload cap.

---

## 4. Asynchronous Inference

**What it is:**
Client POSTs payload; receives acknowledgment; worker processes **asynchronously**; results land in **S3**; optional **SNS** notification.

**When to pick:**
Oversized payloads for sync APIs, **minutes-long** inference, **decoupled** clients.

---

## 5. Batch Transform

**What it is:**
A **batch job**, not a long-lived endpoint. Reads **S3** input, writes **S3** output, tears down compute.

**Knobs:**
- **`SplitType`** / **`BatchStrategy`** -- **MultiRecord** improves throughput
- **`JoinSource`** -- append predictions to input rows for analysis

**When to pick:**
Nightly scoring, **backfills**, **monthly** risk runs.

---

## 6. Multi-Model Endpoint (MME)

**What it is:**
One endpoint hosts **many models** behind one **shared** container type. Models load **on demand** from S3; LRU **eviction** when memory pressure.

**Invoke:** specify **`TargetModel`** to choose which model artifact.

**Trade-off:**
**Loading** latency when a cold model is requested; **cost** savings vs **N** separate endpoints.

---

## 7. Multi-Container Endpoint

**What it is:**
Different **containers** in a **pipeline** on one endpoint -- for example **preprocessing** container then **inference** container.

**Why it matters:**
Encapsulates **full request path** without client-side preprocessing.

---

## 8. Data Capture

**What it is:**
**`DataCaptureConfig`** on deploy logs a **sample** of requests and responses to **S3** as **JSON Lines**.

```python
from sagemaker.model_monitor import DataCaptureConfig

data_capture_config = DataCaptureConfig(
    enable_capture=True,
    sampling_percentage=100,  # lower in prod (10--20% typical)
    destination_s3_uri="s3://bucket/capture/",
)
```

**Why it matters:**
**Fuel** for monitoring jobs and **debugging** production errors.

---

## 9. Baselines (Model Monitor)

**What it is:**
Run a **baselining** job on **training data** to produce:
- **`statistics.json`** -- per-feature means, std, min, max, quantiles
- **`constraints.json`** -- thresholds for violations

**Sketch:**

```python
from sagemaker.model_monitor import DefaultModelMonitor
from sagemaker.model_monitor.dataset_format import DatasetFormat

monitor = DefaultModelMonitor(role=role, instance_count=1, instance_type="ml.m5.xlarge")

monitor.suggest_baseline(
    baseline_dataset="s3://bucket/train.csv",
    dataset_format=DatasetFormat.csv(header=True),
    output_s3_uri="s3://bucket/baseline/",
)
```

---

## 10. Data Quality vs Model Quality Monitoring

| Dimension | Data Quality | Model Quality |
|-----------|--------------|---------------|
| Monitors | **Input** feature distributions | **Predictions** vs **labels** |
| Ground truth | Not required | **Required** (late arriving for fraud) |
| Detects | **Covariate shift**, schema breaks | **Accuracy** decay |

**Why both:**
**Data drift** often appears **before** label feedback exists -- early warning.

---

## 11. Drift Types

| Type | What shifts | Example |
|------|-------------|---------|
| **Data drift** | P(X) | Transaction amounts rise with inflation |
| **Concept drift** | P(y given X) | Fraud pattern changes after a product launch |
| **Prediction drift** | P(y_hat) | Model scores drift even before labels |

---

## 12. Statistical Tests (Monitoring)

| Test | Typical use | Idea |
|------|-------------|------|
| **Kolmogorov-Smirnov** | Continuous features | Max separation between **CDFs** |
| **Chi-squared** | Categorical | Frequency mismatch |
| **L-infinity** | Binned features | Max bin probability difference |

**Interpretation:**
Violations list **which features** crossed **constraints** -- prioritize **high-importance** features first.

---

## 13. Bias and Attribution Drift

**Bias drift:** Performance or calibration differs across **demographic** or **geographic** segments over time -- **Clarify** can monitor **fairness metrics**.

**Attribution drift:** **SHAP**-like importance shifts -- model **relies** on different features than at training.

**Why it matters:**
Regulatory and **risk** teams care about **fairness** and **stability** of explanations.

---

## 14. EventBridge Automation

**What it is:**
**CloudWatch** alarms on monitoring metrics --> **EventBridge** rule --> **Lambda**, **Step Functions**, or **Pipeline** execution.

**Pattern:**

```
Monitoring schedule --> violations --> CW metric --> alarm --> EventBridge --> Retrain pipeline
```

**Governance:**
High-stakes systems often **auto-retrain** but **human-gate** deployment via **Model Registry** approval.

---

# Advanced MLOps on SageMaker (Evolv skill -- production and cost)

These topics come from the **Advanced MLOps and Production Engineering on SageMaker** curriculum and complement the AFD week 2 arc.

---

## Managed Spot Training

**What it is:**
Run **training jobs** on **Spot** instances with **interrupt** risk at much lower price.

**How you survive interrupts:**
- **`checkpoint_s3_uri`** for framework checkpoints
- Logic in training script to **resume** from latest checkpoint

**Why it matters:**
Large **GPU** training cost reduction when checkpointing is feasible.

---

## Spot for HPO

**What it is:**
Each **HPO trial** can use Spot. Same **checkpoint** discipline applies per trial.

**Gotcha:**
Some trials may **restart** -- budget extra wall-clock time.

---

## Instance Right-Sizing

**What it is:**
Pick **CPU/GPU/memory** to match workload -- avoid **p3** GPUs for tiny batch CPU-bound prep.

**Tools:**
**CloudWatch** GPU utilization dashboards; **Inference Recommender** for endpoints.

---

## Auto-Scaling for Endpoints

**What it is:**
Target-tracking on **InvocationsPerInstance** or **CPU/GPU** utilization; **min/max** capacity.

**Why it matters:**
Handle **peaks** without permanently paying for **peak** capacity.

---

## Inference Recommender

**What it is:**
Load-test **candidate** instance types and configs against your **model artifact**; compare **latency** and **cost**.

**Why it matters:**
Replaces guessing **`ml.m5.xlarge`** vs **`ml.c5.2xlarge`** with **data**.

---

## VPC, PrivateLink, KMS

**VPC:** Run **endpoints** and **training** in private subnets; control **egress**.

**PrivateLink:** Private connectivity to **SageMaker API** without public internet.

**KMS:** **Encrypt** **EBS**, **S3**, **artifacts** with **customer-managed keys** for policy control.

---

## Cross-Account Sharing

**What it is:**
Share **Feature Store** or **artifacts** via **RAM** (Resource Access Manager) and **cross-account IAM** roles for **centralized** governance in **landing zone** setups.

---

## CloudWatch Dashboards

**What it is:**
Unified charts for **training** job metrics, **endpoint** invocations, **latency**, **errors**, **GPU** utilization.

**Why it matters:**
**Operations** visibility ties monitoring back to **SRE** practice.

---

# Supplemental Deep Dives (SDK Patterns and Interview Sound Bites)

The sections below condense **repeatable SDK shapes** and **comparison tables** in the same spirit as Week 1’s longer algorithm sections. They are not a substitute for AWS docs version pins -- always check **region**, **SDK version**, and **IAM**.

---

## A. Model Registry and Deployment (High-Level SDK)

**Register** a model package after you have **`model_data`** (S3 `model.tar.gz`) and the **inference image URI** (framework or algorithm image).

**Conceptual steps:**
1. **`sagemaker.model.Model`** -- wrap `model_data`, `role`, `image_uri`, optionally `env` for inference.
2. **`model.create_model_package(...)`** or use **Model Registry** APIs to create a **ModelPackage** inside a **ModelPackageGroup**.
3. Set **ModelApprovalStatus** to **`PendingManualApproval`** until validation completes.
4. **`approved_model.deploy(...)`** or create **EndpointConfig** + **Endpoint** from the approved package.

**Why interviewers care:**
They want to hear **governance**: **who** can move from **Pending** to **Approved**, and how that ties to **change control**.

**Gotcha:**
The **inference image** must match how the artifact was **produced** (same framework major version as training container where applicable).

---

## B. Feature Store: Create, Ingest, Read (Conceptual)

**Create Feature Group** with:
- **`FeatureDefinitions`** (name + type)
- **`RecordIdentifierFeatureName`**
- **`EventTimeFeatureName`**
- **`OnlineStoreConfig`** and **`OfflineStoreConfig`** (S3 for offline)

**Ingest:**
- **`ingest(data_frame=df, max_workers=...)`** for bulk
- **`put_record`** for streaming / single row

**Read online:**

```python
# Conceptual -- API names vary slightly by SDK version
# featurestore_runtime.get_record(
#     FeatureGroupName="fraudshield-transaction-features",
#     RecordIdentifierValueAsString=record_id,
# )
```

**Read offline:**
- **Athena** SQL over the **Glue** table SageMaker creates
- Filter by **event time** window for training

> **Interview Tip:** State clearly: **online = latest**, **offline = history** -- different query patterns, **same definitions**.

---

## C. SageMaker Pipelines: Step Types and Conditionals

**Common step classes** (imports evolve with SDK):

| Step | Purpose |
|------|---------|
| **ProcessingStep** | Run a processing container; data prep; Feature Store export |
| **TrainingStep** | Run `estimator` with input channels |
| **TransformStep** | Batch transform (less common inside training DAGs) |
| **RegisterModel** | Create **ModelPackage** in registry |
| **ConditionStep** | Branch on **Boolean** from **`JsonGet`** on evaluation `PropertyFile` |
| **FailStep** | Explicit failure with message |

**Parameters** make a pipeline reusable across **dev/stage/prod** (different buckets, roles, instance types).

**Gotcha:**
**CacheConfig** (when enabled) skips recomputation if inputs match -- great for cost, confusing when you **intended** to rerun.

---

## D. Transformers vs Recurrent Models (Interview Matrix)

| Topic | LSTM / GRU | Transformer |
|-------|------------|---------------|
| **Parallelism** | Sequential along time | Parallel across tokens (training) |
| **Long range** | Limited by BPTT vanishing | Global attention; cost O(n^2) |
| **Inductive bias** | Order via recurrence | Order via positional encoding |
| **Typical NLP today** | Legacy / small data | Default for accuracy at scale |
| **Time series** | Still common (DeepAR) | Emerging with patches / hybrids |

---

## E. Built-in Algorithms: `image_uris.retrieve` Pattern

```python
from sagemaker import image_uris

xgb_image = image_uris.retrieve(
    framework="xgboost",
    region=region,
    version="1.5-1",  # pin explicitly in production
)

kmeans_image = image_uris.retrieve(
    framework="kmeans",
    region=region,
)

rcf_image = image_uris.retrieve(
    framework="randomcutforest",
    region=region,
)
```

Then pass **`image_uri`** into **`Estimator`** with **`hyperparameters`** set for that algorithm’s contract.

> **Interview Tip:** Built-in means **no `entry_point`** -- different from **Script Mode**.

---

## F. K-Means Training Sketch (Unsupervised)

**Inputs:** **feature columns only** -- **no label column** in the training file for pure clustering.

**Hyperparameters (typical):**
- **`k`** -- number of clusters
- **`feature_dim`** -- must match input dimension

**After training:**
**Deploy** (some workflows) to get **cluster assignments** for each row, or obtain **cluster centers** from artifacts depending on container version.

**Evaluation:**
Clustering metrics (**silhouette**, **inertia**) or **business** validation: does a cluster have **elevated fraud rate**?

---

## G. Random Cut Forest Training Sketch

**Hyperparameters (examples from lectures):**
- **`num_trees`**
- **`num_samples_per_tree`**

**Output at inference:**
An **anomaly score** per record -- higher means **more isolated** vs the training distribution.

**Production pattern:**
Use RCF as **routing** -- send top percentile to manual review **even if** supervised model is uncertain.

---

## H. Hyperparameter Tuning: Early Stopping and Warm Pools

**Early stopping (where supported):**
Stops **underperforming** trials relative to the best-so-far to save budget -- check algorithm + tuner compatibility for your SDK version.

**Warm start:**
Some setups allow **warm start** from prior HPO jobs -- useful when you **narrow** ranges after a coarse search.

**Cost control:**
- Reduce **`max_parallel_jobs`** if account **concurrent training** limits hit
- Shorten **`MaxRuntimeInSeconds`** per training job during exploration

---

## I. Serverless vs Real-Time: When Cold Start Kills the Deal

| Requirement | Favor |
|-------------|-------|
| **Sub-100 ms p99** | Real-time on right-sized instances |
| **Infrequent traffic**, cost priority | Serverless |
| **GPU inference** | Real-time or async (not serverless) |
| **> 4 MB payload** | Async or batch |

**Story for interviews:**
"I would load test with **Inference Recommender** or a scripted **vegeta**/Locust harness, then pick **auto-scaling** real-time if **p99** cannot tolerate **cold start**."

---

## J. Asynchronous Inference: Client Contract

**Caller steps (conceptual):**
1. **POST** payload to async endpoint
2. Receive **HTTP 202** with inference id / output location hint
3. **Poll** S3 output prefix **or** subscribe to **SNS** **Completed** topic
4. Parse output object(s)

**Why not sync:**
**6 MB** / **15 minute** class limits differ by service docs -- async exists for **fire-and-forget** workloads.

---

## K. Batch Transform: `join_source` and `split_type`

**`join_source="Input"`** appends **input columns** to **output** -- useful to keep **transaction id** next to **score**.

**`split_type="Line"`** for CSV line-by-line; **`Record`**-oriented options depend on format.

**Gotcha:**
Output **layout** must match downstream **Athena**/Spark expectations -- agree on **delimiter** and **compression**.

---

## L. Model Monitor: Scheduling and Violations

After **baseline** artifacts exist and **capture** writes data:

1. Create **`DefaultModelMonitor`** (or quality-specific monitor classes)
2. **`create_monitoring_schedule`** with:
   - **`endpoint_input`** pointing at endpoint + capture location
   - **`output_s3_uri`** for **reports**
   - **`statistics`** + **`constraints`** from baseline
   - **Cron** or **rate** expression
3. On schedule, SageMaker runs a **processing** job that emits **constraint_violations.json**

**Operational note:**
Start with **hourly** in learning environments; **daily** or **weekly** is common in production unless risk is high.

---

## M. Clarify: Bias and Explainability (Extension)

**Training-time / batch:**
**Clarify** can compute **SHAP**, **partial dependence**, **bias metrics** against **facet** columns.

**Monitoring extension:**
Compare **live** explanations / bias metrics to **baseline** -- **attribution drift** and **bias drift** from Friday’s lecture.

**Why not default:**
Extra **compute** and **privacy** reviews -- adopt when **regulators** or **internal risk** require it.

---

## N. Cross-Account Feature Store (Evolv)

**Pattern:**
Central **governance** account owns the **Feature Group**; **spoke** accounts **assume roles** or use **RAM**-shared resources to **read** offline for training and **write** online ingests with guardrails.

**Interview line:**
"We use **least privilege** **cross-account roles** so data science accounts **consume** features without **S3-wide** credentials."

---

## O. Spot Training: Checkpoint Contract (Evolv)

**Training script obligations:**
- Periodically **`save`** checkpoint to **`SM_MODEL_DIR`** or **`checkpoint_local_path`** per framework conventions
- On start, **`load`** latest checkpoint if present

**SageMaker config:**
- **`train_use_spot_instances=True`**
- **`checkpoint_s3_uri="s3://.../checkpoints/"`**
- **`max_wait`** / **`max_run`** must exceed longest expected **uninterrupted** training window you need

**Gotcha:**
If you **never checkpoint**, a Spot interrupt loses **all progress** since last save.

---

# Quick-Reference Formula Card

| Formula / Concept | Expression / Summary | Use |
|-------------------|---------------------|-----|
| **Scaled dot-product attention** | `Attention(Q,K,V) = softmax(Q K^T / sqrt(d_k)) V` | Transformer core |
| **Attention weights row** | Row of `softmax(Q K^T / sqrt(d_k))` | Interpret who attended to whom |
| **Multi-head merge** | `Concat(head_1..head_h) W_O` | Mix specialized heads |
| **Positional (sinusoidal)** | `PE(pos,2i)=sin(pos/10000^(2i/d)), PE(pos,2i+1)=cos(...)` | Order without recurrence |
| **Vanilla RNN step** | `h_t = tanh(W_xh x_t + W_hh h_{t-1} + b)` | Sequence baseline |
| **LSTM cell state** | `c_t = f_t * c_{t-1} + i_t * tanh(W_c [h_{t-1},x_t])` | Additive memory path |
| **LSTM hidden** | `h_t = o_t * tanh(c_t)` | Output gate modulates readout |
| **Precision** | `TP / (TP + FP)` | Cost of false positives |
| **Recall** | `TP / (TP + FN)` | Cost of false negatives |
| **F1** | `2PR / (P + R)` | Harmonic mean; imbalance |
| **MCC** (optional) | `(TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))` | Balanced metric when both classes matter |
| **AUC-ROC** | P(random positive > random negative) | Threshold-free ranking quality |
| **XGBoost built-in CSV** | Column 0 = label; `header=False`; `index=False` | Container contract |
| **Boosting residual idea** | Next tree fits negative gradient of loss | Why boosting chases hard rows |
| **K-Means objective** | Minimize within-cluster sum of squares | Unsupervised segmentation |
| **RCF intuition** | Higher score = easier to isolate in random cuts | Unsupervised anomaly |
| **KS statistic** | `sup_x \|F1(x) - F2(x)\|` | Continuous drift |
| **Chi-squared** | `sum ((O-E)^2 / E)` | Categorical drift |
| **L-infinity (bins)** | Max absolute bin frequency difference | Binned continuous/categorical |
| **Self-attention complexity** | O(n^2) in sequence length n | Long-sequence cost |
| **Serverless payload cap** | 4 MB | vs 6 MB synchronous real-time |
| **Real-time payload cap** | 6 MB | Sync inference limit |
| **Batch transform** | Job-based; S3 in/out | Bulk scoring without endpoint |

---

## Week 2 Key Libraries and Services (Quick Reference)

| Tool / Service | Role | Why in Week 2 |
|----------------|------|----------------|
| **SageMaker Python SDK** (`sagemaker`) | Estimators, deploy, tuner, pipelines | Core automation surface |
| **boto3** (`sagemaker-runtime`, `sagemaker`) | `invoke_endpoint`, low-level API | Production clients, automation |
| **SageMaker Feature Store** | Online/offline feature serving | Training-serving consistency |
| **Athena + Glue** | SQL over offline store | Training dataset materialization |
| **SageMaker Experiments** | Runs, metrics, compare | Replace ad-hoc experiment logs |
| **SageMaker Lineage** | Graph of artifacts and actions | Audit and root cause |
| **SageMaker Model Monitor** | Baseline + schedules + violations | Drift detection |
| **CloudWatch + EventBridge** | Metrics, alarms, routing | Automation hooks |
| **Hugging Face `transformers`** | Pre-trained models, tokenizers | NLP transfer learning |
| **XGBoost (built-in container)** | Gradient boosted trees on SageMaker | Strong tabular baseline |
| **Data Wrangler / Canvas / Autopilot** | Prep, no-code, AutoML | Different personas and speeds |

---

## Cross-Week Connections

- **Week 1** **Random Forest** (Script Mode) and **metrics** are baselines against **XGBoost**, **Autopilot**, and **HPO** in Week 2.
- **Model Registry** (Monday) is the **promotion gate** for models produced by **Pipelines** (Tuesday) and **HPO** (Thursday).
- **Feature Store** (Tuesday) feeds **lineage** and **reproducible** training data definitions (Wednesday).
- **Transformers** (Wednesday) supersede **LSTM** for most NLP, but **RNN ideas** remain in **DeepAR** and historical seq2seq.
- **Inference patterns** and **Model Monitor** (Friday) close the loop with **MLOps** monitoring pillar (Monday).
- **Evolv** topics (**Spot**, **Inference Recommender**, **VPC/KMS**) answer **cost**, **security**, and **operations** interview depth beyond notebook workflows.
