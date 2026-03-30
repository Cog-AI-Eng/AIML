# Week 2 Wednesday -- Applied ML Transformers + Advanced SageMaker Experiments and Lineage Tracking

**Total Duration:** 185 Minutes (3 Stages)
**Consolidated Activities:**
- Applied ML: Attention Mechanisms, Transformer Architecture, BERT/GPT/T5 Pre-trained Models, Transfer Learning for Text
- SM Advanced: Experiments & Trials, Lineage Tracking, Reproducibility Patterns, Cross-account Sharing, Feature Store Lineage Integration

| Block | Content | Minutes |
|-------|---------|---------|
| Stage 1 | Attention and Transformer Architecture | 55 |
| Break 1 | Stretch / Questions | 5 |
| Stage 2 | Pre-trained Models and Fine-tuning | 55 |
| Break 2 | Stretch / Questions | 5 |
| Stage 3 | Experiments, Lineage, and Reproducibility | 55 |
| Buffer | Open Q&A, Summary, Thursday Preview | 10 |

---

## Lecture Overview

**Unified Scenario -- FraudShield Risk Analytics**

Associates continue as ML engineers at FraudShield. Friday's encoder-decoder compressed the entire input into a single vector. Monday showed LSTMs solve vanishing gradients but process tokens sequentially. Transformers solve BOTH problems with attention -- parallel processing and direct connections to every input position.

1. **"Why is the encoder-decoder bottleneck a problem?"** (Attention lets the decoder look at every input position, not just a compressed summary)
2. **"How do transformers process all tokens at once?"** (Self-attention replaces sequential recurrence with parallel computation)
3. **"Can we reuse someone else's transformer?"** (Pre-trained models: BERT, GPT, T5 -- transfer learning for text)
4. **"How do we track which experiment produced which model?"** (SageMaker Experiments, lineage, reproducibility)

Stage 1 builds the transformer encoder from first principles in NumPy. Stage 2 connects pre-trained transformers to the transfer learning pattern from Friday's MobileNet fine-tuning. Stage 3 introduces SageMaker Experiments and lineage tracking, tying back to Tuesday's Feature Store to show how features, experiments, and models connect.

---

## Pre-Lecture Setup

### Instructor Checklist

- [ ] Monday's notebook completed (RNN/LSTM/GRU concepts established)
- [ ] Tuesday's Feature Store notebook completed (feature groups created)
- [ ] Companion lecture notebook (`W2-Wednesday-notebook.ipynb`) open and tested
- [ ] SageMaker execution role ARN ready
- [ ] AWS account with SageMaker access verified
- [ ] `transformers` and `torch` libraries installed in the notebook environment
- [ ] Internet access available for Hugging Face model downloads
- [ ] Budget from Friday still active
- [ ] This instructor guide open in a second tab

### Student Prerequisites

- [ ] Completed readings: Attention Mechanisms CT, Transformer Architecture CT, BERT/GPT/T5 CT, Transfer Learning CT, Experiments & Trials CT, Lineage Tracking CT, Reproducibility Patterns CT, Cross-account Sharing CT, Feature Store Lineage Integration CT
- [ ] Monday's notebook completed (understand RNNs, LSTMs, encoder-decoder)
- [ ] Tuesday's notebook completed (Feature Store groups created)
- [ ] AWS credentials configured, SageMaker SDK installed

---

# STAGE 1 -- Attention and Transformer Architecture (55 min)

> **Goal:** Understand why attention was invented, implement scaled dot-product attention from scratch, extend it to multi-head attention, add positional encoding, and assemble a complete transformer encoder block.

**Exit Criteria Addressed:**
- Explain the encoder-decoder bottleneck and how attention solves it (Required)
- Implement scaled dot-product attention with query, key, and value matrices (Required)
- Describe multi-head attention and its advantages over single-head attention (Required)
- Explain why transformers need positional encoding and how sinusoidal encoding works (Required)
- Diagram a complete transformer encoder block (Required)

### Instructor Opening (3 minutes -- talk, no code)

> "Monday you implemented LSTMs and saw how they solve the vanishing gradient problem. But LSTMs have two remaining limitations. First, they process tokens one at a time -- the computation at step 50 cannot begin until steps 1 through 49 are finished. On a modern GPU with thousands of cores, that sequential bottleneck wastes most of the hardware. Second, Friday's encoder-decoder compressed the entire input into a single fixed-size vector. For a 200-word paragraph, that vector must somehow encode everything -- and it inevitably loses detail."

> "Transformers solve both problems. Instead of processing tokens sequentially, they compute relationships between all tokens in parallel. Instead of compressing into a single vector, they let every output position attend directly to every input position. The result is the architecture behind BERT, GPT, and every modern language model."

---

## STEP 1 -- Motivating Attention: The Bottleneck Problem (5 minutes)

**Pacing: conceptual, notebook markdown.**

> "Recall Friday's encoder-decoder. The encoder reads 'I love machine learning' and produces one vector -- the context vector. The decoder must generate the entire translation from that single vector. If the source sentence is 50 words long, that one vector must encode all 50 words. Information inevitably gets lost."

Draw the comparison:

```
Without attention:
  Encoder: x_1, x_2, ..., x_n --> [LSTM] --> single context vector --> Decoder

With attention:
  Encoder: x_1, x_2, ..., x_n --> [LSTM] --> h_1, h_2, ..., h_n
                                                    |  |       |
  Decoder at each step: ---------> weighted sum of ALL encoder hidden states
```

> "Attention lets the decoder 'look back' at every encoder hidden state when generating each output token. For the word 'learning', the decoder can focus on the relevant source words instead of relying on a compressed summary."

> "Transformers take this further: they remove the LSTM entirely. Self-attention computes relationships between all positions in the same sequence, in parallel."

---

## STEP 2 -- Scaled Dot-Product Attention in NumPy (12 minutes)

**Pacing: live code in notebook. Walk through every line.**

Key teaching points:

- **Query (Q):** "What am I looking for?" -- the current position's representation
- **Key (K):** "What do I contain?" -- each position's identifier
- **Value (V):** "What information do I carry?" -- each position's content
- **Scores:** Q * K^T -- how relevant is each key to each query
- **Scaling:** divide by sqrt(d_k) to prevent softmax saturation
- **Attention weights:** softmax turns scores into a probability distribution
- **Output:** weighted sum of values

Walk through the 4-token demo:

1. Create Q, K, V matrices for 4 tokens with d_k = 8
2. Compute raw scores (Q @ K^T)
3. Scale by sqrt(d_k)
4. Apply softmax
5. Compute weighted sum of values
6. Visualize the attention weight heatmap

> "Each row of the attention weight matrix shows how much each token attends to every other token. This is the core mechanism -- everything else in the transformer is built around this operation."

---

## STEP 3 -- Why Scaling Matters (5 minutes)

**Pacing: run the comparison cell.**

> "Why divide by sqrt(d_k)? Watch what happens when we skip the scaling."

Show the comparison between d_k=8 and d_k=512:

- With d_k=8, raw dot products are small and softmax produces a smooth distribution
- With d_k=512, raw dot products are large and softmax collapses to a near-one-hot vector
- After scaling, both produce reasonable distributions

> "Large dot products push softmax into regions where gradients are tiny. Scaling keeps the variance of the scores at approximately 1 regardless of dimension, maintaining healthy gradients."

---

## STEP 4 -- Multi-Head Attention (10 minutes)

**Pacing: markdown explanation, then code walkthrough.**

> "Single-head attention computes one set of attention weights. But a token might need to attend to different things simultaneously: syntactic role, semantic meaning, positional relationships. Multi-head attention runs several attention computations in parallel, each learning different relationship patterns."

Key points:

| Concept | Single-Head | Multi-Head (4 heads) |
|---------|-------------|---------------------|
| Q, K, V dimension | d_model = 32 | d_head = d_model / n_heads = 8 |
| Attention matrices | 1 | 4 (each captures different patterns) |
| Output | One weighted sum | Concatenation of 4 weighted sums |
| Parameters | Same total | Same total (split across heads) |

Walk through the code:

1. Split Q, K, V into n_heads pieces
2. Compute scaled dot-product attention per head
3. Concatenate head outputs
4. Apply output projection (W_O)

> "The parameter count does not increase -- we simply partition the dimensions across heads. Each head operates in a lower-dimensional subspace, learning specialized attention patterns."

---

## STEP 5 -- Positional Encoding (8 minutes)

**Pacing: markdown explanation, then code + visualization.**

> "Self-attention treats all token positions equally -- it has no notion of order. The sentence 'dog bites man' and 'man bites dog' would produce identical attention patterns without positional information. We need to inject position into the representation."

> "The original transformer paper uses sinusoidal positional encoding. Each position gets a unique vector computed from sine and cosine functions at different frequencies."

The formula:

```
PE(pos, 2i)     = sin(pos / 10000^(2i/d_model))
PE(pos, 2i + 1) = cos(pos / 10000^(2i/d_model))
```

> "Low-frequency dimensions encode coarse position (beginning vs end), high-frequency dimensions encode fine position (adjacent tokens). The result is that every position gets a unique signature, and relative positions are captured through linear relationships between the encodings."

Run the visualization cell to show the heatmap of positional encodings.

---

## STEP 6 -- Full Transformer Encoder Block (7 minutes)

**Pacing: conceptual summary with diagram. No new code -- assemble what was built.**

> "We now have all the components. A single transformer encoder block stacks them:"

```
Input embeddings + Positional encoding
            |
    [Multi-Head Self-Attention]
            |
    + Residual connection
            |
    [Layer Normalization]
            |
    [Feed-Forward Network (2 linear layers + ReLU)]
            |
    + Residual connection
            |
    [Layer Normalization]
            |
        Output
```

Key points:
- **Residual connections** let gradients flow (same principle as LSTMs' cell state highway, and ResNets' skip connections)
- **Layer normalization** stabilizes training
- **Feed-forward network** adds non-linear capacity per position
- Stack N of these blocks (BERT uses 12, GPT-3 uses 96)

> "This block processes all positions in parallel. No sequential dependency. On a GPU, a 512-token sequence can be processed in the same time as a 1-token sequence -- the computation is matrix multiplication, which GPUs handle efficiently."

**Discussion Prompt:** "What is the computational trade-off of self-attention compared to LSTMs?" (Self-attention is O(n^2) in sequence length because every token attends to every other. LSTMs are O(n). For very long sequences, this quadratic cost becomes the bottleneck.)

[PAUSE FOR BREAK - 5 MINS]

---

# STAGE 2 -- Pre-trained Models and Fine-tuning (55 min)

> **Goal:** Understand the landscape of pre-trained transformer models (BERT, GPT, T5), connect text transfer learning to Friday's image transfer learning, and use Hugging Face pipelines for practical NLP tasks.

**Exit Criteria Addressed:**
- Compare BERT, GPT, and T5 architectures and their training objectives (Required)
- Explain transfer learning for text and connect it to image transfer learning (Required)
- Use pre-trained models for classification, generation, and summarization (Required)
- Determine when to fine-tune versus use off-the-shelf models (Required)

### Instructor Opening (2 minutes)

> "In Stage 1 you built attention and transformer blocks from scratch. In practice, nobody trains a transformer from zero on their own data -- the compute cost is measured in millions of dollars. Instead, we use pre-trained models. This is the same transfer learning idea from Friday with MobileNet, but applied to text."

---

## STEP 7 -- BERT vs GPT vs T5 Comparison (10 minutes)

**Pacing: conceptual with notebook comparison table.**

> "Three families dominate the pre-trained transformer landscape. Each uses a different part of the transformer and a different training objective."

| Model | Architecture | Training Objective | Best For |
|-------|-------------|-------------------|----------|
| **BERT** | Encoder only | Masked Language Modeling (predict hidden words) | Classification, NER, question answering |
| **GPT** | Decoder only | Causal Language Modeling (predict next word) | Text generation, conversation, code |
| **T5** | Encoder-Decoder | Text-to-Text (convert every task to text output) | Summarization, translation, general-purpose |

Walk through each:

**BERT (Bidirectional Encoder Representations from Transformers):**
- Reads the entire sentence at once (bidirectional)
- Trained by masking 15% of tokens and predicting them from context
- The [CLS] token's output serves as a sentence-level representation
- Fine-tune by adding a classification head on top

**GPT (Generative Pre-trained Transformer):**
- Reads left to right only (causal / autoregressive)
- Trained by predicting the next token given all previous tokens
- Generates text by sampling one token at a time
- Scales with more parameters (GPT-2: 1.5B, GPT-3: 175B, GPT-4: undisclosed)

**T5 (Text-to-Text Transfer Transformer):**
- Full encoder-decoder architecture
- Every task is framed as "text in, text out"
- Translation: "translate English to French: How are you?" -> "Comment allez-vous?"
- Summarization: "summarize: [long article]" -> "[short summary]"

> "Notice the connection to Friday's encoder-decoder: T5 is literally the transformer version of what we built with LSTMs, but with attention replacing recurrence."

---

## STEP 8 -- Transfer Learning for Text (8 minutes)

**Pacing: conceptual, connecting to Friday's MobileNet fine-tuning.**

> "On Friday, you fine-tuned MobileNet V2 for CIFAR-10 images. The pattern was: take a model pre-trained on ImageNet (1.2M images), freeze the feature extractor, retrain only the classification head. Text transfer learning follows exactly the same pattern."

| Concept | Image (Friday) | Text (Today) |
|---------|----------------|-------------|
| Pre-trained model | MobileNet V2 | BERT-base |
| Pre-training data | ImageNet (1.2M images, 1000 classes) | BookCorpus + Wikipedia (3.3B words) |
| What it learned | Edges, textures, shapes, objects | Grammar, semantics, world knowledge |
| Frozen layers | Convolutional feature extractor | Transformer encoder layers |
| Retrained layer | Top classification head | Classification head on [CLS] token |
| Fine-tuning data | CIFAR-10 (50K images, 10 classes) | Your task-specific labeled data |

> "The insight is identical: features learned on large generic datasets transfer to specific tasks. A BERT model that learned English grammar from Wikipedia can classify fraud reports without being trained on fraud data from scratch."

---

## STEP 9 -- Hugging Face Pipeline: Sentiment Classification (8 minutes)

**Pacing: live code. Run the cell and discuss the output.**

> "Hugging Face is the standard library for working with pre-trained transformers. The `pipeline` API abstracts tokenization, model loading, and inference into a single function call."

Run the sentiment analysis pipeline. Discuss:
- The model returns a label and a confidence score
- Under the hood: text is tokenized, passed through a fine-tuned DistilBERT, and the [CLS] token's output is projected to sentiment logits
- This model was fine-tuned on SST-2 (Stanford Sentiment Treebank) -- not trained from scratch

> "We did not train this model. We did not provide any labeled data. Someone already fine-tuned it on sentiment data and shared it on the Hugging Face Hub. We benefit from their work with one line of code."

---

## STEP 10 -- Hugging Face Pipeline: Text Generation (8 minutes)

**Pacing: live code. Run and discuss.**

> "GPT models generate text by predicting one token at a time. Each predicted token is appended to the input, and the model predicts the next token from the extended sequence. This is the autoregressive decoding pattern from Friday's encoder-decoder, but without the encoder."

Run the text generation cell. Discuss:
- `max_new_tokens` controls output length
- `temperature` controls randomness (higher = more creative, lower = more deterministic)
- The model generates plausible continuations because it was trained on billions of words

**Discussion Prompt:** "Could you use this text generation model for fraud detection? Why or why not?" (Not directly -- it generates text, it does not classify. You would need to fine-tune a classifier or use prompt engineering with a larger model.)

---

## STEP 11 -- Hugging Face Pipeline: Summarization (8 minutes)

**Pacing: live code. Run and discuss.**

> "Summarization uses an encoder-decoder model (T5 or BART). The encoder reads the full text, and the decoder generates a shorter version. This is exactly the encoder-decoder pattern from Friday, but with transformers instead of LSTMs."

Run the summarization cell with a multi-sentence paragraph. Discuss:
- `max_length` and `min_length` control summary length
- The model extracts key information and rephrases
- This is abstractive summarization (generates new sentences) vs. extractive (selects existing sentences)

---

## STEP 12 -- When to Fine-tune vs Off-the-shelf (8 minutes)

**Pacing: discussion-driven with notebook markdown.**

> "We have used three off-the-shelf models. When should you fine-tune instead?"

| Scenario | Recommendation | Why |
|----------|---------------|-----|
| Standard sentiment analysis | Off-the-shelf | Pre-trained models already handle this well |
| FraudShield transaction classification (fraud/not fraud from text descriptions) | Fine-tune | Domain-specific vocabulary and patterns |
| Generating marketing copy | Off-the-shelf (large GPT) | General language generation |
| Classifying internal support tickets into 47 custom categories | Fine-tune | Custom label set not in pre-training |
| Quick prototype / proof of concept | Off-the-shelf | Speed matters more than accuracy |
| Production system with strict accuracy requirements | Fine-tune | Need to optimize for your specific data |

> "The decision comes down to: does your task differ enough from the pre-training distribution that the model's default behavior is insufficient? If yes, fine-tune. If no, use off-the-shelf and invest your engineering time elsewhere."

**Discussion Prompt:** "For FraudShield's document verification system (detecting forged IDs), would you fine-tune a text model or an image model? Why?" (Image model -- the task is visual, not textual. You would fine-tune a vision transformer or CNN, not BERT.)

[PAUSE FOR BREAK - 5 MINS]

---

# STAGE 3 -- Experiments, Lineage, and Reproducibility (55 min)

> **Goal:** Use SageMaker Experiments to track training runs, compare metrics across runs, understand lineage tracking from data to deployment, connect Feature Store to lineage, and establish reproducibility patterns.

**Exit Criteria Addressed:**
- Create a SageMaker Experiment and log metrics, parameters, and artifacts to a Run (Required)
- Compare multiple experiment runs to identify the best model configuration (Required)
- Describe lineage tracking from data source through processing to deployed endpoint (Required)
- Connect Feature Store feature groups to experiment lineage (Required)
- Implement reproducibility patterns for ML workflows (Required)

### Instructor Opening (3 minutes)

> "In Stages 1 and 2 you learned the architecture behind modern NLP. But architecture is only half the story. In practice, you train dozens of variations: different hyperparameters, different feature sets, different preprocessing steps. Without a system to track what you tried, what worked, and why, you end up in spreadsheet chaos."

> "SageMaker Experiments is that system. It organizes your work into experiments and runs, logs every metric and parameter, and connects to lineage tracking so you can trace any model back to the exact data and code that produced it. This ties directly to Tuesday's Feature Store -- the features you engineered are part of the lineage."

---

## STEP 13 -- SageMaker Experiments Concepts (5 minutes)

**Pacing: conceptual with notebook markdown.**

> "Think of Experiments like a lab notebook. An Experiment is a project. A Run is one attempt within that project."

| Concept | Analogy | Example |
|---------|---------|---------|
| **Experiment** | A research project | "FraudShield Transaction Classifier" |
| **Run** | One attempt in that project | "RF with 100 trees, feature set A" |
| **Parameters** | What you configured | n_estimators=100, max_depth=10 |
| **Metrics** | What you measured | accuracy=0.92, f1=0.85 |
| **Artifacts** | What you produced | model.tar.gz, confusion_matrix.png |

> "Every training job in SageMaker can automatically log to an experiment run. But you can also log manually for custom workflows, local experiments, or third-party tools."

---

## STEP 14 -- Create an Experiment and Log Metrics (12 minutes)

**Pacing: live code in notebook.**

Walk through the code step by step:

1. Create an experiment named `fraudshield-transformer-study`
2. Create a run within that experiment
3. Log parameters (model type, hyperparameters, feature count)
4. Log metrics (accuracy, precision, recall, F1 -- using Monday's RF validation results as example values)
5. Log an artifact (the model S3 path)

> "Notice we are using Monday's RF metrics as example data. The point is the tracking mechanism, not the specific model. In practice, you would log these metrics at the end of every training job."

Key API calls to highlight:
- `Run.log_parameter(name, value)` -- for configuration
- `Run.log_metric(name, value, step=)` -- for results (step is optional, for time-series metrics)
- `Run.log_file(file_path, name=)` -- for artifacts

---

## STEP 15 -- Compare Experiment Runs (10 minutes)

**Pacing: live code. Create a second run with different parameters, then compare.**

> "The value of experiments becomes clear when you have multiple runs to compare."

Create a second run with different hyperparameters (e.g., 200 trees, max_depth=5) and slightly different metrics. Then:

1. List all runs in the experiment
2. Extract parameters and metrics from each run
3. Display a comparison table
4. Identify the best run by F1 score

> "In production, you might have 50 or 100 runs. The Experiments API lets you filter, sort, and compare programmatically instead of scrolling through CloudWatch logs. SageMaker Studio also provides a visual comparison UI."

**Discussion Prompt:** "Besides hyperparameters, what else should you vary across experiment runs?" (Feature sets from Feature Store, preprocessing steps, training data versions, model architectures.)

---

## STEP 16 -- Lineage Tracking Concepts (8 minutes)

**Pacing: conceptual with notebook diagrams.**

> "Experiments track what you tried and what happened. Lineage tracking goes further: it records the causal chain from raw data to deployed endpoint."

Draw the lineage chain:

```
Data Source (S3)
    |
Processing Job (Data Wrangler / Processing)
    |
Feature Group (Feature Store)       <-- Tuesday's work
    |
Training Dataset
    |
Training Job (Experiment Run)       <-- Today's Stage 3
    |
Model Artifact (S3)
    |
Model Package (Model Registry)     <-- Monday's work
    |
Endpoint (Deployed Model)          <-- Monday's work
```

> "If a deployed model starts producing bad predictions, lineage tracking lets you trace back: which training job produced it, what data was used, what features were selected, what preprocessing was applied. Without lineage, debugging a production model is detective work."

Key SageMaker lineage entities:
- **Context:** groups related entities (an experiment, a pipeline)
- **Action:** a process that transforms data (a training job, a processing job)
- **Artifact:** a data object (a dataset, a model, an endpoint)
- **Association:** a link between entities (input/output relationships)

---

## STEP 17 -- Feature Store Lineage Integration (8 minutes)

**Pacing: conceptual, connecting to Tuesday.**

> "On Tuesday you created feature groups in SageMaker Feature Store. Those feature groups are lineage entities. When a training job reads from a feature group, SageMaker records that relationship."

The connection:

```
Feature Group: "transaction_features"  (Tuesday)
        |
        | [ContributedTo]
        v
Training Job: "rf-100-trees-run"       (Today's experiment)
        |
        | [Produced]
        v
Model Artifact: model.tar.gz
```

> "This means you can answer questions like: 'Which models were trained on the transaction_features feature group?' and 'If I change the amount normalization in transaction_features, which models are affected?' Feature Store lineage turns features from anonymous CSVs into tracked, versioned assets."

Key points:
- Feature groups automatically register as lineage artifacts
- Training jobs that consume feature group data create associations
- Changes to feature definitions can be traced to downstream models
- Cross-account sharing (from your CT reading) extends this lineage across organizational boundaries

---

## STEP 18 -- Reproducibility Patterns (7 minutes)

**Pacing: discussion with notebook checklist.**

> "Lineage tells you what happened. Reproducibility ensures you can make it happen again."

| Pattern | What to Track | SageMaker Mechanism |
|---------|--------------|-------------------|
| **Data versioning** | Exact dataset used for training | S3 versioning + Feature Store point-in-time queries |
| **Code versioning** | Training script, preprocessing code | Git commit hash logged as experiment parameter |
| **Environment versioning** | Container image, library versions | Docker image URI logged with training job |
| **Hyperparameter logging** | All configuration values | Experiment Run parameters |
| **Random seed control** | All sources of randomness | Logged as hyperparameter |
| **Metric logging** | All evaluation results | Experiment Run metrics |
| **Artifact tracking** | Model files, plots, reports | Experiment Run artifacts + S3 |

> "The minimum reproducibility checklist: (1) pin your random seeds, (2) version your data, (3) log your code commit, (4) record your container image, (5) log all hyperparameters. If you do these five things, any colleague can recreate your exact training run six months later."

**Discussion Prompt:** "What happens if you reproduce a training run but get different results? What could cause that?" (Non-deterministic GPU operations, floating-point ordering differences, library version mismatch, different hardware. Some randomness is inherent in distributed training.)

---

## STEP 19 -- Cleanup Experiment Runs (5 minutes)

**Pacing: live code. Clean up the experiment runs created in this session.**

> "Just as we delete endpoints after use, we clean up experiment runs when they are no longer needed. In production you would keep runs indefinitely. For a training exercise, we clean up to avoid clutter."

Walk through:
1. List runs in the experiment
2. Delete each run
3. Delete the experiment
4. Verify cleanup

---

## Wrap-up & Q&A Buffer (10 minutes)

### Summary (4 minutes)

> "Today you accomplished three things. First, you built the attention mechanism from scratch in NumPy and assembled a transformer encoder block -- self-attention replaces sequential recurrence with parallel computation, and multi-head attention lets the model capture different types of relationships simultaneously. You understand why transformers replaced LSTMs as the dominant architecture. Second, you used pre-trained transformers (BERT, GPT, T5) through Hugging Face for classification, generation, and summarization. The transfer learning pattern is identical to Friday's MobileNet: a large model pre-trained on generic data transfers its knowledge to your specific task. Third, you tracked experiments with SageMaker Experiments -- logging parameters, metrics, and artifacts -- and connected this to lineage tracking and Tuesday's Feature Store. You can now trace any model back to the exact data, features, and code that produced it."

### Thursday Preview (2 minutes)

> "Thursday covers NLP pipelines and model optimization. You will tokenize text, build end-to-end NLP workflows, and learn techniques for making models faster and smaller for deployment -- quantization, pruning, and distillation. The question shifts from 'what architecture should I use' to 'how do I make it production-ready.' Read the NLP Pipelines and Model Optimization CTs before Thursday."

### Open Q&A (4 minutes)

---

## Instructor Notes -- Common Issues

| Issue | Resolution |
|-------|-----------|
| `transformers` library not installed | Run `%pip install -q transformers torch` at the top of the notebook. |
| Hugging Face model download times out | Check internet connectivity. Models are cached after first download. |
| `torch` not found | Install with `%pip install -q torch`. CPU-only is fine for pipeline demos. |
| Attention heatmap is blank | Check that Q, K, V matrices are not all zeros. Verify the random seed produces non-degenerate values. |
| Multi-head attention output shape mismatch | Verify d_model is divisible by n_heads. Common error: d_model=32, n_heads=3 (not divisible). |
| SageMaker Experiments `ResourceNotFound` | The experiment or run name may conflict with a previously deleted one. Use a unique name with a timestamp. |
| `AccessDeniedException` on Experiments API | The execution role needs `sagemaker:Create*`, `sagemaker:Describe*`, `sagemaker:List*` permissions. |
| Pipeline demo returns unexpected results | Pre-trained models have biases and limitations. Use unexpected results as a teaching moment about model evaluation. |
| Students confused about attention vs self-attention | Attention: decoder attends to encoder outputs (cross-attention). Self-attention: tokens in the same sequence attend to each other. |
| NumPy attention implementation is slow | Expected -- pure Python loops over tokens. Emphasize this is pedagogical, not production. |
| Feature Store lineage not appearing | Lineage associations are created asynchronously. They may take a few minutes to appear after a training job completes. |
