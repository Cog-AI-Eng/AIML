# Applying AI and Machine Learning Concept Review Interview

**Activity ID:** AIML-RV-Int
**Display Name:** Applying AI and Machine Learning Concept Review Interview
**Duration:** 30 minutes
**Type:** Technical Interview

---

## Interview Overview

This interview assesses a candidate's understanding of applied machine learning across five modules: Foundations, Evaluation, Neural Networks, Deep Learning, and Transformers. The guide contains 14 prompts spanning Beginner, Intermediate, and Advanced difficulty levels. At least 70% of prompts are scenario-based.

**Time allocation suggestion:**
- Modules 1-2 (Foundations and Evaluation): ~10 minutes (Prompts 1-5)
- Module 3 (Neural Networks): ~7 minutes (Prompts 6-8)
- Modules 4-5 (Deep Learning and Transformers): ~10 minutes (Prompts 9-14)
- Buffer / follow-ups: ~3 minutes

---

## Prompt 1 -- ML Lifecycle and Reproducibility (Beginner, Scenario-Based)

**Module:** 1 -- Foundations

> You have joined TechPulse Analytics as a junior data scientist. On your first day, a colleague hands you a Jupyter notebook that trains a housing price model. Every time you run it, the results are slightly different. Your colleague says, "It is fine, the numbers are close enough." Walk me through why reproducibility matters and what concrete steps you would take to make this pipeline deterministic.

<details>
<summary>Interviewer Guide</summary>

**Target Keywords:** random seed, `random_state=42`, deterministic, reproducibility, NumPy seed, Python random seed, version pinning, data leakage, pipeline

**Expected Good Answer:**
The candidate should identify that non-deterministic results undermine trust, debugging, and comparison of experiments. They should recommend setting `random_state` in all scikit-learn calls (splits, models), setting seeds for `numpy.random` and Python's built-in `random` module, pinning dependency versions in `requirements.txt`, and ensuring data loading order is fixed. Strong answers reference that the Foundations assignment (TechPulse Analytics scenario) specifically required `random_state=42` everywhere and that Milestone 1 of the assignment tested this explicitly.

**Red Flags:**
- Cannot name any mechanism for setting seeds (e.g., does not know `random_state` parameter)
- Dismisses reproducibility as unimportant or "nice to have"
- Confuses reproducibility with model accuracy
- Does not mention that seeds must be set before any randomized operation

**Follow-up Prompts:**
- "What happens if you set a seed for scikit-learn's `train_test_split` but not for NumPy? Is the pipeline fully reproducible?"
- "In the Foundations assignment, the constraint was to use `random_state=42` everywhere. Why is a single fixed seed value important across the entire pipeline?"

</details>

---

## Prompt 2 -- Supervised vs. Unsupervised Learning (Beginner)

**Module:** 1 -- Foundations

> Explain the difference between supervised and unsupervised learning. For each, give me one concrete business problem it would solve and name a scikit-learn class you would use.

<details>
<summary>Interviewer Guide</summary>

**Target Keywords:** labeled data, classification, regression, clustering, `fit(X, y)` vs. `fit(X)`, KMeans, LogisticRegression, LinearRegression

**Expected Good Answer:**
Supervised learning uses labeled data (inputs paired with known outputs) to learn a mapping function. Business example: predicting customer churn (classification) or forecasting house prices (regression). Scikit-learn classes: `LogisticRegression`, `LinearRegression`. Unsupervised learning finds structure in data without labels. Business example: customer segmentation, anomaly detection. Scikit-learn class: `KMeans`, `PCA`. A strong candidate will note the key API difference: supervised uses `model.fit(X, y)` while unsupervised uses `model.fit(X)`.

**Red Flags:**
- Cannot distinguish between the two paradigms
- Confuses classification with clustering
- Calls a problem "unsupervised" just because labels are noisy (the curriculum explicitly warns against this)
- Cannot name any concrete scikit-learn estimator

**Follow-up Prompts:**
- "Where does reinforcement learning fit relative to these two? Give me a one-sentence distinction."
- "A colleague says their customer segmentation problem is unsupervised because the labels are messy. How would you respond?"

</details>

---

## Prompt 3 -- Algorithm Selection and Baselines (Intermediate, Scenario-Based)

**Module:** 1 -- Foundations

> You are handed a tabular dataset with 5,000 rows, 30 numeric features, and a binary target column. Your manager wants a model by end of week. Walk me through the algorithm selection framework you would use, starting from problem framing through to your shortlist of candidate models. Why would you build a baseline first?

<details>
<summary>Interviewer Guide</summary>

**Target Keywords:** algorithm selection framework, problem type, data description, baselines first, DummyClassifier, LogisticRegression, cross-validation, multiple metrics, constraints, interpretability

**Expected Good Answer:**
The candidate should articulate the five-step framework from the curriculum: (1) anchor on problem type -- binary classification, supervised; (2) describe data -- 5,000 rows is moderate, 30 numeric features, check for class imbalance; (3) apply constraints early -- interpretability needs, latency, regulatory requirements; (4) baselines first -- `DummyClassifier` to see "how hard is this if I barely try," then `LogisticRegression` as a linear baseline; (5) shortlist candidates -- maybe tree-based ensembles, compare with same validation strategy and multiple metrics. The "baselines first" rule exists because they turn model selection from opinion into measurement. The Foundations assignment (Milestone 7) explicitly required implementing `select_algorithm()` with structured rationale.

**Red Flags:**
- Jumps straight to complex models (XGBoost, neural nets) without considering baselines
- Does not mention checking class imbalance in the target
- No awareness of constraints beyond accuracy
- Cannot explain why a dummy model is useful

**Follow-up Prompts:**
- "Your baseline logistic regression gets 0.92 accuracy but only 0.40 F1 on the minority class. What does this tell you, and what would you do next?"
- "When would you escalate from linear models to something more complex? What diagnostic would tell you it is time?"

</details>

---

## Prompt 4 -- Data Splitting and Leakage (Intermediate, Scenario-Based)

**Module:** 2 -- Evaluation

> In the Evaluation assignment, you built a patient readmission prediction pipeline for a hospital analytics company. The dataset had severe class imbalance -- about 15% readmissions. A junior team member scaled the features using the entire dataset before splitting into train and test sets. Explain what went wrong, why it matters, and how you would fix it.

<details>
<summary>Interviewer Guide</summary>

**Target Keywords:** data leakage, fit on training set only, `StandardScaler`, `fit_transform` on train, `transform` on test, stratification, class imbalance, train/validation/test split

**Expected Good Answer:**
The candidate should explain that fitting a scaler (or any transformer) on the full dataset before splitting constitutes data leakage -- the test set's statistics bleed into the training pipeline, producing overly optimistic evaluation metrics that do not reflect real-world performance. The fix: split first, then `fit_transform` the scaler on the training set only and `transform` the test set. They should also mention stratified splitting to preserve the 15/85 class ratio in each subset. Strong candidates will reference that the Foundations assignment Milestone 3 explicitly tested for leakage ("CRITICAL: You must fit the scaler on the training set ONLY") and the Evaluation assignment required stratified splits.

**Red Flags:**
- Does not know what data leakage is
- Cannot articulate why fitting on the full dataset inflates metrics
- Describes the fix in vague terms without specifying the correct fit/transform order
- Does not mention stratification for an imbalanced dataset

**Follow-up Prompts:**
- "Beyond scaling, give me another example of how data leakage can occur in a pipeline."
- "Why is stratification important for this dataset specifically?"

</details>

---

## Prompt 5 -- Metrics, AUC-ROC, and SHAP (Advanced, Scenario-Based)

**Module:** 2 -- Evaluation

> You have trained a logistic regression model for the hospital readmission prediction task. The model achieves 89% accuracy. Your manager is satisfied, but you are not. Why might accuracy be misleading here? Walk me through how you would use precision, recall, F1, AUC-ROC, and a confusion matrix together to present a more honest evaluation. Then explain how you would use SHAP values to investigate whether the model is relying on the right features.

<details>
<summary>Interviewer Guide</summary>

**Target Keywords:** class imbalance, accuracy paradox, precision, recall, F1-score, AUC-ROC, confusion matrix, threshold selection, SHAP values, Shapley, feature importance, data leakage detection, beeswarm plot, waterfall plot

**Expected Good Answer:**
With 85% negative class, a model that always predicts "no readmission" gets 85% accuracy -- that is the accuracy paradox. The candidate should explain: precision measures how many predicted readmissions were real; recall measures how many actual readmissions were caught; F1 balances both. AUC-ROC evaluates ranking quality across thresholds -- critical because the deployment threshold depends on the cost of false negatives (missed readmissions) vs. false positives (unnecessary follow-ups). The confusion matrix shows the specific error breakdown at a chosen operating point. For SHAP: each feature gets a contribution value showing how it pushes a prediction above or below the baseline. They would look at global feature importance (bar plot of mean |SHAP|) to sanity-check that top features make clinical sense and local explanations (waterfall) for individual patients. If a proxy or leaky feature dominates, investigate. The Evaluation assignment (Milestone 4-6) required implementing all these metrics, and Milestone 6 required SHAP values.

**Red Flags:**
- Accepts 89% accuracy at face value without questioning class imbalance
- Cannot explain the relationship between precision and recall
- Thinks AUC-ROC is a single threshold metric rather than an aggregate across thresholds
- Describes SHAP as "just feature importance" without mentioning the additive contribution or baseline concept
- Cannot connect SHAP to practical debugging (leakage detection, fairness review)

**Follow-up Prompts:**
- "If the confusion matrix shows 200 false negatives and 50 false positives at the default 0.5 threshold, how would you use the ROC curve to find a better operating point?"
- "A SHAP analysis shows that 'patient_id' is the most important feature. What does this tell you?"

</details>

---

## Prompt 6 -- Bias-Variance and Regularization (Intermediate, Scenario-Based)

**Module:** 2 -- Evaluation

> Your model scores 0.98 R-squared on the training set but only 0.72 on the validation set. Diagnose the problem using the bias-variance framework, and then recommend a regularization strategy. What is the difference between using L1 and L2 regularization, and when would you choose one over the other?

<details>
<summary>Interviewer Guide</summary>

**Target Keywords:** overfitting, high variance, low bias, learning curve, regularization, L1/Lasso, L2/Ridge, sparsity, feature selection, Elastic Net, alpha hyperparameter, dropout

**Expected Good Answer:**
The gap between training (0.98) and validation (0.72) indicates high variance / overfitting -- the model is memorizing training noise rather than learning generalizable patterns. The candidate should mention learning curves as a diagnostic tool. For regularization: L2 (Ridge) adds a penalty proportional to squared coefficients, shrinking all weights smoothly -- good when many features contribute a little and you want stable coefficients. L1 (Lasso) adds a penalty proportional to absolute values, which can drive coefficients to exactly zero, performing implicit feature selection -- good when you suspect only a subset of features matter. They might mention Elastic Net as a combination of both. For neural networks, dropout serves a similar purpose by randomly zeroing activations during training. Strong candidates will reference the Evaluation assignment where they implemented Ridge and Lasso logistic regression models.

**Red Flags:**
- Confuses bias-variance direction (calls this "high bias" or "underfitting")
- Cannot explain the mathematical difference between L1 and L2
- Does not know that L1 can produce sparse models
- Recommends adding more features or complexity to an already overfitting model

**Follow-up Prompts:**
- "If you plot a learning curve and the validation curve is still climbing as you add more data, what does that suggest?"
- "In neural networks, what is the analogous technique to regularization, and how does it work mechanically?"

</details>

---

## Prompt 7 -- Perceptrons, Activations, and MLPs (Intermediate, Scenario-Based)

**Module:** 3 -- Neural Networks

> In the Neural Networks assignment, you built a perceptron and then an MLP for digit classification. Explain why a single perceptron cannot solve the XOR problem, and then describe how adding hidden layers with non-linear activation functions fixes this. If I asked you to choose between ReLU, sigmoid, and tanh for the hidden layers of a deep network, which would you pick and why?

<details>
<summary>Interviewer Guide</summary>

**Target Keywords:** linear decision boundary, XOR, non-linearity, activation function, hidden layers, ReLU, vanishing gradient, sigmoid saturation, tanh, dying ReLU, Leaky ReLU, GELU, linear/dense layer, sequential layer stack

**Expected Good Answer:**
A single perceptron computes a weighted sum plus bias and passes it through an activation -- producing a linear decision boundary. XOR is not linearly separable, so no single hyperplane can classify it. Adding hidden layers with non-linear activations (like ReLU) allows the network to compose multiple linear boundaries into non-linear regions. For hidden layers in a deep network, ReLU is the standard choice because its derivative is 1 for positive inputs, preventing vanishing gradients during backpropagation. Sigmoid and tanh saturate for large or small inputs, producing near-zero gradients that choke learning in deep architectures. The candidate should mention the "dying ReLU" tradeoff (neurons stuck at zero) and that Leaky ReLU or GELU are alternatives. Strong answers reference the assignment's MLP class that accepted configurable activations ("relu", "tanh", "sigmoid") and used a sequential layer stack.

**Red Flags:**
- Cannot explain why non-linearity is necessary (thinks stacking more linear layers helps)
- Does not know the vanishing gradient problem
- Recommends sigmoid for hidden layers of a deep network
- Cannot articulate what ReLU actually computes (max(0, z))

**Follow-up Prompts:**
- "If you stacked ten linear layers without any activation function, what would the resulting computation be equivalent to?"
- "When would sigmoid be the correct choice for an activation function?"

</details>

---

## Prompt 8 -- Backpropagation and Optimizers (Intermediate, Scenario-Based)

**Module:** 3 -- Neural Networks

> You are training an MLP for digit classification. After 50 epochs, the training loss has plateaued and is barely decreasing. Walk me through the training loop mechanics -- forward pass, loss computation, backward pass, weight update -- and then diagnose what might be causing the plateau. How would you decide between SGD and Adam, and what role does a learning rate scheduler play?

<details>
<summary>Interviewer Guide</summary>

**Target Keywords:** forward pass, loss function, cross-entropy, backward pass / gradient computation, gradient, chain rule, backpropagation, weight update step, gradient clearing, SGD, momentum, Adam, adaptive learning rate, learning rate scheduler, cosine annealing schedule, reduce-on-plateau schedule

**Expected Good Answer:**
The candidate should describe the training loop: (1) forward pass -- input flows through layers to produce predictions; (2) loss computation -- compare predictions to labels using a loss function (cross-entropy for classification); (3) backward pass -- gradient computation via the chain rule, walking backward through the computational graph; (4) weight update -- the optimizer adjusts weights in the direction that reduces loss; (5) gradient clearing resets accumulated gradients before the next iteration. For the plateau: learning rate may be too small, model may lack capacity (underfitting), or the optimizer may be stuck. SGD is simple but requires careful LR tuning and benefits from momentum; Adam adapts per-parameter learning rates using first and second moments, making it less sensitive to initial LR and good for fast prototyping. A scheduler (e.g., cosine annealing) decreases LR over time for fine-grained convergence. The Neural Networks assignment (Milestone 4) required comparing SGD and Adam on identical architectures.

**Red Flags:**
- Cannot describe the backward pass / gradient computation step or mentions "automatic" without explaining the chain rule concept
- Does not know why gradient clearing is necessary (gradients accumulate by default in many frameworks)
- Cannot name a concrete difference between SGD and Adam
- Suggests only changing the model architecture without considering the optimizer or learning rate

**Follow-up Prompts:**
- "Why do some deep learning frameworks accumulate gradients by default? Is there ever a scenario where you would want that?"
- "In the assignment's optimizer comparison, what pattern did you observe when comparing SGD and Adam convergence curves?"

</details>

---

## Prompt 9 -- CNNs for Image Data (Intermediate, Scenario-Based)

**Module:** 4 -- Deep Learning

> Your team needs to build an image classifier for a dataset of 32x32 color images with 10 classes (think CIFAR-10). A colleague suggests flattening each image and feeding it into a fully connected network. Explain why this is a poor approach and how a CNN solves the problem. Describe the key building blocks of the CNN architecture you would design.

<details>
<summary>Interviewer Guide</summary>

**Target Keywords:** spatial structure, local connectivity, weight sharing, convolutional layer, kernel/filter, feature map, pooling, max pooling, translation invariance, stride, padding, flatten, fully connected head, data augmentation, overfitting

**Expected Good Answer:**
Flattening a 32x32x3 image produces 3,072 features, and a fully connected layer would have millions of parameters for a modest hidden size -- wasteful because distant pixels rarely interact directly. This approach ignores spatial structure. CNNs solve this via: (1) convolutional layers with small learnable filters (e.g., 3x3) that slide across the image, detecting local patterns (edges, textures) with far fewer parameters due to weight sharing; (2) pooling layers (e.g., max pooling) that downsample feature maps, reducing computation and adding translation invariance; (3) stacking conv-relu-pool blocks to build hierarchical features (low-level edges to high-level objects); (4) flattening and feeding into a fully connected classifier head. The candidate should also mention data augmentation (random flips, rotations, color jitter) for training -- applied only to training data, not validation/test. Strong answers will describe the architecture pattern: `Conv -> ReLU -> MaxPool -> Conv -> ReLU -> MaxPool -> Flatten -> Dense -> ReLU -> Dense`.

**Red Flags:**
- Cannot explain weight sharing or local connectivity
- Does not know what a convolutional filter/kernel does
- Confuses pooling with convolution
- Applies data augmentation to the test set
- Cannot articulate why CNNs have fewer parameters than fully connected networks on image data

**Follow-up Prompts:**
- "What is the purpose of padding in a convolutional layer?"
- "Why would horizontal flipping be a good augmentation for natural photos but harmful for digit recognition?"

</details>

---

## Prompt 10 -- RNNs, LSTMs, and GRUs (Advanced, Scenario-Based)

**Module:** 4 -- Deep Learning

> You are building a sentiment classification model for product reviews that average 150 words. You start with a vanilla RNN but notice it performs poorly on reviews where the sentiment-determining word appears early in the text. Diagnose the problem, explain how LSTMs and GRUs address it, and describe the gating mechanism in an LSTM at a conceptual level. When would you choose a GRU over an LSTM?

<details>
<summary>Interviewer Guide</summary>

**Target Keywords:** vanishing gradient, backpropagation through time, long-range dependency, LSTM, cell state, forget gate, input gate, output gate, additive update, GRU, reset gate, update gate, bidirectional, LSTM layer, GRU layer

**Expected Good Answer:**
Vanilla RNNs suffer from the vanishing gradient problem: during backpropagation through time, gradients shrink exponentially across steps, so the network cannot learn dependencies spanning more than about 10-20 steps. For a 150-word review, the early sentiment signal is lost. LSTMs solve this with a cell state -- a separate memory channel that uses additive updates (not multiplicative through squashing functions), allowing gradients to flow backward without shrinking. Three gates control the cell state: (1) forget gate -- decides what to erase from memory; (2) input gate -- decides what new information to write; (3) output gate -- decides what to expose as the hidden state. GRUs simplify this to two gates (reset and update) and merge the cell state with the hidden state. GRUs are faster to train and have fewer parameters, making them preferable when data is limited or speed matters. For very long sequences (500+ steps), LSTMs often perform better. Bidirectional variants process the sequence in both directions, which is useful when the entire sequence is available at inference time.

**Red Flags:**
- Does not know the vanishing gradient problem or cannot connect it to RNNs
- Cannot explain the LSTM cell state or any of the gates conceptually
- Confuses LSTM and GRU architectures
- Does not know the practical tradeoffs between LSTM and GRU

**Follow-up Prompts:**
- "Why does the additive update in the LSTM cell state help preserve gradients, compared to the multiplicative update in a vanilla RNN?"
- "When would a bidirectional LSTM be inappropriate?"

</details>

---

## Prompt 11 -- Encoder-Decoder and Attention (Advanced, Scenario-Based)

**Module:** 4-5 -- Deep Learning / Transformers

> You are building a machine translation system that translates English sentences to French. The sentences vary in length from 5 to 50 words. Explain the encoder-decoder (seq2seq) architecture you would use, including what teacher forcing is. Then explain why this architecture struggles with long sentences, and how the attention mechanism solves that problem. Walk me through the scaled dot-product attention formula.

<details>
<summary>Interviewer Guide</summary>

**Target Keywords:** encoder, decoder, context vector, bottleneck, teacher forcing, scheduled sampling, seq2seq, attention, Query, Key, Value, scaled dot-product, softmax, `QK^T / sqrt(d_k)`, information bottleneck

**Expected Good Answer:**
Encoder-decoder: The encoder (typically LSTM/GRU) reads the entire input sequence and compresses it into a fixed-size context vector (final hidden state). The decoder generates the output one token at a time, initialized with this context vector. Teacher forcing: during training, the decoder receives the ground-truth previous token instead of its own prediction, speeding convergence but creating a train/inference mismatch. Bottleneck: the fixed-size context vector cannot hold all information from long inputs, degrading performance as length increases. Attention solves this by letting the decoder look at all encoder hidden states at each decoding step, computing a weighted sum based on relevance. Scaled dot-product attention: `Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V`. (1) QK^T computes similarity scores between queries and keys; (2) dividing by sqrt(d_k) prevents large values from pushing softmax into near-one-hot regions with vanishing gradients; (3) softmax normalizes to a distribution; (4) multiply by V to get a weighted combination of values.

**Red Flags:**
- Cannot explain why a fixed-size context vector is limiting
- Does not know what teacher forcing is or confuses it with something else
- Cannot articulate the Q/K/V framework
- Does not know why scaling by sqrt(d_k) is necessary
- Thinks attention replaces the encoder-decoder structure rather than augmenting it

**Follow-up Prompts:**
- "What is the difference between self-attention and cross-attention?"
- "If d_k is 512 and you skip the scaling, what happens to the softmax distribution and why is that a problem?"

</details>

---

## Prompt 12 -- Transformer Architecture (Advanced, Scenario-Based)

**Module:** 5 -- Transformers

> Your company wants to build a text classification system and a text generation system. A colleague suggests using BERT for both. Explain the full Transformer architecture at a high level -- positional encoding, multi-head self-attention, feed-forward network, residual connections, layer normalization -- and then explain why BERT is the right choice for classification but the wrong choice for generation. What would you use for generation instead, and why?

<details>
<summary>Interviewer Guide</summary>

**Target Keywords:** positional encoding, sinusoidal, learned embeddings, multi-head attention, attention heads, feed-forward network, residual connection, layer normalization, encoder-only, decoder-only, causal mask, BERT, GPT, bidirectional, autoregressive, MLM (masked language modeling), CLM (causal language modeling)

**Expected Good Answer:**
Transformer architecture: Token embeddings are added to positional encodings (sinusoidal or learned) to inject order information since attention is permutation-invariant. Each layer contains: (1) multi-head self-attention -- runs multiple attention heads in parallel, each learning different relationship types, then concatenates and projects; (2) add & norm -- residual connection plus layer normalization for gradient flow and training stability; (3) feed-forward network -- a two-layer MLP (typically 4x expansion) applied per position; (4) another add & norm. This block repeats N times.

BERT uses only the encoder with bidirectional attention (every token sees every other token). Pre-trained with masked language modeling. Ideal for understanding/classification because the model can attend to context on both sides. Not suitable for generation because it has no autoregressive decoding mechanism.

GPT uses only the decoder with causal masking (each token can only attend to previous tokens). Pre-trained with causal language modeling. Natural fit for text generation since it produces tokens left-to-right. The candidate might also mention T5 (encoder-decoder) for seq2seq tasks like summarization and translation.

**Red Flags:**
- Cannot explain why positional encoding is necessary (does not know attention is permutation-invariant)
- Confuses encoder-only (BERT) with decoder-only (GPT) architectures
- Thinks BERT can generate text
- Does not know the difference between bidirectional and causal attention
- Cannot describe what multi-head attention accomplishes vs. single-head

**Follow-up Prompts:**
- "BERT has a max sequence length of 512 tokens. Why? What architectural component imposes that limit?"
- "When would you choose T5 over both BERT and GPT?"

</details>

---

## Prompt 13 -- Transfer Learning and Fine-tuning (Advanced, Scenario-Based)

**Module:** 5 -- Transformers

> Your team has 3,000 labeled customer support tickets that need to be classified into 5 categories. You have a single GPU and a one-week deadline. Walk me through the complete transfer learning pipeline you would use, from choosing a pre-trained model to deploying a fine-tuned classifier. What are the critical hyperparameter decisions, and what mistakes could waste your week?

<details>
<summary>Interviewer Guide</summary>

**Target Keywords:** transfer learning, fine-tuning, pre-trained model, pre-trained model with classification head, tokenizer matched to pre-trained model, tokenization, training configuration (learning rate, epochs, batch size), training loop / training utility, learning rate (2e-5), epochs (2-5), freezing layers, catastrophic forgetting, feature extraction, classification head, model serialization / saving, data efficiency

**Expected Good Answer:**
Pipeline: (1) Choose a pre-trained model -- a BERT-base model is a solid default for classification with moderate data; (2) Load a pre-trained model with a classification head (adds a randomly initialized classification head on top of pre-trained weights) and a tokenizer matched to that pre-trained model; (3) Tokenize dataset with padding, truncation, and a reasonable max length; (4) Configure training -- learning rate is the most critical hyperparameter, typically 1e-5 to 5e-5 (larger values cause catastrophic forgetting by destroying pre-trained representations); 2-5 epochs is typical since you are adapting, not training from scratch; (5) Train using a training loop / training utility with evaluation after each epoch and checkpoint selection based on best validation metric; (6) Save and deploy via model serialization / saving. With 3,000 examples, full fine-tuning should work; with fewer (hundreds), consider freezing the pre-trained body and training only the head. Mistakes: learning rate too high (catastrophic forgetting), too many epochs (overfitting), not using evaluation during training, not setting max length appropriately.

**Red Flags:**
- Suggests training a Transformer from scratch with 3,000 examples
- Does not know the typical fine-tuning learning rate range
- Cannot explain catastrophic forgetting
- Confuses feature extraction (frozen body) with full fine-tuning
- Does not mention the tokenizer or assumes raw text goes directly into the model

**Follow-up Prompts:**
- "With only 200 labeled examples instead of 3,000, how would your approach change?"
- "Your fine-tuned model gets 92% accuracy on the validation set but only 78% on new production data. What might be happening?"

</details>

---

## Prompt 14 -- End-to-End Model Selection (Advanced, Scenario-Based)

**Module:** All (Synthesis)

> I am going to describe three different projects. For each one, tell me what type of model architecture you would use, why, and what evaluation strategy you would employ.
>
> **Project A:** A bank wants to predict which loan applicants will default. They have 50,000 rows of tabular data with 40 features and need the model to be explainable to regulators.
>
> **Project B:** A media company wants to automatically generate 2-sentence summaries of news articles that are 500-1,000 words long.
>
> **Project C:** A hospital has 10,000 chest X-ray images and wants to classify them as normal or abnormal.

<details>
<summary>Interviewer Guide</summary>

**Target Keywords (across all three):**

- Project A: tabular data, logistic regression baseline, tree-based ensemble, L1/L2 regularization, SHAP values, interpretability, AUC-ROC, precision/recall, stratification, algorithm selection framework
- Project B: encoder-decoder, T5, seq2seq, text generation, BERT inappropriate, summarization pipeline, ROUGE metrics
- Project C: CNN, convolutional neural network, transfer learning, data augmentation, binary classification, AUC-ROC, sensitivity/specificity, pre-trained ImageNet weights

**Expected Good Answer:**

**Project A:** Tabular data with interpretability requirements points toward logistic regression (with L1 or L2 regularization) as a strong baseline, possibly followed by a gradient-boosted tree model. Use the algorithm selection framework: problem type is supervised binary classification, 50K rows is moderate, 40 features need careful handling. Evaluation: AUC-ROC and precision/recall (not just accuracy, especially if defaults are rare). SHAP values for regulatory explainability -- global feature importance plus local explanations for individual decisions. Stratified splits for class imbalance.

**Project B:** This is a sequence-to-sequence task where input and output are different lengths. T5 (encoder-decoder) is the natural choice because it was designed for text-to-text tasks including summarization. BERT is wrong (encoder-only, cannot generate). GPT could work but T5's encoder-decoder split naturally separates comprehension from generation. Fine-tune a T5-base or Flan-T5-base model using a transfer learning pipeline. Evaluate with ROUGE metrics and human evaluation.

**Project C:** Images with spatial structure require a CNN. With only 10,000 images, transfer learning from a model pre-trained on ImageNet is essential to avoid overfitting. Data augmentation (flips, rotations, color jitter) on the training set only. Evaluation: AUC-ROC, sensitivity (recall for abnormal class), specificity, and a confusion matrix to understand error types (false negatives are especially costly in medical imaging).

**Red Flags:**
- Recommends a neural network for the tabular problem without considering simpler models first
- Suggests BERT for summarization or text generation
- Does not mention transfer learning or data augmentation for the image task with limited data
- Uses accuracy as the sole metric for any of these problems
- Cannot distinguish when to use CNN vs. RNN vs. Transformer
- Does not consider explainability requirements for the banking use case

**Follow-up Prompts:**
- "For Project A, if the gradient-boosted model beats logistic regression by 2 points of AUC but the regulator requires full model transparency, what do you recommend?"
- "For Project C, why would you not use a Transformer-based vision model with only 10,000 images?"

</details>

---

## Scoring Rubric

| Competency Level | Description |
|---|---|
| **Strong Pass** | Answers 10+ prompts well. Demonstrates clear mental models for all 5 modules. Uses correct terminology. Connects theory to practice (references assignments, provides code-level details). Identifies tradeoffs without prompting. |
| **Pass** | Answers 7-9 prompts well. Solid understanding of Foundations and Evaluation. Reasonable grasp of Neural Networks and Deep Learning. May need follow-ups to reach full depth on Transformers topics. |
| **Borderline** | Answers 5-6 prompts well. Understands supervised vs. unsupervised and basic metrics but struggles with architecture distinctions (CNN vs. RNN vs. Transformer) or cannot articulate regularization and optimization concepts clearly. |
| **Does Not Pass** | Answers fewer than 5 prompts adequately. Cannot distinguish between learning paradigms, does not understand data leakage, or confuses model architectures. Cannot connect any curriculum concept to a practical scenario. |

---

## Notes for Interviewers

- Start with Prompts 1-2 to establish baseline comfort. If the candidate struggles, spend more time on Modules 1-2 and use fewer advanced prompts.
- If the candidate is strong, skip some Beginner prompts and focus on Advanced scenarios (Prompts 10-14).
- The synthesis prompt (Prompt 14) is the single most revealing question -- prioritize it if time is limited.
- Watch for candidates who recite definitions without understanding. The scenario framing is designed to test application, not memorization.
- Follow-up prompts are optional -- use them to probe depth or clarify ambiguous answers.
- Reference the specific assignments (TechPulse Analytics for Module 1, Hospital Readmission for Module 2, Digit Classification for Module 3) to see if the candidate can connect lecture material to hands-on work.
