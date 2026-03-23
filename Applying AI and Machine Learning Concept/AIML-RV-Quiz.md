# Applying AI and Machine Learning Concept Review Quiz

**Activity Name:** AIML-RV-Quiz
**Display Name:** Applying AI and Machine Learning Concept Review Quiz
**Duration:** 30 min
**Total Questions:** 28
**Question Types:** Multiple Choice (20), True/False (5), Matching (3)
**Difficulty Distribution:** Beginner (9), Intermediate (11), Advanced (8)

---

## Questions

---

### Question 1 -- MCQ | Module 1 | Beginner | Conceptual

**Which stage of the ML lifecycle is primarily responsible for transforming raw fields into model-ready inputs, such as scaling numeric columns or encoding categorical variables?**

A) Feature engineering and preprocessing
B) Exploratory Data Analysis (EDA)
C) Model selection and training
D) Deployment and monitoring

**Correct Answer: A**

**Rationale:** Feature engineering and preprocessing is the stage where domain knowledge is encoded into transformations that convert raw data into inputs the model can consume. EDA (B) is about profiling and understanding distributions, not transforming for model input. Model selection and training (C) comes after features are prepared. Deployment (D) is about integrating a trained model into production systems.

---

### Question 2 -- MCQ | Module 1 | Intermediate | Scenario-Based

**A data scientist runs the same training notebook twice on the same dataset but gets different evaluation metrics each time. She has not changed any code or data between runs. Which practice would most directly resolve this inconsistency?**

A) Switching from a Jupyter notebook to a `.py` script
B) Adding more data to the training set
C) Using stratified k-fold cross-validation
D) Setting explicit random seeds for Python, NumPy, and scikit-learn's `random_state`

**Correct Answer: D**

**Rationale:** The inconsistency is caused by uncontrolled randomness in operations like train/test splitting, weight initialization, and shuffling. Setting explicit random seeds (D) fixes the pseudo-random sequences so results are reproducible. Switching to `.py` (A) improves maintainability but does not control randomness. More data (B) may reduce variance but will not make runs identical. Cross-validation (C) gives more stable estimates but each individual run would still differ without seeds.

---

### Question 3 -- MCQ | Module 1 | Beginner | Scenario-Based

**A product manager asks you to build a model that identifies whether customer support tickets should be routed to billing, technical support, or general inquiries. What type of ML task is this?**

A) Regression
B) Unsupervised clustering
C) Supervised classification
D) Reinforcement learning

**Correct Answer: C**

**Rationale:** The task has predefined categories (billing, technical, general) and labeled examples can be obtained from historical ticket routing. This makes it a supervised classification problem. Regression (A) predicts continuous values. Clustering (B) discovers groups without labels. Reinforcement learning (D) involves an agent interacting with an environment and receiving rewards.

---

### Question 4 -- MCQ | Module 1 | Intermediate | Code-Reading

**Consider the following code:**

```python
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()
clf.fit(X_train, y_train)
result = clf.predict_proba(X_test)[:, 1]
```

**What does the variable `result` contain?**

A) The predicted class labels for each test sample
B) The probability of each test sample belonging to the positive class
C) The raw logit scores before sigmoid transformation
D) The coefficients (weights) learned by the model

**Correct Answer: B**

**Rationale:** `predict_proba` returns probability estimates for each class. Indexing with `[:, 1]` selects the second column, which is the probability of the positive class in a binary setting. `predict` (not `predict_proba`) would return class labels (A). Scikit-learn does not directly expose raw logits through this API (C). Coefficients are accessed via `clf.coef_` (D).

---

### Question 5 -- MCQ | Module 1 | Advanced | Scenario-Based

**You are working on a tabular dataset with 500 rows, 200 features (many highly correlated), and a binary classification target. Your stakeholders require interpretable feature importance. Following the algorithm selection framework, which baseline model is the strongest starting point?**

A) A deep MLP with 6 hidden layers
B) K-Means clustering followed by logistic regression on cluster assignments
C) A Random Forest with 1000 estimators
D) Logistic regression with L1 regularization (Lasso)

**Correct Answer: D**

**Rationale:** With 500 rows, 200 correlated features, and a need for interpretability, logistic regression with L1 regularization is ideal: L1 drives irrelevant coefficients to zero (implicit feature selection), it is interpretable, and it handles correlated features better than unregularized models. A deep MLP (A) is overkill for 500 rows and is not interpretable. K-Means (B) is unsupervised and does not directly address the supervised classification task. Random Forest (C) is less interpretable and risks overfitting on only 500 samples with 200 features.

---

### Question 6 -- True/False | Module 1 | Beginner | Conceptual

**In unsupervised learning, the `.fit()` method in scikit-learn requires both feature matrix `X` and label vector `y` as arguments.**

**Correct Answer: False**

**Rationale:** Unsupervised learning algorithms such as KMeans operate without labels. The `.fit()` method only requires the feature matrix `X`. For example, `KMeans(n_clusters=3).fit(X)` does not use a `y` argument. Supervised algorithms are the ones that require both `X` and `y`.

---

### Question 7 -- MCQ | Module 2 | Beginner | Conceptual

**What is the primary purpose of keeping a held-out test set completely separate from model development?**

A) To provide an unbiased estimate of how the model will perform on unseen data
B) To speed up the training process
C) To increase the total amount of training data available
D) To ensure the model achieves 100% accuracy

**Correct Answer: A**

**Rationale:** The test set serves as a final, unbiased check on generalization. If the test set is used during model selection or hyperparameter tuning, it becomes a validation set and can no longer provide a fair estimate of future performance. It does not speed up training (B), it reduces available training data (C), and no split guarantees perfect accuracy (D).

---

### Question 8 -- MCQ | Module 2 | Intermediate | Scenario-Based

**A model achieves 0.02 MSE on the training set but 0.45 MSE on the validation set. What does this gap most likely indicate, and what is an appropriate first response?**

A) Underfitting; increase model complexity by adding more layers
B) Overfitting; apply regularization such as L2 (Ridge) or gather more training data
C) A data leakage problem; re-examine the feature pipeline
D) The loss function is mismatched; switch from MSE to cross-entropy

**Correct Answer: B**

**Rationale:** A large gap between low training error and high validation error is the classic signature of overfitting -- the model memorized training data rather than learning generalizable patterns. Regularization (L2/Ridge, L1/Lasso, or Dropout for neural networks) penalizes complexity and is a direct response. Underfitting (A) would show high error on both sets. Data leakage (C) typically shows unrealistically good validation performance. The task uses continuous targets (MSE), so cross-entropy (D) would be inappropriate.

---

### Question 9 -- MCQ | Module 2 | Advanced | Code-Reading

**Given this code snippet:**

```python
from sklearn.linear_model import Ridge, Lasso

ridge = Ridge(alpha=1.0).fit(X_train, y_train)
lasso = Lasso(alpha=0.01, max_iter=5000).fit(X_train, y_train)

print((lasso.coef_ != 0).sum())
```

**What does the printed value represent, and why might it differ significantly from the total number of features?**

A) The number of training samples used; Ridge uses fewer samples than Lasso
B) The total regularization penalty applied to the model
C) The number of iterations needed for convergence; Lasso typically needs more iterations than Ridge
D) The count of non-zero coefficients after L1 regularization, which drives some weights exactly to zero for feature selection

**Correct Answer: D**

**Rationale:** L1 regularization (Lasso) adds a penalty proportional to the absolute value of coefficients, which can drive some coefficients exactly to zero. The expression `(lasso.coef_ != 0).sum()` counts how many features retained non-zero weights, effectively showing which features the model considers useful. This count is often smaller than the total feature count because irrelevant features get zeroed out. Ridge (L2) shrinks weights toward zero but rarely sets them exactly to zero.

---

### Question 10 -- MCQ | Module 2 | Intermediate | Scenario-Based

**You are building a fraud detection model where only 0.5% of transactions are fraudulent. Your model achieves 99.5% accuracy. A colleague says the model is excellent. What is the most important concern?**

A) Accuracy is misleading here; the model could be predicting "not fraud" for every transaction and still achieving 99.5%
B) The model is underfitting because accuracy should be closer to 100%
C) The model needs more training epochs to improve beyond 99.5%
D) The test set is too large relative to the training set

**Correct Answer: A**

**Rationale:** With 99.5% of transactions being legitimate, a trivial classifier that always predicts "not fraud" achieves 99.5% accuracy while catching zero fraud. Precision, recall, and F1 on the fraud class are the meaningful metrics here. The model may have learned nothing useful despite the impressive-sounding accuracy number. This is the core lesson of why accuracy alone is insufficient for imbalanced datasets.

---

### Question 11 -- MCQ | Module 2 | Advanced | Scenario-Based

**A medical screening model needs to ensure that very few actual disease cases are missed, even if it means more false alarms. Which metric should be the primary optimization target?**

A) Precision
B) Recall
C) AUC-ROC
D) Mean Squared Error

**Correct Answer: B**

**Rationale:** Recall measures the fraction of actual positives that the model correctly identifies. When the cost of missing a true positive (false negative) is very high -- as in medical screening -- recall should be prioritized. High precision (A) minimizes false alarms but may miss real cases. AUC-ROC (C) summarizes ranking quality across thresholds but does not directly optimize for catching all positives. MSE (D) is a regression metric.

---

### Question 12 -- True/False | Module 2 | Intermediate | Conceptual

**SHAP values explain model predictions by attributing each feature's contribution to the difference between the model's output and a baseline (average) prediction, using principles from cooperative game theory.**

**Correct Answer: True**

**Rationale:** SHAP (SHapley Additive exPlanations) is rooted in Shapley values from game theory. For each prediction, SHAP calculates how much each feature contributes to the deviation from the baseline prediction, considering all possible orderings of features. This provides consistent, locally accurate explanations of individual predictions.

---

### Question 13 -- MCQ | Module 2 | Intermediate | Conceptual

**Which statement best describes the relationship between an ROC curve and a confusion matrix?**

A) They measure exactly the same thing and are interchangeable
B) The ROC curve is only valid for regression tasks, while the confusion matrix is for classification
C) The confusion matrix requires probability outputs, while the ROC curve uses only predicted labels
D) The confusion matrix shows errors at one specific threshold, while the ROC curve shows the trade-off between true positive rate and false positive rate across all thresholds

**Correct Answer: D**

**Rationale:** A confusion matrix captures the four outcomes (TP, FP, TN, FN) at a single chosen decision threshold. The ROC curve sweeps across all possible thresholds and plots TPR versus FPR, showing the full frontier of trade-offs. They are complementary: ROC helps choose a threshold, and the confusion matrix shows what happens at that threshold. Both are classification tools (B is wrong), and the relationship is reversed from what C states.

---

### Question 14 -- MCQ | Module 3 | Beginner | Conceptual

**Why do neural networks require nonlinear activation functions in their hidden layers?**

A) To speed up the forward pass computation
B) To reduce the number of parameters in the model
C) Because without nonlinearity, stacking multiple linear layers still produces only a linear transformation
D) To ensure that all outputs are positive numbers

**Correct Answer: C**

**Rationale:** A composition of linear transformations is still linear. Without nonlinear activation functions, no matter how many layers are stacked, the network can only represent linear mappings from input to output. Activation functions like ReLU introduce nonlinearity, enabling the network to learn complex, curved decision boundaries. They do not primarily affect speed (A) or parameter count (B), and not all activations restrict outputs to positive values (D) -- tanh outputs range from -1 to 1.

---

### Question 15 -- MCQ | Module 3 | Advanced | Conceptual

**A standard neural network training loop repeats several ordered steps for each batch: forward pass, loss computation, backpropagation, weight update, and gradient reset. What would happen if the gradient reset step were removed entirely?**

A) Gradients from previous batches would accumulate, causing weight updates that no longer reflect the current batch's loss and leading to training instability
B) The training framework would automatically adjust the learning rate to compensate for extra gradient information
C) Training would proceed normally because gradients are always overwritten during backpropagation, not accumulated
D) The loss function would become undefined and the model could not produce predictions

**Correct Answer: A**

**Rationale:** In the standard training loop (forward pass → loss computation → backpropagation → weight update → gradient reset), failing to reset gradients causes them to accumulate across batches. Each backpropagation step adds new gradients to the existing values rather than replacing them. The resulting weight updates reflect a noisy mixture of current and past batches, producing increasingly incorrect updates and training instability. Gradients are not automatically overwritten (C), the learning rate is not automatically adjusted (B), and the loss function remains valid regardless of gradient management (D).

---

### Question 16 -- MCQ | Module 3 | Intermediate | Scenario-Based

**You are training a deep MLP (8 hidden layers) with sigmoid activations and notice that the first few layers' weights barely change during training, while later layers learn normally. What is the most likely cause?**

A) The learning rate is too high
B) The vanishing gradient problem caused by sigmoid's small derivatives in deep networks
C) The batch size is too large
D) The model needs more output neurons

**Correct Answer: B**

**Rationale:** Sigmoid activations squash inputs to (0, 1) and have maximum derivatives of 0.25. When gradients are multiplied through 8 layers during backpropagation, they shrink exponentially -- this is the vanishing gradient problem. Early layers receive near-zero gradients and barely update. Replacing sigmoid with ReLU (derivative of 1 for positive inputs) in hidden layers is the standard fix. A high learning rate (A) would cause instability, not frozen layers. Batch size (C) and output neuron count (D) are unrelated to this gradient flow issue.

---

### Question 17 -- MCQ | Module 3 | Intermediate | Conceptual

**Which statement correctly distinguishes SGD with momentum from the Adam optimizer?**

A) SGD with momentum adapts the learning rate per parameter; Adam uses a fixed learning rate for all parameters
B) Adam maintains running averages of both gradients and squared gradients, effectively giving each parameter its own learning rate, while SGD with momentum only tracks a running average of gradients
C) SGD with momentum converges faster than Adam in all cases
D) Adam requires no learning rate hyperparameter

**Correct Answer: B**

**Rationale:** Adam combines momentum (first moment -- running average of gradients) with adaptive learning rates (second moment -- running average of squared gradients). This gives each parameter an effective learning rate scaled by its gradient history. SGD with momentum only maintains the first moment (velocity). The relationship in (A) is reversed. Adam does not universally converge slower (C). Adam still requires a learning rate hyperparameter, typically 0.001 (D).

---

### Question 18 -- True/False | Module 3 | Beginner | Conceptual

**The Universal Approximation Theorem states that a single hidden layer with enough neurons can approximate any continuous function, meaning deeper networks offer no practical advantage over wider ones.**

**Correct Answer: False**

**Rationale:** While the Universal Approximation Theorem does state that a single hidden layer can theoretically approximate any continuous function given sufficient width, the key qualifier is "enough." In practice, the required width may be impractically large. Deep networks can often represent the same functions with exponentially fewer total parameters by building hierarchical, compositional features. Depth also provides practical advantages in learning efficiency and generalization.

---

### Question 19 -- MCQ | Module 3 | Advanced | Scenario-Based

**You are designing an MLP for a 10-class classification task on tabular data with 50 features and 2,000 samples. The model has 4 hidden layers of 512 neurons each. Training loss is near zero but validation accuracy is only 55%. What is the best first action?**

A) Add two more hidden layers to increase model capacity
B) Switch from ReLU to sigmoid activation in hidden layers
C) Reduce the network's capacity by narrowing layers or removing layers, and consider adding dropout regularization
D) Increase the number of training epochs from 50 to 500

**Correct Answer: C**

**Rationale:** Near-zero training loss with poor validation accuracy is textbook overfitting. The model has far too much capacity for only 2,000 samples (4 layers of 512 neurons with 50 input features). Reducing width, depth, or both limits the model's ability to memorize training noise. Adding dropout further regularizes by randomly disabling neurons during training. Adding more layers (A) worsens the problem. Sigmoid (B) introduces vanishing gradients without helping generalization. More epochs (D) would deepen overfitting.

---

### Question 20 -- MCQ | Module 4 | Beginner | Conceptual

**What is the primary advantage of convolutional layers over fully connected layers for image data?**

A) Convolutional layers exploit spatial structure through local connectivity and weight sharing, drastically reducing parameter count
B) Convolutional layers use more parameters, giving them greater capacity
C) Convolutional layers can only process grayscale images
D) Convolutional layers eliminate the need for activation functions

**Correct Answer: A**

**Rationale:** Fully connected layers treat every pixel as independent, creating an enormous number of parameters (e.g., 25+ million for just the first layer on a 128x128 RGB image). Convolutional layers use small filters that slide across the image, sharing weights at every position. This drastically reduces parameters while preserving spatial relationships. They handle color images (C is wrong), still use activations like ReLU (D is wrong), and use fewer, not more, parameters (B is wrong).

---

### Question 21 -- MCQ | Module 4 | Intermediate | Scenario-Based

**You are building a model to predict tomorrow's stock price given the past 60 days of price data. Which architecture is most appropriate for this sequential prediction task?**

A) A CNN with 2D convolutions
B) A fully connected MLP
C) An LSTM or GRU network
D) K-Means clustering

**Correct Answer: C**

**Rationale:** Stock price prediction is a time-series task where temporal order and long-range dependencies matter. LSTMs and GRUs are designed specifically for sequential data, maintaining hidden states that carry information across time steps. A CNN with 2D convolutions (A) is designed for spatial data like images. An MLP (B) treats each input independently with no temporal awareness. K-Means (D) is an unsupervised clustering method, not a prediction model.

---

### Question 22 -- MCQ | Module 4 | Advanced | Conceptual

**An LSTM layer processes a batch of 4 sequences, each containing 20 time steps, with an input feature dimension of 10 and a hidden size of 64. What are the dimensions of the LSTM's output (hidden states at all time steps) and the final hidden state, and what role does the cell state play?**

A) Output is (4, 64), final hidden state is (4, 64), and the cell state stores the learning rate schedule
B) Output is (20, 4, 10), final hidden state is (4, 10), and the cell state records the confusion matrix
C) Output is (4, 20, 10), final hidden state is (1, 4, 10), and the cell state accumulates raw gradients
D) Output is (4, 20, 64), final hidden state is (1, 4, 64), and the cell state provides long-term memory through additive updates controlled by forget and input gates

**Correct Answer: D**

**Rationale:** The LSTM produces a hidden state vector of size 64 (the hidden size, not the input size) at each of the 20 time steps, for each of the 4 sequences, giving an output shape of (4, 20, 64). The final hidden state has an additional leading dimension for the number of layers, giving (1, 4, 64). The cell state is the LSTM's key architectural innovation: it maintains long-term memory through additive updates (not multiplicative), controlled by the forget gate (what to discard) and input gate (what to store). This additive pathway alleviates the vanishing gradient problem that limits vanilla RNNs.

---

### Question 23 -- MCQ | Module 4 | Intermediate | Scenario-Based

**You are building a machine translation system that converts English sentences into French sentences of varying lengths. Which architecture is most appropriate?**

A) A single LSTM that classifies each input token independently
B) An encoder-decoder (seq2seq) architecture where the encoder reads the source sentence and the decoder generates the target sentence
C) A CNN that processes the English sentence as an image
D) A standard MLP with input size equal to the maximum possible sentence length

**Correct Answer: B**

**Rationale:** Machine translation maps variable-length input sequences to variable-length output sequences with no simple one-to-one alignment between tokens. Encoder-decoder (seq2seq) architectures were specifically designed for this: the encoder compresses the source into a context vector, and the decoder generates the target one token at a time. Independent token classification (A) ignores word order and context in the output. A CNN (C) does not handle variable-length text generation. A fixed-size MLP (D) cannot produce variable-length outputs.

---

### Question 24 -- True/False | Module 4 | Intermediate | Conceptual

**A GRU uses three gates (forget, input, and output) to control information flow, while an LSTM uses only two gates (reset and update).**

**Correct Answer: False**

**Rationale:** This statement has the architectures reversed. The LSTM uses three gates: forget, input, and output, plus a separate cell state. The GRU simplifies this to two gates: reset and update, with no separate cell state. The GRU merges the cell state and hidden state into a single vector, making it more parameter-efficient but sometimes less capable on very long sequences.

---

### Question 25 -- MCQ | Module 5 | Intermediate | Conceptual

**In scaled dot-product attention, why is the dot product divided by the square root of the key dimension (sqrt(d_k))?**

A) To ensure the output vectors have unit length
B) To convert the scores into probabilities
C) To prevent large dot-product magnitudes from pushing softmax into regions with extremely small gradients, which would hinder learning
D) To reduce the computational cost of matrix multiplication

**Correct Answer: C**

**Rationale:** As the key dimension d_k grows, the variance of dot products grows proportionally. Large-magnitude scores cause softmax to produce near-one-hot distributions, where gradients are nearly zero for all but one position. Dividing by sqrt(d_k) normalizes the variance back to approximately 1, keeping softmax in a region with healthy gradients. This does not normalize vector length (A), softmax handles probability conversion (B), and the division has negligible computational cost (D).

---

### Question 26 -- MCQ | Module 5 | Advanced | Scenario-Based

**You need to build a production text classification service for customer feedback (positive/negative) and have 5,000 labeled examples. Which approach is most appropriate?**

A) Fine-tune a pre-trained Transformer encoder (such as BERT) by adding a classification head and training with a small learning rate for a few epochs to preserve pre-trained representations
B) Train a large Transformer language model from scratch on your 5,000 examples
C) Use a vanilla RNN with randomly initialized weights
D) Apply K-Means clustering to TF-IDF vectors of the feedback text

**Correct Answer: A**

**Rationale:** Transfer learning with a pre-trained Transformer encoder is ideal here. The pre-trained model has already learned rich language representations from large corpora. Fine-tuning with a small learning rate preserves these representations while adapting the model to the specific classification task, and 5,000 examples is sufficient for effective fine-tuning. Training a Transformer from scratch (B) requires orders of magnitude more data to learn useful representations. A vanilla RNN (C) lacks pre-trained knowledge and suffers from vanishing gradients on longer sequences. K-Means on TF-IDF (D) is unsupervised and ignores the labeled data entirely.

---

### Question 27 -- True/False | Module 5 | Advanced | Conceptual

**GPT models use bidirectional self-attention, allowing each token to attend to all other tokens in the sequence during both training and generation.**

**Correct Answer: False**

**Rationale:** GPT uses causal (left-to-right) masking in its self-attention, meaning each token can only attend to tokens at its own position and earlier positions. This is what makes GPT suitable for autoregressive text generation: it never "sees" future tokens. BERT, by contrast, uses bidirectional attention where each token attends to all positions. This architectural difference is why BERT excels at understanding tasks and GPT excels at generation tasks.

---

### Question 28 -- MCQ | Module 5 | Intermediate | Scenario-Based

**Your team needs to build an internal document summarization pipeline. The input documents are 1-3 pages long, and the summaries should be 2-3 sentences. Which model architecture and specific model are the best fit?**

A) BERT (encoder-only), because it understands the full document context
B) GPT-2 (decoder-only), because it generates text autoregressively
C) T5 (encoder-decoder), because summarization requires reading the full input and generating a structurally different, shorter output
D) A vanilla RNN, because documents are sequential data

**Correct Answer: C**

**Rationale:** Summarization is a sequence-to-sequence task: the input (full document) and output (short summary) are structurally different and have different lengths. T5's encoder-decoder architecture naturally separates comprehension (encoder reads the document) from generation (decoder produces the summary). BERT (A) cannot generate text. GPT-2 (B) can generate text but lacks a separate encoder for comprehensive input reading. A vanilla RNN (D) suffers from vanishing gradients and lacks pre-trained knowledge.

---

## Matching Sections

---

### Question 29 -- Matching | Modules 1-2 | Beginner | Conceptual

**Match each evaluation concept (left) with its correct definition (right).**

| # | Concept | | Definition |
|---|---------|---|------------|
| 1 | Precision | | A. The fraction of actual positives correctly identified by the model |
| 2 | Recall | | B. A penalty on model complexity to reduce overfitting |
| 3 | Regularization | | C. Of all predictions labeled positive, the fraction that are truly positive |
| 4 | Stratified Split | | D. A splitting strategy that preserves class proportions in each subset |

**Correct Matches:**

- 1 --> C
- 2 --> A
- 3 --> B
- 4 --> D

**Rationale:**
- **Precision** (1-C): Precision answers "of everything we predicted as positive, how many were actually positive?" -- it measures the quality of positive predictions.
- **Recall** (2-A): Recall answers "of all actual positives, how many did we catch?" -- it measures the completeness of positive detection.
- **Regularization** (3-B): Regularization techniques (L1, L2, Dropout) add a penalty for model complexity to prevent overfitting to training noise.
- **Stratified Split** (4-D): Stratified splitting ensures each split maintains the same class distribution as the full dataset, which is critical when classes are imbalanced.

---

### Question 30 -- Matching | Module 3 | Intermediate | Conceptual

**Match each activation function (left) with its most appropriate use case (right).**

| # | Activation Function | | Best Use Case |
|---|---------------------|---|---------------|
| 1 | ReLU | | A. Output layer for multi-class classification |
| 2 | Sigmoid | | B. Hidden layers in most feedforward and convolutional networks |
| 3 | Softmax | | C. Output layer for binary classification (probability output) |
| 4 | GELU | | D. Hidden layers in Transformer-based architectures |

**Correct Matches:**

- 1 --> B
- 2 --> C
- 3 --> A
- 4 --> D

**Rationale:**
- **ReLU** (1-B): ReLU (max(0, z)) is the default activation for hidden layers in feedforward and convolutional networks due to its computational efficiency and resistance to vanishing gradients.
- **Sigmoid** (2-C): Sigmoid maps values to (0, 1) and is used at the output layer for binary classification to produce a probability estimate.
- **Softmax** (3-A): Softmax converts a vector of logits into a probability distribution over multiple classes, making it the standard choice for multi-class output layers.
- **GELU** (4-D): GELU provides a smooth approximation of ReLU and is the activation function used in Transformer architectures like BERT and GPT.

---

### Question 31 -- Matching | Modules 4-5 | Advanced | Conceptual

**Match each architecture (left) with the problem it was specifically designed to solve (right).**

| # | Architecture | | Designed For |
|---|-------------|---|-------------|
| 1 | CNN | | A. Compressing an input sequence into a fixed-size context vector and generating a variable-length output sequence |
| 2 | LSTM | | B. Processing spatial data (images) using local filters and weight sharing |
| 3 | Encoder-Decoder (Seq2Seq) | | C. Eliminating recurrence entirely, using self-attention and positional encoding for parallelizable sequence processing |
| 4 | Transformer | | D. Capturing long-range dependencies in sequential data through gated memory cells |

**Correct Matches:**

- 1 --> B
- 2 --> D
- 3 --> A
- 4 --> C

**Rationale:**
- **CNN** (1-B): CNNs exploit spatial structure through convolutional filters that slide across images, detecting local patterns like edges and textures with shared weights.
- **LSTM** (2-D): LSTMs use forget, input, and output gates with a separate cell state to maintain long-range memory in sequential data, solving the vanishing gradient problem that limits vanilla RNNs.
- **Encoder-Decoder** (3-A): Seq2seq models use an encoder to compress the input and a decoder to generate variable-length outputs, designed for tasks like translation where input and output lengths differ.
- **Transformer** (4-C): Transformers replaced recurrence with self-attention and positional encodings, enabling fully parallelized training and superior scaling to long sequences.

---

## Answer Key Summary

| Q# | Type | Module | Difficulty | Answer |
|----|------|--------|------------|--------|
| 1 | MCQ | 1 | Beginner | A |
| 2 | MCQ | 1 | Intermediate | D |
| 3 | MCQ | 1 | Beginner | C |
| 4 | MCQ | 1 | Intermediate | B |
| 5 | MCQ | 1 | Advanced | D |
| 6 | T/F | 1 | Beginner | False |
| 7 | MCQ | 2 | Beginner | A |
| 8 | MCQ | 2 | Intermediate | B |
| 9 | MCQ | 2 | Advanced | D |
| 10 | MCQ | 2 | Intermediate | A |
| 11 | MCQ | 2 | Advanced | B |
| 12 | T/F | 2 | Intermediate | True |
| 13 | MCQ | 2 | Intermediate | D |
| 14 | MCQ | 3 | Beginner | C |
| 15 | MCQ | 3 | Advanced | A |
| 16 | MCQ | 3 | Intermediate | B |
| 17 | MCQ | 3 | Intermediate | B |
| 18 | T/F | 3 | Beginner | False |
| 19 | MCQ | 3 | Advanced | C |
| 20 | MCQ | 4 | Beginner | A |
| 21 | MCQ | 4 | Intermediate | C |
| 22 | MCQ | 4 | Advanced | D |
| 23 | MCQ | 4 | Intermediate | B |
| 24 | T/F | 4 | Intermediate | False |
| 25 | MCQ | 5 | Intermediate | C |
| 26 | MCQ | 5 | Advanced | A |
| 27 | T/F | 5 | Advanced | False |
| 28 | MCQ | 5 | Intermediate | C |
| 29 | Match | 1-2 | Beginner | 1-C, 2-A, 3-B, 4-D |
| 30 | Match | 3 | Intermediate | 1-B, 2-C, 3-A, 4-D |
| 31 | Match | 4-5 | Advanced | 1-B, 2-D, 3-A, 4-C |
