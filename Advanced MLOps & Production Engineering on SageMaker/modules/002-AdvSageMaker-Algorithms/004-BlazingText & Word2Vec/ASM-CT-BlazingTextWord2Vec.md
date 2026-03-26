# BlazingText & Word2Vec

**Estimated Time:** 10 Minutes

## Introduction

The built-in algorithms covered so far -- XGBoost, K-Means, RCF -- operate on numeric tabular data. Many production ML tasks, however, involve text: classifying support tickets, categorizing product reviews, detecting spam, or extracting intent from user queries. SageMaker provides **BlazingText** as a built-in algorithm for two text-related tasks: **text classification** and **word embedding generation (Word2Vec)**.

BlazingText is Amazon's optimized implementation of Facebook's fastText library. It trains orders of magnitude faster than standard implementations by leveraging multi-core CPUs and optional GPU acceleration, making it practical for datasets with millions of text documents.

## Core Concepts

### Two modes of operation

BlazingText operates in two distinct modes, selected via the `mode` hyperparameter:

**Supervised mode (`mode=supervised`):** Multi-class or multi-label text classification. Given labeled text documents (each line is a label followed by the text), BlazingText trains a classifier that maps raw text to categories. Under the hood, it uses the fastText architecture: each word is represented as a bag of character n-grams, averaged into a document vector, and passed through a linear classifier.

**Unsupervised mode (`mode=batch_skipgram`, `mode=skipgram`, or `mode=cbow`):** Word embedding generation. Given a corpus of unlabeled text, BlazingText trains Word2Vec embeddings -- dense vector representations where semantically similar words have similar vectors. These embeddings can be used as features for downstream models.

### Text classification with BlazingText

**Input format:** One document per line. Each line starts with the label prefixed by `__label__`, followed by the text:

```
__label__positive This product exceeded my expectations
__label__negative Arrived damaged and customer service was unhelpful
__label__neutral Adequate for the price point
```

**Key hyperparameters for classification:**

| Parameter | Default | Purpose |
| :--- | :--- | :--- |
| `mode` | `supervised` | Enables classification mode |
| `epochs` | 5 | Number of passes over the training data |
| `learning_rate` | 0.05 | Step size for SGD optimization |
| `min_count` | 5 | Minimum word frequency to include in vocabulary |
| `word_ngrams` | 2 | Character n-gram length for subword features |
| `vector_dim` | 100 | Dimensionality of the internal word vectors |

**Console configuration:**

1. Navigate to **SageMaker > Training > Training jobs > Create training job**.
2. Select **BlazingText** from the built-in algorithm list.
3. Set `mode` to `supervised`.
4. Set the S3 path for training data (channel name: `train`) and optional validation data (channel name: `validation`).
5. Instance type: `ml.m5.xlarge` for CPU training. For datasets over 1M documents, consider `ml.p3.2xlarge` for GPU acceleration.

**Output:** A trained model artifact in S3 that can be deployed to an endpoint. The endpoint accepts raw text (one document per line) and returns predicted labels with confidence scores.

### Word2Vec embedding generation

**Input format:** One sentence per line, no labels:

```
the quick brown fox jumps over the lazy dog
machine learning models require feature engineering
```

**Key modes:**
- **`skipgram`:** Predicts context words given a target word. Produces higher-quality embeddings for rare words. Single-threaded.
- **`batch_skipgram`:** Multi-threaded skipgram for faster training on multi-core instances.
- **`cbow` (Continuous Bag of Words):** Predicts a target word given context words. Faster training than skipgram but slightly lower quality for rare words.

**Output:** A file containing word vectors (one word per line, followed by its vector components). These vectors can be loaded into downstream models as pre-trained embeddings or used for nearest-neighbor similarity search.

### When to use BlazingText vs. transformer models

BlazingText is fast and lightweight, but it has limitations compared to modern transformer-based NLP models (BERT, GPT):

| Factor | BlazingText | Transformers |
| :--- | :--- | :--- |
| Training speed | Minutes to hours | Hours to days |
| Context awareness | Bag-of-words (no word order) | Full sequential context |
| Accuracy on complex tasks | Good for straightforward classification | Superior for nuanced understanding |
| Infrastructure cost | `ml.m5.xlarge` sufficient | Often requires GPU instances |
| Use case fit | High-volume, simple classification (spam, category, intent) | Sentiment analysis, question answering, summarization |

For production systems that need fast, cost-effective text classification at scale (e.g., routing millions of support tickets daily), BlazingText is often the right choice. For tasks requiring deeper language understanding, transformer-based models deployed via SageMaker JumpStart or custom containers are more appropriate.

## Connecting to Practice

BlazingText adds NLP capabilities to your SageMaker toolkit. The next topic, *DeepAR Time Series*, covers the built-in algorithm for forecasting. In the module assignment, you will train a BlazingText classifier on a text classification dataset using the `supervised` mode and deploy it to an endpoint for real-time inference.

## Further Learning & Resources

**Documentation and reading**

- **[SageMaker BlazingText](https://docs.aws.amazon.com/sagemaker/latest/dg/blazingtext.html)** - *Docs*: Complete reference for both classification and Word2Vec modes, including input format specifications.
- **[fastText Documentation](https://fasttext.cc/)** - *Docs*: The open-source library that BlazingText is based on, with tutorials and pre-trained embeddings.

**Interactive practice**

- **[Text Classification with BlazingText](https://github.com/aws/amazon-sagemaker-examples/tree/main/introduction_to_amazon_algorithms/blazingtext_text_classification_dbpedia)** - *Interactive*: Sample notebook demonstrating multi-class text classification with BlazingText on the DBpedia dataset.
