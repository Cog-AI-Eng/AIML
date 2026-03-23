# Metrics: Precision, Recall, F1

**Estimated Time:** 10 Minutes

## Introduction

If you have ever seen a medical test that is "99% accurate" but still misses most sick patients, you have already met the limits of **accuracy** as a single number. In real projects -- fraud detection, disease screening, spam filtering -- classes are often **imbalanced**: one outcome is rare but costly to miss. Precision, recall, and F1 are three lenses that answer different questions about mistakes. They do not replace accuracy; they **refine** it so you can align the model with business risk.

## Core Concepts

**Precision** answers: *Of everything we predicted as positive, how many were actually positive?*
Think of a fishing net: precision is "how much of what we caught was the fish we wanted," not "how many fish exist in the ocean."

**Recall** answers: *Of all actual positives, how many did we catch?*
Same net analogy: recall is "what fraction of all the fish we cared about ended up in the net." A huge net might catch every fish (high recall) but also a lot of junk (low precision). A tiny, careful net might catch almost only good fish (high precision) but miss many (low recall).

**F1** is the **harmonic mean** of precision and recall. It punishes lopsided scores: you cannot hide a terrible recall behind great precision (or the reverse) as easily as with a simple average. Use F1 when you need one summary number and both false positives and false negatives matter -- but neither dominates the story alone.

**Class imbalance** means one class has far more examples than another. A naive model that always predicts "not fraud" can look 99% accurate while catching zero fraud. Precision, recall, and F1 force you to look at **errors on the minority class**, not just overall hit rate.

**When each metric matters**

- **Precision** matters when the **cost of a false positive** is high (e.g., flagging innocent users, unnecessary invasive follow-up).
- **Recall** matters when the **cost of a false negative** is high (e.g., missing fraud, missing a treatable condition).
- **F1** is a reasonable default when you need balance and a single score for comparison, but you should still read precision and recall separately for imbalanced problems.

**Trade-offs** usually come from changing the **decision threshold** (not only from changing the model). Moving the threshold trades precision for recall and vice versa. No single threshold is "correct" until you know the costs.

```python
from sklearn.metrics import precision_score, recall_score, f1_score

precision = precision_score(y_true, y_pred, average="binary")
recall = recall_score(y_true, y_pred, average="binary")
f1 = f1_score(y_true, y_pred, average="binary")
```

For multiple classes, `average` choices (`"macro"`, `"weighted"`, etc.) change *what question* you are answering -- whether each class counts equally or proportional to frequency.

## Connecting to Practice

On a project, start by writing down **which mistake is worse** and whether the dataset is imbalanced. Then report precision, recall, and F1 (with the same threshold and same split) so stakeholders can see the trade-off. If you only optimize accuracy on imbalanced data, you may ship a model that looks good on paper and fails in production. Pair these metrics with **confusion-matrix thinking** (which errors happen?) and, when you compare models, use the **same** evaluation protocol so the numbers are comparable.

---

## Further Learning & Resources

**Documentation**

- **[Scikit-learn: Classification metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics)** - *Docs*: Comprehensive guide to all classification metrics.
- **[Scikit-learn: Precision, recall, F-measure](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html)** - *Docs*: API reference for the combined metrics function.

**Interactive**

- **[MLU-Explain: Visual introduction to precision and recall](https://mlu-explain.github.io/precision-recall/)** - *Interactive*: Excellent visual explanation of the precision-recall tradeoff.
- **[Scikit-learn example gallery (model evaluation)](https://scikit-learn.org/stable/auto_examples/index.html#model-selection)** - *Interactive*: Runnable examples of metric computation and visualization.
