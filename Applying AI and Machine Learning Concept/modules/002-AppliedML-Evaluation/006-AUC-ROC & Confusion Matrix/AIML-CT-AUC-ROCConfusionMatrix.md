# AUC-ROC & Confusion Matrix

**Estimated Time:** 10 Minutes

## Introduction

A **confusion matrix** is a simple table that answers: *Where do predictions land relative to the truth?* An **ROC curve** and its **AUC** (area under the curve) answer a related but different question: *How well does the model rank positives above negatives across many possible operating points?* Together, they connect **what happened at one threshold** (confusion matrix) to **how the model behaves if you move that threshold** (ROC/AUC).

## Core Concepts

**Confusion matrix (binary case)**

Rows are often **actual** class; columns are **predicted** class (conventions vary -- always check the plot labels). The four cells are true negatives, false positives, false negatives, and true positives. From these you derive precision, recall, and error types you care about. The matrix is **not** a single score; it is a **picture of mistakes**.

**ROC curve**

The model usually outputs **scores or probabilities**, not just hard 0/1 labels. The ROC curve plots **true positive rate (recall)** versus **false positive rate** as you sweep the **classification threshold** from strict to loose. Intuitively: "If we become more willing to call something positive, how fast do we catch real positives versus how fast do we alarm on negatives?"

**AUC-ROC**

AUC summarizes the ROC curve as one number: the probability that a randomly chosen positive is scored higher than a randomly chosen negative (for ranking interpretations). Higher AUC usually means better **separation** between classes in ranking terms. AUC can still be misleading under strong imbalance or when the cost structure does not match "ranking quality" -- which is why you still inspect the confusion matrix at **your** chosen threshold.

**Threshold selection**

There is no universal "best" threshold. You pick it using domain costs (false positive vs false negative), regulatory constraints, or validation performance. ROC helps you **see** the frontier of trade-offs; the confusion matrix shows **one** operating point. Changing the threshold moves you along the ROC curve and changes the confusion matrix.

```python
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix

auc = roc_auc_score(y_true, y_score)
fpr, tpr, thresholds = roc_curve(y_true, y_score)
cm = confusion_matrix(y_true, y_pred)
```

**matplotlib** / **seaborn** are typical for plotting ROC and heatmapping the matrix; the important part is **reading** the plot, not memorizing styling.

## Connecting to Practice

Use the **confusion matrix** to debug *which* errors dominate and to communicate with non-technical partners. Use **ROC/AUC** to compare models that output scores and to reason about performance **across thresholds** -- especially when the business has not fixed a threshold yet. Once a threshold is fixed for deployment, report **precision/recall** and the confusion matrix at **that** threshold; reporting only AUC can hide bad behavior at the actual operating point.

---

## Further Learning & Resources

**Documentation**

- **[Scikit-learn: ROC curves and AUC](https://scikit-learn.org/stable/modules/model_evaluation.html#receiver-operating-characteristic-roc)** - *Docs*: Official guide to ROC/AUC computation and interpretation.
- **[Scikit-learn: Confusion matrix](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html)** - *Docs*: API reference for confusion matrix generation.

**Interactive**

- **[MLU-Explain: ROC and AUC](https://mlu-explain.github.io/roc/)** - *Interactive*: Outstanding visual walkthrough of ROC curves and AUC.
- **[Scikit-learn: ConfusionMatrixDisplay examples](https://scikit-learn.org/stable/auto_examples/index.html)** - *Interactive*: Runnable examples for plotting confusion matrices.
