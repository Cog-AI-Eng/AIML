# Guide 2: Compare Experiment Runs

This guide walks you through using the SageMaker Experiments comparison features in Studio to filter runs by metrics, identify the best-performing configuration, and analyze training curves side by side.

---

## Steps

### Step 1 -- Open the Experiment in Studio
1. In SageMaker Studio, click **Experiments** in the left sidebar.
2. Click the experiment `fraudshield-fraud-detection`.
3. Verify that all three runs (run1-conservative, run2-moderate, run3-aggressive) are listed and their training jobs have completed.

### Step 2 -- Select Runs for Comparison
1. Select all three runs by clicking the checkbox next to each run name.
2. With all runs selected, look for the **Analyze** or **Chart** option in the toolbar above the run list.
3. Click **Analyze** to open the comparison view. This opens a panel where you can create charts and tables comparing runs.

### Step 3 -- Create a Metric Comparison Chart
1. In the analysis panel, click **Add chart**.
2. Select chart type: **Line chart** (or **Bar chart** for final metric values).
3. For the X-axis, select **Epoch** or **Step** (depending on how metrics were logged).
4. For the Y-axis, select `validation:auc`.
5. The chart renders a line for each run, showing how validation AUC evolved during training.
6. Hover over the lines to see exact values at each step.

### Step 4 -- Filter and Sort by Metric
1. Return to the runs table view.
2. Click on the **validation:auc** column header to sort runs by their final AUC value in descending order.
3. Identify which run achieved the highest validation AUC.
4. Note the hyperparameter values for the best run. Record:
   - `max_depth`, `eta`, `num_round`, `subsample`
   - Final `validation:auc` value

### Step 5 -- Compare Hyperparameter Settings
1. In the comparison view, add a **Table** chart if not already visible.
2. Select the parameters to display: `max_depth`, `eta`, `num_round`, `subsample`.
3. Also include the metric columns: `train:auc`, `validation:auc`.
4. Review the table to see all runs side by side. Look for patterns:
   - Does a deeper tree (`max_depth=8`) overfit compared to a shallow tree (`max_depth=3`)?
   - Does a higher learning rate (`eta=0.3`) converge faster but plateau lower?
5. Note any cases where training AUC is much higher than validation AUC (indicates overfitting).

### Step 6 -- Analyze Training Curves for Overfitting
1. Create another line chart with both `train:auc` and `validation:auc` on the Y-axis.
2. Filter to show only the best run and the worst run.
3. Compare the gap between training and validation curves:
   - A small, consistent gap indicates good generalization.
   - A large or widening gap indicates overfitting.
4. Document your observations. The best configuration for production is the one with the highest validation AUC and the smallest train-validation gap.

---

## Presentation Checkpoint
Be prepared to show:
- The metric comparison chart with all three runs plotted.
- The sorted runs table with the best run highlighted.
- A training curve analysis showing the difference between the best and worst configurations.

---

## Key Concepts
- **Experiment Comparison:** SageMaker Studio allows side-by-side visualization of multiple runs, enabling data-driven hyperparameter selection.
- **Validation AUC as Selection Criterion:** The validation metric (not the training metric) is the correct basis for model selection because it measures performance on unseen data.
- **Overfitting Indicators:** A growing gap between training and validation metrics across epochs signals that the model is memorizing training data rather than learning generalizable patterns.
- **Run Metadata:** Each run automatically captures hyperparameters, metrics, input data paths, and output artifacts, providing a complete record for comparison.
