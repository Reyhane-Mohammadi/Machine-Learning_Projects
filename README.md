# Machine Learning Problem Sets

My Python solutions to three machine learning problem sets from Winter 2026. The assignments cover regression, classification, and experiments on how model choices affect performance.

## [PS01 — Regression](./PS01/)

- **Car price prediction:** Compare a numeric-only linear regression baseline with models that use one-hot encoding and polynomial features.
- **Data splitting:** Implement gradient descent and explore how train/validation/test splits, polynomial degree, and early stopping affect test error.
- **Locally weighted linear regression:** Visualize how the bandwidth parameter changes the fitted curve.
- **Hierarchical linear models:** Fit a model to synthetic grouped data and compare predictions across groups.

## [PS02 — Classification](./PS02/)

- **Weighted logistic regression:** Implement logistic regression from scratch using Newton’s method and compare decision boundaries at different bandwidths.
- **Naive Bayes:** Classify breast tumors in the Wisconsin Diagnostic dataset and compare a NumPy-style implementation from scratch with scikit-learn’s Gaussian Naive Bayes.
- **Diabetes model comparison:** Evaluate logistic regression, SVM, decision trees, Naive Bayes, and XGBoost on the Pima Indians Diabetes dataset.
- **SVM tuning:** Explore how kernel choice and hyperparameters affect diabetes classification.

## [PS03 — Model Comparison and Robustness](./PS03/)

Five experiments on the Wisconsin Diagnostic Breast Cancer dataset:

1. Compare linear, RBF, and polynomial SVM kernels using cross-validation.
2. Examine feature importance and overfitting in random forests.
3. Track XGBoost learning curves and evaluate early stopping.
4. Compare SVM, random forest, and XGBoost using ROC curves.
5. Test how random forest and XGBoost respond to Gaussian noise and corruption of a single feature.

## Running the exercises

Each exercise is a standalone Python script. Check its imports for the required packages and its data-loading lines for the expected input filename and path. Some datasets are not included in this repository. For example, the PS01 car price exercise expects `data.csv`; provide your own authorized copy before running it.
