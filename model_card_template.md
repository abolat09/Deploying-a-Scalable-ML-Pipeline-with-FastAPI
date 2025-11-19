# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
* **Developed by:** Ahmet Bolat
* **Date:** November 2025
* **Model Type:** Random Forest Classifier (`sklearn.ensemble.RandomForestClassifier`)
* **Version:** 1.0
* **License:** MIT
* **Input:** 14 features from the US Census dataset (categorical and continuous).
* **Output:** Binary classification. 0 indicates income `<=50K`, 1 indicates income `>50K`.

## Intended Use
* **Primary Use:** To predict whether an individual's annual income exceeds $50,000 based on demographic and employment attributes (e.g., age, education, occupation).
* **Intended Users:** Researchers, data scientists, and social scientists studying income inequality or demographic trends.
* **Out-of-Scope Use:** This model should **not** be used for automated hiring decisions, loan approvals, or any high-stakes decision-making that affects individuals' livelihoods without human oversight.

## Training Data
* **Source:** UCI Machine Learning Repository (Census Income Dataset).
* **Size:** The training set consists of 80% of the original 32,561 rows.
* **Preprocessing:**
    * Categorical features (e.g., `workclass`, `education`) were One-Hot Encoded.
    * The target label `salary` was binarized using `LabelBinarizer`.
    * Rows with missing values (`?`) in critical columns were removed during preprocessing.

## Evaluation Data
* **Source:** A 20% holdout set from the original Census Income Dataset.
* **Method:** The model was evaluated using Precision, Recall, and F1-score (Fbeta).

## Metrics
_Please include the metrics used and your model's performance on those metrics._

The model performance on the test set is as follows:

* **Precision:** 0.7391
* **Recall:** 0.6384
* **F1-Score:** 0.6851

## Ethical Considerations
The model was evaluated for bias across different demographic slices. Significant disparities were found:

* **Sex Bias:** The model performs better for Males (F1: 0.6985) than for Females (F1: 0.5995).
* **Race Bias:** Performance varies by race, with 'Asian-Pac-Islander' (F1: 0.7458) having higher scores than 'Black' (F1: 0.6723).
* **Education Bias:** The model fails completely for certain low-education groups (e.g., '7th-8th' grade has an F1 score of 0.0000), indicating it relies heavily on higher education as a predictor.

## Caveats and Recommendations
* **Data Age:** The dataset is from 1994. Inflation and economic changes mean the "$50K" threshold is not representative of modern wealth.
* **Bias Mitigation:** Before deployment, techniques such as re-sampling or adversarial debiasing should be applied to improve fairness for female and minority groups.
* **Recommendation:** Use this model only for aggregate analysis, not for individual predictions.