# PROJECT_DECISIONS.md

## Project Title

**Student Performance Decision-Support System Using Machine Learning**

## Purpose of This Document

This file defines the shared design decisions, technical standards, naming conventions, evaluation rules, and ethical boundaries for the project.

All team members should use this document as the main implementation guide when writing code manually or with AI assistance.

Before generating, editing, or reviewing project code, every team member should provide this file to the AI tool they are using and instruct it to follow these decisions consistently.

---

## 1. Project Goal

The project will develop a machine-learning decision-support system that helps teachers and school administrators identify students who may benefit from additional academic support.

The system is intended to support professional judgment, not replace it.

The project will:

- prepare and explore a student-performance dataset;
- create a binary classification target;
- compare logistic regression, decision tree, and random forest models;
- evaluate model performance using suitable classification metrics and cross-validation;
- build a Streamlit application for prediction and results visualisation;
- communicate predictions responsibly and ethically.

---

## 2. Dataset

### Approved dataset

The project will use the **Portuguese-language file from the UCI Student Performance Dataset**.

The dataset contains:

- 649 student records;
- approximately 30 variables;
- academic, behavioural, family-support, resource-access, and demographic information;
- three grade variables:
  - `G1`: first-period grade;
  - `G2`: second-period grade;
  - `G3`: final grade.

### Raw data rule

The original dataset must be stored unchanged in:

```text
data/raw/
```

Do not overwrite the raw data file.

Any cleaned or transformed version must be stored separately in:

```text
data/processed/
```

---

## 3. Target Variable

The binary target will be created from `G3`.

### Target definition

```python
needs_support = 1 if G3 < 10 else 0
```

Equivalent implementation:

```python
df["needs_support"] = (df["G3"] < 10).astype(int)
```

### Class meaning

| Value | Meaning |
|---|---|
| `1` | May require academic support |
| `0` | Likely to pass |

### Positive class

The positive class is:

```text
needs_support = 1
```

This convention must remain the same across all notebooks, scripts, evaluation functions, saved models, and Streamlit outputs.

---

## 4. Data-Leakage Rules

`G3` directly determines the target and must never be used as an input feature.

After target creation:

```python
X = df.drop(columns=["G3", "needs_support"])
y = df["needs_support"]
```

The following rules are mandatory:

- `G3` must not appear in any model input;
- preprocessing must be fitted only on training data;
- the test set must remain untouched until final evaluation;
- encoders and scalers must be placed inside scikit-learn pipelines;
- no preprocessing step may use the full dataset before the split;
- no feature created from `G3` may be used as an input.

---

## 5. Experiments

Two supervised-learning experiments will be conducted.

### Experiment A: Early-Warning Model

This model must exclude:

```text
G1
G2
G3
```

Purpose:

To predict support needs before substantial academic results are available.

Suggested naming:

```python
experiment_name = "early_warning"
```

### Experiment B: Progress-Informed Model

This model may include:

```text
G1
G2
```

It must still exclude:

```text
G3
```

Purpose:

To test how much earlier grades improve prediction of final performance.

Suggested naming:

```python
experiment_name = "progress_informed"
```

### Shared feature-set definitions

```python
X_base = df.drop(columns=["G3", "needs_support"])

X_early = X_base.drop(columns=["G1", "G2"])
X_progress = X_base.copy()
```

---

## 6. Approved Models

The project will compare exactly these three primary supervised-learning algorithms:

1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier

Suggested model names:

```text
logistic_regression
decision_tree
random_forest
```

A neural network is not part of the approved core project.

K-means clustering is optional and may only be added after the supervised models, evaluation, Streamlit app, documentation, and testing are complete.

---

## 7. Data Splitting

The project will use a stratified train-test split.

Recommended shared configuration:

```python
RANDOM_STATE = 42
TEST_SIZE = 0.20
```

Example:

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    stratify=y,
    random_state=RANDOM_STATE
)
```

### Split rules

- use the same random seed throughout the project;
- stratify using the target;
- do not repeatedly create different test sets for different models;
- all models within the same experiment must use the same split;
- cross-validation must be performed only on the training portion.

---

## 8. Cross-Validation

The project will use five-fold stratified cross-validation.

Recommended configuration:

```python
from sklearn.model_selection import StratifiedKFold

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE
)
```

Cross-validation results should report:

- mean score;
- standard deviation;
- scoring metric used.

Preferred cross-validation scoring:

```text
recall
f1
roc_auc
```

Recall for the support class should receive special attention.

---

## 9. Preprocessing

### Column types

Features must be separated into:

- numerical columns;
- categorical columns.

### Numerical preprocessing

Numerical features may require:

- missing-value imputation where necessary;
- scaling for logistic regression.

Recommended numerical pipeline:

```python
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)
```

### Categorical preprocessing

Categorical features should use:

- missing-value imputation where necessary;
- one-hot encoding.

Recommended categorical pipeline:

```python
from sklearn.preprocessing import OneHotEncoder

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)
```

### Combined preprocessing

Use a `ColumnTransformer`.

```python
from sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features)
    ]
)
```

### Pipeline rule

Every model must be wrapped in a complete pipeline:

```python
from sklearn.pipeline import Pipeline

model_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ]
)
```

Do not save the classifier separately from its preprocessing pipeline.

---

## 10. Evaluation Metrics

Each model must be evaluated using:

- accuracy;
- precision;
- recall;
- F1-score;
- confusion matrix;
- ROC-AUC, where appropriate;
- five-fold cross-validation results;
- training-versus-testing performance.

### Primary practical metric

The most important metric is:

```text
Recall for the "may require academic support" class
```

Reason:

A false negative means the system fails to identify a student who may need support.

### Evaluation rule

No model should be selected using accuracy alone.

Model selection must consider:

- support-class recall;
- F1-score;
- ROC-AUC;
- cross-validation stability;
- overfitting;
- interpretability;
- fairness;
- usefulness in the Streamlit application.

---

## 11. Standard Evaluation Output

Every model evaluation should return or record the following fields:

```text
experiment_name
model_name
training_accuracy
testing_accuracy
support_precision
support_recall
support_f1
roc_auc
cv_mean
cv_std
confusion_matrix
```

Recommended result dictionary:

```python
result = {
    "experiment_name": experiment_name,
    "model_name": model_name,
    "training_accuracy": training_accuracy,
    "testing_accuracy": testing_accuracy,
    "support_precision": support_precision,
    "support_recall": support_recall,
    "support_f1": support_f1,
    "roc_auc": roc_auc,
    "cv_mean": cv_mean,
    "cv_std": cv_std,
    "confusion_matrix": confusion_matrix
}
```

All team members should use the same output structure so results can be combined easily.

---

## 12. Overfitting Checks

Each model must compare training and testing performance.

A model may be overfitting when:

- training performance is very high;
- testing performance is noticeably lower;
- cross-validation results vary greatly across folds.

The decision tree and random forest should be checked especially carefully for overfitting.

Any tuning decision must be documented.

---

## 13. Reproducibility

Use the same random seed throughout the project:

```python
RANDOM_STATE = 42
```

All relevant model constructors and data splits should use this value where supported.

Examples:

```python
DecisionTreeClassifier(random_state=RANDOM_STATE)

RandomForestClassifier(random_state=RANDOM_STATE)
```

The project should run from a fresh environment using:

```bash
pip install -r requirements.txt
```

---

## 14. File and Folder Structure

The agreed project structure is:

```text
student-performance-decision-support/
│
├── README.md
├── PROJECT_DECISIONS.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── 00_project_foundation.ipynb
│   ├── 01_data_cleaning_eda.ipynb
│   ├── 02_preprocessing_logistic.ipynb
│   ├── 03_tree_random_forest.ipynb
│   └── 04_final_evaluation.ipynb
│
├── src/
│   ├── config.py
│   ├── data.py
│   ├── preprocessing.py
│   ├── evaluation.py
│   └── utils.py
│
├── models/
│   └── .gitkeep
│
├── reports/
│   └── figures/
│
├── tests/
│
└── app.py
```

---

## 15. Coding Standards

### General rules

- use clear and descriptive variable names;
- avoid unnecessary duplication;
- place repeated logic in reusable functions;
- add docstrings to important functions;
- include comments only where they improve understanding;
- keep notebook cells focused;
- remove unused imports;
- do not leave unexplained TODO sections in the final submission;
- do not hard-code local file paths;
- use relative paths;
- do not commit secrets or personal information.

### Naming conventions

Use:

- `snake_case` for variables and functions;
- `PascalCase` for classes;
- uppercase names for constants.

Examples:

```python
RANDOM_STATE = 42

def create_target(dataframe):
    ...

class ModelEvaluator:
    ...
```

---

## 16. Git and Collaboration Rules

### Branches

Each team member should work on a separate branch.

Suggested branch names:

```text
feature/data-eda
feature/preprocessing-logistic
feature/tree-random-forest
feature/streamlit-integration
```

### Commit messages

Use clear commit messages.

Good examples:

```text
Add target creation and class-balance analysis
Build preprocessing pipeline for mixed data
Add logistic regression evaluation
Create Streamlit prediction form
Fix feature-order mismatch during inference
```

Avoid vague messages such as:

```text
update
changes
final
work
```

### Merge rules

Before merging:

- pull the latest changes;
- run the notebook or script;
- confirm that no existing functionality is broken;
- review the pull request;
- resolve conflicts carefully;
- update documentation where necessary.

---

## 17. Model Saving

The selected final model must be saved as a complete pipeline.

Recommended format:

```python
import joblib

joblib.dump(final_pipeline, "models/final_model.joblib")
```

The saved object should include:

```text
preprocessing + classifier
```

Do not save only the classifier.

Suggested metadata file:

```text
models/model_metadata.json
```

Metadata should record:

- selected experiment;
- selected model;
- target definition;
- feature names;
- metric results;
- training date;
- random seed;
- package versions.

---

## 18. Streamlit Application

The Streamlit application must allow a teacher or administrator to:

- enter selected student characteristics;
- receive a prediction;
- view an estimated probability;
- see a responsible-use message;
- review model comparisons;
- view a confusion matrix;
- explore selected dataset charts.

### Streamlit rules

The app must:

- use the same saved pipeline used during evaluation;
- use the same feature names and data types used during training;
- validate user inputs;
- display readable labels;
- avoid collecting student names or identification numbers;
- avoid presenting predictions as certainty;
- explain the output in plain language.

### Approved prediction labels

Use:

```text
Likely to pass
May benefit from additional academic support
```

Do not use:

```text
Bad student
Certain to fail
Weak student
Hopeless
```

### Responsible-use message

The application should display a message similar to:

> This prediction is intended to support professional judgment. It should not be used as the sole basis for academic, disciplinary, or counselling decisions.

---

## 19. Ethics and Fairness

The project uses a public dataset and must not collect real student identities.

The team must acknowledge that:

- the data comes from two Portuguese secondary schools;
- results may not generalise directly to Ghanaian schools or universities;
- subgroup sizes may be small;
- predictions may reflect limitations or biases in the original data;
- no academic or disciplinary decision should rely on the model alone.

Where feasible, subgroup performance should be examined using available variables such as:

- sex;
- school;
- address type.

Subgroup results must be interpreted cautiously.

---

## 20. Optional K-Means Extension

K-means clustering is optional.

It may only begin after completing:

- supervised models;
- evaluation;
- cross-validation;
- Streamlit application;
- testing;
- README;
- final documentation.

If completed, the clustering section should:

- exclude the pass/support target;
- use selected academic and behavioural features;
- scale appropriate variables;
- use the elbow method;
- calculate silhouette scores;
- explain the meaning of each cluster cautiously.

Clustering must not replace the supervised models.

---

## 21. Required Foundation Notebook

Before dividing model-development work, the team must complete:

```text
notebooks/00_project_foundation.ipynb
```

This notebook should:

1. import libraries;
2. set the random seed;
3. load the raw Portuguese dataset;
4. inspect shape, columns, and data types;
5. check duplicates and missing values;
6. create the binary target;
7. display class balance;
8. define the early-warning feature set;
9. define the progress-informed feature set;
10. verify that `G3` is excluded;
11. create the agreed stratified split;
12. document the evaluation metrics;
13. confirm the shared project conventions.

No team member should create a separate target definition or independent split without team approval.

---

## 22. Team Responsibilities

### Kwantsimah Orleans Boham — Data Preparation Lead

Primary responsibilities:

- clean the dataset;
- create the target variable;
- explore the data;
- prepare charts and visualisations;
- document data quality and class balance.

### Keoni Melvin Mensah Daniels — Data Processing Lead

Primary responsibilities:

- identify numerical and categorical variables;
- build the preprocessing pipeline;
- prepare the data for training;
- build the first logistic regression model.

### Selorm Kwame Hlodze — Model Testing Lead

Primary responsibilities:

- build the decision tree model;
- build the random forest model;
- compare model results;
- perform cross-validation;
- check overfitting;
- examine model fairness.

### Jamal Kwesi Gbana — App Development Lead

Primary responsibilities:

- build the Streamlit application;
- connect the saved pipeline to the app;
- integrate team outputs;
- test the complete system;
- prepare the README;
- support the final demonstration.

---

## 23. Definition of Done

The project will be considered complete when:

- the raw dataset is preserved;
- the target is created correctly;
- both experiments are implemented;
- all three approved models are compared;
- preprocessing occurs inside pipelines;
- leakage checks are passed;
- five-fold cross-validation is completed;
- required metrics are reported;
- overfitting is assessed;
- the final model is saved as a full pipeline;
- the Streamlit application works;
- responsible-use language is included;
- subgroup performance is examined where feasible;
- the README explains how to install and run the project;
- another person can reproduce the results from the repository.

---

## 24. Instructions for AI-Assisted Coding

When using an AI coding assistant, provide this file and use a prompt similar to:

> Follow the attached `PROJECT_DECISIONS.md` as the governing technical design guide. Do not change the target definition, positive-class meaning, experiment structure, approved models, data-leakage rules, evaluation priorities, folder structure, or ethical language without explicit approval. Produce code that is compatible with the shared repository and explain any assumptions.

AI-generated code must still be:

- read and understood by the team member;
- tested before committing;
- checked against this document;
- reviewed for leakage;
- reviewed for incorrect assumptions;
- documented clearly.

AI output should not automatically override an approved team decision.

---

## 25. Change-Control Rule

Any change to this document must be:

1. discussed by the team;
2. justified;
3. recorded in Git;
4. reflected in affected notebooks, scripts, tests, and documentation.

Important decisions must not be changed silently by one team member.

---

## 26. Current Approved Constants

```python
PROJECT_NAME = "Student Performance Decision-Support System"
TARGET_COLUMN = "needs_support"
TARGET_SOURCE_COLUMN = "G3"
PASS_THRESHOLD = 10
POSITIVE_CLASS = 1
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
```

---

## 27. Final Reminder

The project is a decision-support prototype.

Its success will be judged by:

- completeness;
- reliability;
- reproducibility;
- prevention of data leakage;
- appropriate evaluation;
- clear integration;
- responsible presentation.

High accuracy alone does not make the project successful.
