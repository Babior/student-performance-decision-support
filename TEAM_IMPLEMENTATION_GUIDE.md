# TEAM_IMPLEMENTATION_GUIDE.md

## Project Title
**Student Performance Decision-Support System Using Machine Learning**

## Purpose
This guide explains how each team member should implement the project using the decisions already fixed in `PROJECT_DECISIONS.md`.

`PROJECT_DECISIONS.md` defines **what the team has agreed**.  
This guide defines **who does what, where they work, which variables to use, what outputs to produce, and how work is handed over**.

---

# 1. Shared Project Rules

All members must use these constants:

```python
PROJECT_NAME = "Student Performance Decision-Support System"

TARGET_COLUMN = "needs_support"
TARGET_SOURCE_COLUMN = "G3"

PASS_THRESHOLD = 10
POSITIVE_CLASS = 1

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
DEFAULT_CLASSIFICATION_THRESHOLD = 0.50
```

## Target definition

```python
df["needs_support"] = (df["G3"] < 10).astype(int)
```

| Value | Meaning |
|---|---|
| `1` | May require academic support |
| `0` | Likely to pass |

## Shared feature sets

```python
X_base = df.drop(columns=["G3", "needs_support"])
X_early = X_base.drop(columns=["G1", "G2"])
X_progress = X_base.copy()
y = df["needs_support"]
```

## Approved experiments

### Early-warning experiment
Uses:

```python
X_early
```

Must exclude:

```text
G1
G2
G3
```

### Progress-informed experiment
Uses:

```python
X_progress
```

May include:

```text
G1
G2
```

Must exclude:

```text
G3
```

## Approved models

```text
logistic_regression
decision_tree
random_forest
```

K-means clustering is optional and must not begin until the supervised project is complete.

## Required metrics

Every model must report:

```text
training accuracy
testing accuracy
support precision
support recall
support F1-score
ROC-AUC
confusion matrix
cross-validation mean
cross-validation standard deviation
```

The most important practical metric is:

```text
Recall for needs_support = 1
```

---

# 2. Work That Must Be Completed Before Independent Work Begins

## 2.1 Add the dataset

Store the Portuguese UCI file as:

```text
data/raw/student-por.csv
```

Do not alter the raw file.

## 2.2 Complete the foundation notebook

File:

```text
notebooks/00_project_foundation.ipynb
```

It must:

1. import libraries;
2. load the dataset;
3. inspect shape, columns, and data types;
4. check missing values;
5. check duplicates;
6. create the binary target;
7. display class balance;
8. define `X_base`, `X_early`, `X_progress`, and `y`;
9. identify numerical and categorical columns;
10. confirm that `G3` is excluded;
11. confirm that `G1` and `G2` are excluded from `X_early`;
12. create the shared stratified split;
13. save the shared train and test row indices.

Recommended files:

```text
data/processed/train_indices.csv
data/processed/test_indices.csv
```

The same students must appear in the train and test sets for both experiments. Only the feature columns should differ.

## 2.3 Update `src/config.py`

Recommended contents:

```python
from pathlib import Path

PROJECT_NAME = "Student Performance Decision-Support System"

TARGET_COLUMN = "needs_support"
TARGET_SOURCE_COLUMN = "G3"

PASS_THRESHOLD = 10
POSITIVE_CLASS = 1

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
DEFAULT_CLASSIFICATION_THRESHOLD = 0.50

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "student-por.csv"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_PATH = PROJECT_ROOT / "models" / "final_model.joblib"
MODEL_METADATA_PATH = PROJECT_ROOT / "models" / "model_metadata.json"

EARLY_WARNING_EXPERIMENT = "early_warning"
PROGRESS_INFORMED_EXPERIMENT = "progress_informed"
```

## 2.4 Complete `src/data.py`

Required functions:

```python
load_student_data()
validate_required_columns()
create_target()
get_feature_sets()
save_split_indices()
load_split_indices()
```

## 2.5 Shared approval check

Before separate work begins, all four members must confirm:

- the dataset loads correctly;
- the target contains only `0` and `1`;
- `G3` is absent from all model features;
- `G1` and `G2` are absent from `X_early`;
- `G1` and `G2` are present in `X_progress`;
- the class distribution is agreed;
- the train-test split is shared;
- the repository runs in each member's environment.

---

# 3. Kwantsimah Orleans Boham

## Role
**Data Preparation and Exploratory Data Analysis Lead**

## Branch

```bash
git checkout -b feature/data-eda
```

## Primary files

```text
notebooks/00_project_foundation.ipynb
notebooks/01_data_cleaning_eda.ipynb
src/data.py
reports/figures/
data/processed/
```

## Responsibilities

### 3.1 Lead the foundation notebook

Required variables:

```python
df
X_base
X_early
X_progress
y
numeric_features
categorical_features
train_indices
test_indices
```

Required checks:

```python
df.shape
df.head()
df.info()
df.describe(include="all")
df.isnull().sum()
df.duplicated().sum()
df[TARGET_COLUMN].value_counts()
df[TARGET_COLUMN].value_counts(normalize=True)
```

### 3.2 Complete exploratory data analysis

The EDA notebook should explain:

- dataset size;
- feature meanings;
- data types;
- missing values;
- duplicates;
- unusual values;
- numerical distributions;
- categorical frequencies;
- target balance;
- important patterns related to support status.

### 3.3 Required visualisations

At minimum:

```text
target_distribution.png
grade_distributions.png
absences_by_support.png
studytime_by_support.png
subgroup_distributions.png
correlation_heatmap.png
```

Save them in:

```text
reports/figures/
```

### 3.4 Subgroup summaries

Prepare summaries for:

```text
sex
school
address
```

Suggested code:

```python
df.groupby("sex")[TARGET_COLUMN].agg(["count", "mean"])
df.groupby("school")[TARGET_COLUMN].agg(["count", "mean"])
df.groupby("address")[TARGET_COLUMN].agg(["count", "mean"])
```

These outputs will support Selorm's fairness analysis.

### 3.5 Data function responsibilities

Ensure `src/data.py`:

- loads `student-por.csv` with `sep=";"`;
- validates required columns;
- creates the target correctly;
- returns all feature sets;
- saves and loads split indices.

### 3.6 Rules

- Do not remove duplicates silently.
- Report duplicates before removing them.
- Do not overwrite the raw dataset.
- Do not perform model-time imputation in the EDA notebook.
- Keep imputation inside preprocessing pipelines.

## Handoff

Provide:

- completed foundation notebook;
- completed EDA notebook;
- feature lists;
- target distribution;
- subgroup summaries;
- split-index files;
- saved figures;
- working `src/data.py`.

## Completion standard

Kwantsimah is finished when another teammate can run the notebooks and obtain the same target, feature sets, class balance, and split.

---

# 4. Keoni Melvin Mensah Daniels

## Role
**Preprocessing and Logistic Regression Lead**

## Branch

```bash
git checkout -b feature/preprocessing-logistic
```

## Primary files

```text
notebooks/02_preprocessing_logistic.ipynb
src/preprocessing.py
```

## Dependencies

Keoni requires:

- the agreed feature sets;
- numerical and categorical column lists;
- shared split indices;
- the agreed target.

## Responsibilities

### 4.1 Build preprocessing functions

Recommended functions:

```python
build_numeric_pipeline()
build_categorical_pipeline()
build_preprocessor()
build_model_pipeline()
```

Recommended numerical pipeline:

```python
numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)
```

Recommended categorical pipeline:

```python
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

Recommended transformer:

```python
preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features)
    ]
)
```

### 4.2 Train logistic regression

Train:

```text
logistic regression + early_warning
logistic regression + progress_informed
```

Suggested variables:

```python
logistic_early_pipeline
logistic_progress_pipeline

logistic_early_result
logistic_progress_result
```

Recommended classifier:

```python
LogisticRegression(
    max_iter=1000,
    random_state=RANDOM_STATE
)
```

### 4.3 Cross-validation

Use:

```python
cv = StratifiedKFold(
    n_splits=CV_FOLDS,
    shuffle=True,
    random_state=RANDOM_STATE
)
```

Required scoring:

```text
recall
f1
roc_auc
```

### 4.4 Result format

```python
result = {
    "experiment_name": experiment_name,
    "model_name": "logistic_regression",
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

### 4.5 Leakage checks

Confirm that:

- preprocessing is fitted only on training data;
- the encoder is inside the pipeline;
- the scaler is inside the pipeline;
- `G3` is absent;
- `G1` and `G2` are absent from the early-warning model;
- the shared split is reused;
- the test set is not used during training or tuning.

## Handoff

Provide:

- completed `src/preprocessing.py`;
- two logistic pipelines;
- two result dictionaries;
- cross-validation outputs;
- explanation of preprocessing and leakage prevention.

## Completion standard

Keoni is finished when both logistic experiments run from raw mixed-type inputs and return the agreed result format.

---

# 5. Selorm Kwame Hlodze

## Role
**Tree Models and Final Evaluation Lead**

## Branch

```bash
git checkout -b feature/tree-models-evaluation
```

## Primary files

```text
notebooks/03_tree_random_forest.ipynb
notebooks/04_final_evaluation.ipynb
src/evaluation.py
```

## Dependencies

Selorm requires:

- the shared split;
- preprocessing functions;
- logistic results;
- subgroup summaries.

## Responsibilities

### 5.1 Train four tree-based pipelines

```text
decision tree + early_warning
decision tree + progress_informed
random forest + early_warning
random forest + progress_informed
```

Suggested variables:

```python
tree_early_pipeline
tree_progress_pipeline
forest_early_pipeline
forest_progress_pipeline
```

Suggested result variables:

```python
tree_early_result
tree_progress_result
forest_early_result
forest_progress_result
```

### 5.2 Baseline models

Decision tree:

```python
DecisionTreeClassifier(
    random_state=RANDOM_STATE
)
```

Random forest:

```python
RandomForestClassifier(
    n_estimators=200,
    random_state=RANDOM_STATE,
    n_jobs=-1
)
```

### 5.3 Controlled tuning

Permitted parameters include:

```text
max_depth
min_samples_split
min_samples_leaf
n_estimators
class_weight
```

Any tuning must explain:

- what changed;
- why it changed;
- the effect on support recall;
- the effect on overfitting.

### 5.4 Evaluation functions

Complete these functions in `src/evaluation.py`:

```python
evaluate_classifier()
cross_validate_classifier()
plot_confusion_matrix()
compare_model_results()
evaluate_subgroups()
```

Use:

```python
pos_label=POSITIVE_CLASS
```

### 5.5 Overfitting checks

Compare:

```text
training performance
testing performance
cross-validation stability
```

Warning signs include:

- very high training performance;
- much lower testing performance;
- unstable fold results.

### 5.6 Final comparison

Combine all six model-experiment combinations in:

```python
model_results_df
```

### 5.7 Final model selection order

1. reject models with leakage;
2. reject models with severe overfitting;
3. prioritise support recall;
4. compare support F1;
5. compare ROC-AUC;
6. compare cross-validation stability;
7. consider interpretability;
8. prefer the simpler model when performance is close.

### 5.8 Threshold rule

Use `0.50` for the first comparison.

Any threshold adjustment must:

- use training or validation data only;
- show the recall-precision trade-off;
- report both the default and adjusted threshold;
- not use the final test set to choose the threshold.

### 5.9 Fairness checks

Evaluate, where sample sizes allow:

```text
sex
school
address
```

Report:

```text
group count
accuracy
support recall
support precision
support F1
```

Interpret results cautiously.

## Handoff

Provide:

- four trained tree-based pipelines;
- overfitting analysis;
- controlled tuning results;
- model comparison table;
- subgroup evaluation;
- final model recommendation;
- final confusion matrix;
- selected complete pipeline.

## Completion standard

Selorm is finished when all model comparisons are complete and the final model recommendation is clearly justified.

---

# 6. Jamal Kwesi Gbana

## Role
**Application Development, Integration, Testing, and Documentation Lead**

## Branch

```bash
git checkout -b feature/streamlit-integration
```

## Primary files

```text
app.py
src/utils.py
tests/
models/
README.md
```

## Dependencies

Jamal may begin the Streamlit interface immediately with placeholders.

Final prediction integration requires Selorm's selected complete pipeline.

## Responsibilities

### 6.1 Streamlit structure

The app should include:

```text
Home
Student Input
Prediction Result
Model Comparison
Confusion Matrix
Dataset Visualisations
Ethics and Limitations
```

### 6.2 Input form

The form must:

- use only model input features;
- avoid student names and identification numbers;
- validate numerical ranges;
- use approved categorical options;
- return a one-row DataFrame.

Suggested variables:

```python
input_data
input_df
```

Recommended structure:

```python
input_df = pd.DataFrame([input_data])
```

### 6.3 Save and load model

Save:

```python
joblib.dump(final_pipeline, MODEL_PATH)
```

Load:

```python
final_pipeline = joblib.load(MODEL_PATH)
```

The saved object must include:

```text
preprocessing + classifier
```

### 6.4 Prediction variables

```python
prediction
support_probability
display_label
```

Approved labels:

```text
Likely to pass
May benefit from additional academic support
```

### 6.5 Responsible-use message

Display:

> This prediction is intended to support professional judgment. It should not be used as the sole basis for academic, disciplinary, or counselling decisions.

### 6.6 Utility functions

Recommended `src/utils.py` functions:

```python
load_model()
load_model_metadata()
prepare_input_dataframe()
format_prediction_label()
validate_input_columns()
```

### 6.7 Model metadata

Create:

```text
models/model_metadata.json
```

Recommended fields:

```json
{
  "selected_experiment": "",
  "selected_model": "",
  "target_column": "needs_support",
  "positive_class": 1,
  "feature_names": [],
  "support_recall": 0.0,
  "support_f1": 0.0,
  "roc_auc": 0.0,
  "random_state": 42,
  "classification_threshold": 0.5
}
```

### 6.8 Tests

Create:

```text
tests/test_data.py
tests/test_preprocessing.py
tests/test_model.py
tests/test_app_inputs.py
```

Required tests:

1. target contains only `0` and `1`;
2. `G3` is absent from all features;
3. `G1` and `G2` are absent from early-warning features;
4. split indices load correctly;
5. saved model loads;
6. valid input produces a prediction;
7. prediction is a valid class;
8. probability is between `0` and `1`;
9. app input columns match model expectations;
10. missing files raise clear errors.

### 6.9 README

Each member writes their own technical section.

Jamal combines them into a final README containing:

- project overview;
- dataset;
- target definition;
- project structure;
- team roles;
- installation;
- notebook order;
- experiments;
- final model;
- Streamlit instructions;
- ethical limitations;
- testing instructions;
- demonstration steps.

### 6.10 Demonstration flow

1. explain the project;
2. enter sample student data;
3. generate a prediction;
4. explain probability;
5. show model comparison;
6. show confusion matrix;
7. explain limitations and ethics.

## Handoff

Provide:

- working Streamlit app;
- integrated final model;
- input validation;
- metadata;
- tests;
- final README;
- demonstration-ready repository.

## Completion standard

Jamal is finished when `streamlit run app.py` works, predictions are valid, tests pass, and another person can follow the README from a fresh environment.

---

# 7. File Ownership

| File or folder | Primary owner | Reviewer |
|---|---|---|
| `PROJECT_DECISIONS.md` | All members | All members |
| `TEAM_IMPLEMENTATION_GUIDE.md` | All members | All members |
| `00_project_foundation.ipynb` | Kwantsimah | Entire team |
| `01_data_cleaning_eda.ipynb` | Kwantsimah | Keoni |
| `src/data.py` | Kwantsimah | Keoni |
| `02_preprocessing_logistic.ipynb` | Keoni | Selorm |
| `src/preprocessing.py` | Keoni | Selorm |
| `03_tree_random_forest.ipynb` | Selorm | Keoni |
| `04_final_evaluation.ipynb` | Selorm | Entire team |
| `src/evaluation.py` | Selorm | Jamal |
| `app.py` | Jamal | Entire team |
| `src/utils.py` | Jamal | Selorm |
| `tests/` | Jamal | Entire team |
| `README.md` | Jamal integrates | All members contribute |
| `models/` | Jamal manages | Selorm supplies final pipeline |

---

# 8. Work Sequence

## Stage 1: Foundation

All members:

- review both guide files;
- add the dataset;
- complete the foundation notebook;
- agree on feature sets;
- agree on the split.

## Stage 2: Parallel work

Kwantsimah:

- completes EDA and subgroup summaries.

Keoni:

- builds preprocessing and logistic regression.

Selorm:

- prepares evaluation utilities.

Jamal:

- builds the Streamlit interface using placeholders.

## Stage 3: Tree models and comparison

Selorm:

- trains tree models;
- combines all results;
- checks overfitting and fairness;
- recommends the final model.

## Stage 4: Integration

Jamal:

- saves and loads the selected pipeline;
- integrates predictions;
- adds charts;
- completes tests and README.

## Stage 5: Final review

All members:

- run notebooks;
- run tests;
- review ethics;
- review documentation;
- test Streamlit;
- prepare presentation material;
- rehearse the demonstration.

---

# 9. Git Workflow

Before working:

```bash
git checkout main
git pull origin main
git checkout -b feature/your-branch-name
```

During work:

```bash
git status
git add <relevant-files>
git commit -m "Clear description of completed work"
```

Before opening a pull request:

```bash
git checkout main
git pull origin main
git checkout feature/your-branch-name
git merge main
```

Push:

```bash
git push -u origin feature/your-branch-name
```

No branch should be merged into `main` without at least one teammate reviewing and approving the pull request.

Good commit messages:

```text
Add target creation and shared feature sets
Build logistic regression preprocessing pipeline
Add tree model overfitting analysis
Integrate final model into Streamlit
Add prediction input validation tests
```

Avoid:

```text
update
changes
work
final
done
```

---

# 10. Handoff Format

Every handoff should include:

```text
Completed:
Files changed:
Functions added:
Variables created:
Outputs produced:
How to run:
Tests performed:
Known issues:
Next dependency:
```

---

# 11. Common Variable Names

Use these names consistently:

```python
df
X_base
X_early
X_progress
y

X_train
X_test
y_train
y_test

numeric_features
categorical_features

preprocessor
model_pipeline

train_predictions
test_predictions
test_probabilities

support_precision
support_recall
support_f1
roc_auc

model_results
model_results_df

final_pipeline
final_model_metadata

input_data
input_df
prediction
support_probability
display_label
```

---

# 12. Shared Result Format

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

---

# 13. Definition of Done

The project is complete when:

- the raw dataset is preserved;
- the target is correct;
- both experiments are implemented;
- all three models are compared;
- preprocessing remains inside pipelines;
- leakage checks pass;
- five-fold cross-validation is complete;
- support recall is reported;
- overfitting is assessed;
- subgroup performance is examined;
- the final pipeline is saved;
- Streamlit runs correctly;
- tests pass;
- responsible-use language is included;
- README instructions are reproducible;
- all members can explain their work.

---

# 14. AI-Assisted Coding Prompt

When using AI assistance, provide both:

```text
PROJECT_DECISIONS.md
TEAM_IMPLEMENTATION_GUIDE.md
```

Use:

> Follow `PROJECT_DECISIONS.md` as the governing design document and `TEAM_IMPLEMENTATION_GUIDE.md` as the implementation plan. Work only on my assigned files and responsibilities. Preserve the approved target, class meanings, feature experiments, leakage rules, model list, metric priorities, variable names, result format, folder structure, and ethical language. Explain assumptions and do not change shared decisions without approval.

All AI-generated work must be understood, tested, reviewed, and checked for leakage before it is committed.
