# Diabetes Prediction with Machine Learning Classification Models

## Project Overview

This project focuses on building a machine learning classification pipeline to predict whether a person is likely to have diabetes based on demographic and medical indicators.

The project uses a cleaned diabetes prediction dataset and compares three different classification algorithms:

- Logistic Regression
- Support Vector Machine (SVM)
- Gaussian Naive Bayes

The main goal is not only to train classification models, but also to demonstrate a complete machine learning workflow including data exploration, preprocessing, feature encoding, feature scaling, model training, hyperparameter tuning, evaluation, model comparison, and model saving.

---

## Dataset

The dataset used in this project is the **Diabetes Prediction Dataset** from Kaggle.

The dataset contains patient-level demographic and medical information. The target variable is:

```text
diabetes
```

Target classes:

```text
0 = Non-diabetic
1 = Diabetic
```

### Main Features

The dataset includes the following features:

| Feature | Description |
|---|---|
| `gender` | Gender of the patient |
| `age` | Age of the patient |
| `hypertension` | Whether the patient has hypertension |
| `heart_disease` | Whether the patient has heart disease |
| `smoking_history` | Smoking history category |
| `bmi` | Body Mass Index |
| `HbA1c_level` | HbA1c blood test level |
| `blood_glucose_level` | Blood glucose level |
| `diabetes` | Target variable |

After preprocessing, the final dataset contains **96,128 rows**.

---

## Problem Type

This is a **binary classification** problem.

The objective is to predict whether a patient belongs to one of two classes:

```text
0 = No diabetes
1 = Diabetes
```

Because this is a healthcare-related classification task, accuracy alone is not enough to evaluate the models. Metrics such as **recall**, **precision**, and **F1-score** are especially important.

In this project, recall is important because a false negative means that a person who actually has diabetes is predicted as non-diabetic.

---

## Technologies Used

The project was developed using Python and common data science libraries:

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Jupyter Notebook

---

## Project Structure

```text
diabetes-prediction-classification/
│
├── data/
│   ├── raw/
│   │   └── diabetes_prediction_dataset.csv
│   │
│   └── processed/
│       └── diabetes_cleaned.csv
│
├── notebooks/
│   ├── 01_data_loading_and_overview.ipynb
│   ├── 02_data_preprocessing.ipynb
│   └── 03_model_training_and_evaluation.ipynb
│
├── outputs/
│   ├── figures/
│   │   └── confusion_matrix_svm.png
│   │
│   └── reports/
│       └── model_comparison.csv
│
├── models/
│   ├── best_svm_model.pkl
│   ├── onehot_encoder.pkl
│   ├── standard_scaler.pkl
│   └── feature_columns.pkl
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Notebook Workflow

The project is organized into three notebooks.

---

## 1. Data Loading and Overview

Notebook:

```text
01_data_loading_and_overview.ipynb
```

The first notebook focuses on understanding the raw dataset.

### Main Steps

- Load the raw dataset
- Inspect the first rows
- Check dataset shape
- Review column names and data types
- Check missing values
- Check duplicate rows
- Analyze the target variable distribution
- Explore numerical and categorical features
- Visualize feature distributions
- Analyze relationships between important features and the target variable
- Create correlation analysis
- Summarize initial findings

### Key Observations

The initial analysis showed that:

- The target variable is `diabetes`
- The dataset contains both numerical and categorical features
- `HbA1c_level`, `blood_glucose_level`, `bmi`, and `age` appear to be important features for diabetes prediction
- Fully duplicated rows exist in the dataset
- The `gender` column contains a rare `Other` category
- The `smoking_history` column is categorical and needs encoding before model training

---

## 2. Data Preprocessing

Notebook:

```text
02_data_preprocessing.ipynb
```

The second notebook cleans the dataset and prepares it for machine learning.

### Main Steps

- Load the raw dataset
- Remove fully duplicated rows
- Remove the rare `Other` category from the `gender` column
- Check missing values
- Encode the binary `gender` feature
- Identify numerical and categorical features
- Save the cleaned dataset

### Duplicate Handling

Fully duplicated rows were removed because repeated observations can cause some patterns to be overrepresented during model training.

### Gender Encoding

After removing the rare `Other` category, the `gender` column contains two categories:

```text
Female
Male
```

The column was converted into numerical form:

```text
Female -> 0
Male   -> 1
```

### Smoking History Strategy

The `smoking_history` column was intentionally kept as a categorical feature in the preprocessing notebook.

It was not label encoded because it contains multiple categories without a natural numerical order. Label encoding could incorrectly introduce an artificial order between categories.

Instead, `smoking_history` was encoded using one-hot encoding in the model training notebook.

### Output

The cleaned dataset was saved as:

```text
data/processed/diabetes_cleaned.csv
```

---

## 3. Model Training and Evaluation

Notebook:

```text
03_model_training_and_evaluation.ipynb
```

The third notebook trains, tunes, evaluates, compares, and saves the machine learning models.

### Main Steps

- Load the cleaned dataset
- Apply one-hot encoding to `smoking_history`
- Separate features and target variable
- Split the dataset into training and test sets
- Apply feature scaling using `StandardScaler`
- Train Logistic Regression
- Train and tune SVM using `GridSearchCV`
- Train Gaussian Naive Bayes
- Evaluate models using classification metrics
- Compare models in a summary table
- Visualize the confusion matrix for the best-performing model
- Save the best model and preprocessing objects

---

## Feature Encoding

The `smoking_history` feature was encoded using Scikit-learn's `OneHotEncoder`.

One-hot encoding was used because `smoking_history` is a nominal categorical feature. The categories do not have a natural order, so label encoding would not be appropriate.

The encoder was configured with:

```text
drop="first"
handle_unknown="ignore"
sparse_output=False
```

### Why One-Hot Encoding?

One-hot encoding creates separate binary columns for each category. This allows the model to use the categorical information without assuming any ordinal relationship between categories.

---

## Feature Scaling

`StandardScaler` was used to scale the features before model training.

Scaling is especially important for:

- Logistic Regression
- Support Vector Machine

These models are sensitive to differences in feature magnitudes.

The scaler was fitted only on the training data and then applied to the test data to prevent data leakage.

---

## Models Used

### Logistic Regression

Logistic Regression was used as a baseline classification model.

It is simple, interpretable, and commonly used for binary classification problems.

### Support Vector Machine

Support Vector Machine was used as a more advanced classification model.

SVM was tuned using `GridSearchCV` to find the best combination of hyperparameters.

The tuned parameters included:

- `C`
- `gamma`
- `kernel`

The best SVM parameters found were:

```text
C = 1000
gamma = scale
kernel = poly
```

### Gaussian Naive Bayes

Gaussian Naive Bayes was used as a fast probabilistic baseline model.

It is based on Bayes' theorem and assumes that features are conditionally independent given the class label.

---

## Evaluation Metrics

The models were evaluated using the following metrics:

| Metric | Meaning |
|---|---|
| Accuracy | Overall proportion of correct predictions |
| Precision | How many predicted diabetic cases were actually diabetic |
| Recall | How many actual diabetic cases were correctly detected |
| F1-score | Balance between precision and recall |
| Confusion Matrix | Breakdown of correct and incorrect predictions |

For this healthcare-related classification task, **recall** and **F1-score** are especially important.

---

## Model Results

### Logistic Regression

| Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Non-diabetic | 0.97 | 0.99 | 0.98 |
| Diabetic | 0.88 | 0.63 | 0.74 |

Overall accuracy:

```text
0.96
```

---

### Support Vector Machine

Best SVM model after GridSearchCV:

```text
C = 1000
gamma = scale
kernel = poly
```

| Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Non-diabetic | 0.97 | 1.00 | 0.98 |
| Diabetic | 0.99 | 0.64 | 0.77 |

Overall accuracy:

```text
0.97
```

Confusion matrix for the tuned SVM model:

```text
[[21872    15]
 [  781  1364]]
```

---

### Gaussian Naive Bayes

| Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Non-diabetic | 0.97 | 0.93 | 0.95 |
| Diabetic | 0.47 | 0.66 | 0.55 |

Overall accuracy:

```text
0.90
```

---

## Model Comparison

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.96 | 0.88 | 0.63 | 0.74 |
| SVM | 0.97 | 0.99 | 0.64 | 0.77 |
| Naive Bayes | 0.90 | 0.47 | 0.66 | 0.55 |

### Interpretation

The tuned SVM model achieved the strongest overall performance, with the highest accuracy, precision, and F1-score among the tested models.

Naive Bayes achieved slightly higher recall for the diabetic class compared to the other models, but its precision was much lower. This means it identified more diabetic cases but also produced more false positives.

Because this is a healthcare-related problem, recall is important. However, the final model should also maintain reasonable precision and F1-score. Based on the overall balance of metrics, the tuned SVM model was selected as the best-performing model.

---

## Saved Model Artifacts

The following objects were saved using `joblib`:

```text
models/best_svm_model.pkl
models/onehot_encoder.pkl
models/standard_scaler.pkl
models/feature_columns.pkl
```

### Why Save These Files?

The trained model alone is not enough for future predictions.

Future input data must go through the same preprocessing steps:

- Same one-hot encoder
- Same scaler
- Same feature column order
- Same trained model

Saving these objects makes the workflow reusable and more reproducible.

---

## How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/diabetes-prediction-classification.git
cd diabetes-prediction-classification
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it:

For macOS/Linux:

```bash
source .venv/bin/activate
```

For Windows:

```bash
.venv\Scripts\activate
```

### 3. Install Requirements

```bash
pip install -r requirements.txt
```

### 4. Open Jupyter Notebook

```bash
jupyter notebook
```

Then run the notebooks in order:

```text
01_data_loading_and_overview.ipynb
02_data_preprocessing.ipynb
03_model_training_and_evaluation.ipynb
```

---

## Important Notes

- The dataset file should be placed inside `data/raw/`.
- The cleaned dataset is generated in the preprocessing notebook.
- Model outputs and reports are saved under the `outputs/` directory.
- Trained model files are saved under the `models/` directory.
- SVM with GridSearchCV can take a long time to run because multiple parameter combinations are tested using cross-validation.

---

## Future Improvements

Possible future improvements include:

- Testing K-Nearest Neighbors
- Testing Decision Tree and Random Forest models
- Adding ROC-AUC analysis
- Adding ROC curve visualization
- Using cross-validation for all models
- Trying class imbalance techniques
- Comparing threshold tuning strategies
- Building a simple prediction interface
- Creating a Streamlit dashboard
- Saving a complete Scikit-learn Pipeline instead of separate preprocessing objects

---

## Conclusion

This project demonstrates a complete beginner-to-intermediate machine learning classification workflow for diabetes prediction.

The workflow includes:

- Data loading
- Exploratory data analysis
- Data preprocessing
- Categorical encoding
- Feature scaling
- Model training
- Hyperparameter tuning
- Model evaluation
- Model comparison
- Model saving

Among the tested models, the tuned SVM model provided the strongest overall performance based on accuracy, precision, recall, and F1-score balance.

This project can be extended in the future with additional models, more advanced evaluation techniques, and deployment-oriented improvements.

