
---

# 🎓 Week 4: Introduction to Machine Learning & Final Project

---

## 🧠 1. Introduction to Machine Learning

### What is Machine Learning?

> **Machine Learning (ML)** is a branch of artificial intelligence (AI) focused on building systems that learn from data and improve automatically without being explicitly programmed.

### Key Categories:

| Type | Description | Example |
|------|-------------|---------|
| **Supervised Learning** | Learns from labeled data (input + output) | Predicting house prices |
| **Unsupervised Learning** | Finds hidden patterns in unlabeled data | Customer segmentation |
| **Reinforcement Learning** | Learns by interacting with environment | Game AI, robotics |

---

### Supervised Learning

- **Inputs**: Features (X)
- **Outputs**: Labels (Y)
- Goal: Learn a mapping from X to Y

#### Tasks:
- **Regression**: Predict continuous values (e.g., price, salary)
- **Classification**: Predict discrete labels (e.g., spam/not spam)

---

### Unsupervised Learning

- **No labels**
- Goal: Group data or reduce dimensions

#### Tasks:
- **Clustering** (e.g., K-Means)
- **Dimensionality Reduction** (e.g., PCA)

---

## 📈 2. Linear Regression & Logistic Regression

---

### 🔹 Linear Regression (Regression Problem)

**Use Case**: Predict numerical outcomes  
**Equation**: 𝑦 = 𝑚𝑥 + 𝑏

```python
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Sample data
df = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv')

# Features and label
X = df[['total_bill']]
y = df['tip']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction & Evaluation
y_pred = model.predict(X_test)
print("MSE:", mean_squared_error(y_test, y_pred))
```

---

### 🔸 Logistic Regression (Classification Problem)

**Use Case**: Binary classification  
**Output**: Probability between 0 and 1

```python
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.metrics import classification_report

# Load dataset
iris = load_iris()
X = iris.data
y = (iris.target == 2).astype(int)  # Binary classification

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict & Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
```

---

## 🧪 3. Model Evaluation Metrics

| Metric | Formula | Use Case |
|--------|---------|----------|
| **Accuracy** | (TP+TN)/(TP+FP+TN+FN) | Balanced classes |
| **Precision** | TP/(TP+FP) | Minimize false positives |
| **Recall** | TP/(TP+FN) | Minimize false negatives |
| **F1 Score** | 2*(P*R)/(P+R) | Imbalanced data |

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
```

---

## 🔧 4. Implementing Models with Scikit-Learn

### ML Workflow:
1. Import libraries
2. Load dataset
3. Clean/preprocess data
4. Split into training/testing
5. Train model
6. Predict
7. Evaluate

---

### Example: Classification with RandomForest

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load and split
data = load_wine()
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Train
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Predict & Evaluate
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
```

---

## 💼 5. Final Project Guidelines

### 🎯 Project Goal:
Apply the entire pipeline:  
**Data Cleaning ➝ EDA ➝ Model Building ➝ Evaluation ➝ Presentation**

---

### 🔹 Recommended Dataset:
**Titanic Dataset** (binary classification)  
Source: [Kaggle Titanic](https://www.kaggle.com/c/titanic)

---

### 🔧 Step-by-Step:
#### Step 1: Load and Clean Data
```python
import pandas as pd
df = pd.read_csv("titanic.csv")
df = df.drop(['Cabin', 'Ticket', 'Name'], axis=1)
df['Age'].fillna(df['Age'].mean(), inplace=True)
df['Embarked'].fillna('S', inplace=True)
```

#### Step 2: Feature Engineering
```python
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)
```

#### Step 3: Model Training
```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

X = df.drop('Survived', axis=1)
y = df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
```

---

## 🎤 6. Final Presentation & Wrap-Up

### Slide Content Suggestions:
- ✅ Project Title
- 🧠 Problem Statement
- 📊 Dataset Description
- 🧹 Data Cleaning Process
- 📉 Visualizations (charts/graphs)
- 🤖 Model Chosen & Why
- 📈 Evaluation Metrics
- 💡 Insights & Takeaways

---

