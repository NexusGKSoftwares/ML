
---

# 📘 **Week 1: Introduction to Data Analytics**

---

## 📍 **Lesson 1: Introduction to Data Analytics**

### ✅ What is Data Analytics?

> Data Analytics is the process of examining, cleaning, transforming, and modeling data to discover useful information, draw conclusions, and support decision-making.

---

### 💡 Types of Data Analytics:

| Type | Description | Example |
|------|-------------|---------|
| **Descriptive** | What happened? | Sales report for last month |
| **Diagnostic** | Why did it happen? | Sales dropped due to bad marketing |
| **Predictive** | What will happen? | Forecast future sales |
| **Prescriptive** | What should we do? | Optimize pricing for best revenue |

---

### 🌍 Real-World Applications

- **Healthcare**: Predict patient readmission
- **Finance**: Detect fraud
- **Marketing**: Customer segmentation
- **Retail**: Inventory forecasting
- **Sports**: Player performance analysis

🧠 **Quick Task**: Think of 2 ways data is used in your industry.

---

## 📍 **Lesson 2: Introduction to Python & Development Environment Setup**

### ✅ Why Python?

- Easy to learn and read
- Huge community support
- Libraries for data science: `Pandas`, `NumPy`, `Matplotlib`, `Seaborn`, `Scikit-learn`

---

### 🔧 Development Environment Setup

**Option 1: Install via Anaconda (Recommended)**
- Download from: [https://www.anaconda.com](https://www.anaconda.com)
- Comes with Python, Jupyter Notebook, libraries

**Option 2: Manual Setup**
```bash
Install Python: https://www.python.org  
Install VS Code: https://code.visualstudio.com  
Install Libraries:
pip install pandas numpy matplotlib seaborn jupyterlab
```

---

### ✅ First Python Program

```python
print("Hello, Data Analytics!")
```

---

## 📍 **Lesson 3: Introduction to Pandas, NumPy, Matplotlib, Seaborn**

---

### 🧾 **Pandas** – Work with structured data (CSV, Excel, SQL, etc.)

```python
import pandas as pd

# Load CSV
df = pd.read_csv('sample.csv')

# Preview
df.head()
```

---

### 🔢 **NumPy** – Work with arrays & numerical operations

```python
import numpy as np

arr = np.array([1, 2, 3])
print(arr.mean(), arr.max(), arr.min())
```

---

### 📊 **Matplotlib** – Basic plotting

```python
import matplotlib.pyplot as plt

x = [1, 2, 3]
y = [4, 1, 6]
plt.plot(x, y)
plt.title("Line Chart")
plt.show()
```

---

### 🎨 **Seaborn** – Beautiful statistical plots

```python
import seaborn as sns
df = sns.load_dataset("tips")
sns.histplot(df['total_bill'])
```

---

## 📍 **Lesson 4: Data Manipulation with Pandas**

---

### 🔍 DataFrame Basics

```python
df.info()       # Structure
df.describe()   # Summary statistics
df.columns      # Column names
df.shape        # Rows x Columns
```

---

### 🎯 Indexing and Filtering

```python
df['total_bill']           # Access a column
df[['total_bill', 'tip']]  # Access multiple columns
df[df['tip'] > 5]          # Filter rows
```

---

### 🧱 Add / Remove Columns

```python
df['tip_percentage'] = df['tip'] / df['total_bill'] * 100
df.drop('column_name', axis=1, inplace=True)
```

---

### 🔄 Sorting & Aggregation

```python
df.sort_values(by='total_bill', ascending=False)

df.groupby('sex')['tip'].mean()
```

🧩 **Mini Task**: Show average tip per day.

---

## 📍 **Lesson 5: Basic Visualization with Matplotlib & Seaborn**

---

### 📈 Matplotlib Examples

```python
# Line Chart
plt.plot(df['total_bill'])
plt.title("Total Bill Over Index")
plt.show()

# Bar Chart
df['day'].value_counts().plot(kind='bar')
plt.title("Count by Day")
plt.show()
```

---

### 🎨 Seaborn Examples

```python
# Scatter Plot
sns.scatterplot(x='total_bill', y='tip', data=df)

# Boxplot
sns.boxplot(x='day', y='tip', data=df)

# Heatmap
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
```

---

## 🎯 Week 1 Mini Project

Choose a dataset like Titanic, Tips, or Iris:

1. Load the dataset using Pandas
2. Display structure, types, summary
3. Add new columns or filter data
4. Visualize insights using Matplotlib/Seaborn

---

## 📚 Suggested Datasets:

- [Titanic](https://www.kaggle.com/c/titanic)
- [Tips (Seaborn)](https://github.com/mwaskom/seaborn-data)
- [Iris](https://archive.ics.uci.edu/ml/datasets/Iris)
- [CSV Playground](https://people.sc.fsu.edu/~jburkardt/data/csv/csv.html)

---

## ✅ Week 1 Checklist:

✅ Know what Data Analytics is  
✅ Install Python and Jupyter/VS Code  
✅ Load and inspect data with Pandas  
✅ Manipulate data with filtering, sorting, new columns  
✅ Visualize data using Matplotlib and Seaborn

---






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

