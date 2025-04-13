
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

# 📘 **Week 2: Exploratory Data Analysis (EDA) & Statistics**

---

## 📍 **Lesson 1: Descriptive Statistics**

### 📊 What is Descriptive Statistics?
Descriptive stats help **summarize** and **understand** your dataset.

---

### 🧮 Key Metrics:

| Metric        | Description                        | Pandas Code              |
|---------------|------------------------------------|---------------------------|
| **Mean**      | Average value                      | `df['col'].mean()`       |
| **Median**    | Middle value                       | `df['col'].median()`     |
| **Mode**      | Most frequent value                | `df['col'].mode()`       |
| **Variance**  | Measure of spread                  | `df['col'].var()`        |
| **Std Dev**   | Spread around mean (dispersion)    | `df['col'].std()`        |
| **Min/Max**   | Smallest & largest value           | `df['col'].min()/max()`  |
| **IQR**       | 75th - 25th percentile             | `Q3 - Q1` (manual calc)  |

---

### 🧪 Example:
```python
df = sns.load_dataset('tips')
df['total_bill'].mean()
df['tip'].std()
```

🧩 **Try it**: What’s the median tip for male vs. female customers?

---

## 📍 **Lesson 2: Correlation and Covariance**

### 🔗 What is Correlation?

- Measures **relationship** between two variables
- Value between `-1` and `1`

```python
df.corr()  # Pairwise correlation
```

✅ Use `.corr()` for understanding strength of relationships.

---

### 🔍 Heatmap of Correlation

```python
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()
```

---

### 🔗 Covariance

- Tells **direction** of the relationship, but not strength
- Less interpretable than correlation

```python
df.cov()
```

---

## 📍 **Lesson 3: Introduction to EDA**

> **EDA** = Exploring your data before applying models.

---

### ✅ Steps in EDA:

1. Load the data
2. Understand structure and types
3. Handle missing values
4. Check outliers
5. Visualize data patterns

---

### 🔍 Key Functions:

```python
df.info()
df.describe()
df.isnull().sum()
df.duplicated().sum()
```

🧩 **Try it**: Run full EDA on the Titanic dataset

---

## 📍 **Lesson 4: Data Visualization for EDA**

### 🔸 Count Plot

```python
sns.countplot(x='sex', data=df)
```

### 🔸 Histogram

```python
sns.histplot(df['total_bill'], kde=True)
```

### 🔸 Box Plot (for outliers)

```python
sns.boxplot(x='day', y='total_bill', data=df)
```

### 🔸 Violin Plot

```python
sns.violinplot(x='day', y='tip', data=df)
```

---

## 📍 **Lesson 5: Advanced Visualization with Seaborn & Matplotlib**

### 🔹 Pair Plot

```python
sns.pairplot(df)
```

### 🔹 Joint Plot

```python
sns.jointplot(x='total_bill', y='tip', data=df, kind='hex')
```

### 🔹 Swarm Plot

```python
sns.swarmplot(x='day', y='tip', data=df)
```

---

## 📈 Matplotlib Extras

```python
# Multiple subplots
plt.subplot(1, 2, 1)
plt.hist(df['tip'])

plt.subplot(1, 2, 2)
plt.boxplot(df['total_bill'])
plt.show()
```

---

## 🎯 Week 2 Mini Project

Use **Titanic** or **Iris dataset**:

1. Calculate mean, median, mode for key columns  
2. Visualize correlation heatmap  
3. Use boxplots and histograms to show insights  
4. Summarize EDA findings

---

## ✅ Week 2 Checklist:

✅ Understand descriptive statistics  
✅ Perform correlation and interpret it  
✅ Use boxplots, histograms, violin plots  
✅ Apply full EDA process to any dataset  
✅ Identify insights visually before modeling

---



---

# 📘 **Week 3: Data Cleaning and Preprocessing**

---

## 📍 **Lesson 1: Handling Missing Values and Duplicates**

---

### 🔎 Detecting Missing Values

```python
df.isnull().sum()
```

### 🧽 Handling Missing Data

| Method               | When to Use                           | Code Example |
|----------------------|----------------------------------------|--------------|
| **Drop Rows**        | If missing data is small in number     | `df.dropna()` |
| **Fill with Mean**   | For numerical columns                  | `df['col'].fillna(df['col'].mean())` |
| **Fill with Mode**   | For categorical columns                | `df['col'].fillna(df['col'].mode()[0])` |
| **Custom Fill**      | If domain knowledge helps              | `df['col'].fillna(0)` |

---

### 🔁 Handling Duplicates

```python
df.duplicated().sum()       # Check
df.drop_duplicates(inplace=True)   # Drop
```

🧩 **Try it**: Clean a dataset with at least 2 missing columns and duplicates.

---

## 📍 **Lesson 2: Detecting and Treating Outliers**

---

### 🧰 Method 1: IQR (Interquartile Range)

```python
Q1 = df['col'].quantile(0.25)
Q3 = df['col'].quantile(0.75)
IQR = Q3 - Q1

outliers = df[(df['col'] < Q1 - 1.5 * IQR) | (df['col'] > Q3 + 1.5 * IQR)]
```

---

### 🧰 Method 2: Z-Score

```python
from scipy import stats
z_scores = stats.zscore(df['col'])
df[abs(z_scores) > 3]
```

---

### 🎯 Handling Outliers

- Remove: `df = df[df['col'] < upper_limit]`
- Cap/Floor: Replace with percentiles
- Transform: Use log or square root

🧩 **Task**: Visualize outliers using `sns.boxplot()` and treat them.

---

## 📍 **Lesson 3: Encoding Categorical Data**

---

### 🔡 Why Encoding?

> Machine learning models only understand **numbers** – not text.

---

### ✅ One-Hot Encoding (for Nominal Categories)

```python
pd.get_dummies(df, columns=['gender', 'smoker'], drop_first=True)
```

---

### ✅ Label Encoding (for Ordinal Categories)

```python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['education_level'] = le.fit_transform(df['education_level'])
```

🧩 **Tip**: Use One-Hot for unordered categories, LabelEncoder for ordered ones.

---

## 📍 **Lesson 4: Feature Scaling: Normalization & Standardization**

---

### 🔢 Why Scaling?

> To make all features comparable in scale for better model performance.

---

### 📏 Normalization (Min-Max Scaling)

```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df[['income', 'age']])
```

Result: Values between **0 and 1**

---

### 📐 Standardization (Z-score Scaling)

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[['income', 'age']])
```

Result: Mean = 0, Std = 1

🧩 **Mini Task**: Try both on a dataset and compare with a histogram.

---

## 📍 **Lesson 5: Data Transformation: Pivot, Melt, Reshape**

---

### 🔄 Pivot Table

```python
df.pivot_table(values='sales', index='region', columns='month', aggfunc='sum')
```

---

### 🔃 Melt (Wide → Long)

```python
pd.melt(df, id_vars=['id'], var_name='feature', value_name='value')
```

---

### 🔁 Reshaping with `.stack()` and `.unstack()`

```python
df.set_index(['A', 'B']).unstack()
```

🧩 **Tip**: Use this for time-series and multi-level grouped data.

---

## 📍 **Lesson 6: Preprocessing Text Data**

---

### 🔤 Basic Text Cleaning

```python
text = "This is SO COOL!! 😎"
cleaned = text.lower().replace("!", "")
```

---

### 🪄 Tokenization, Stopwords, Lemmatization

```python
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

tokens = word_tokenize(text)
tokens = [t for t in tokens if t not in stopwords.words('english')]

lemmatizer = WordNetLemmatizer()
tokens = [lemmatizer.lemmatize(t) for t in tokens]
```

🧩 **Optional**: Use `CountVectorizer()` or `TfidfVectorizer()` for model-ready features.

---

## 🎯 Week 3 Mini Project

1. Load a messy dataset (Titanic, housing, or customer data)
2. Clean missing values and duplicates
3. Handle outliers in numerical data
4. Encode categorical columns
5. Scale numerical features
6. Create pivot or reshape data
7. If text exists, clean and tokenize it

---

## ✅ Week 3 Checklist:

✅ Handle missing values  
✅ Remove or treat duplicates  
✅ Detect & treat outliers  
✅ Encode categorical features  
✅ Normalize & standardize numerical features  
✅ Transform wide vs. long data  
✅ Clean basic text for NLP

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

