# Customer Churn Prediction using Artificial Neural Networks (ANN)

## 📌 Project Overview

Customer churn refers to the situation where customers stop using a company's products or services. Predicting customer churn helps businesses identify customers who are likely to leave and take proactive steps to retain them.

This project builds a **Binary Classification model using an Artificial Neural Network (ANN)** to predict whether a bank customer will exit or continue their relationship with the bank.

The model uses customer-related information such as age, credit score, balance, geography, gender, and other banking attributes to predict churn.

## 🎯 Objectives

- Understand and preprocess customer churn data.
- Perform exploratory data analysis.
- Convert categorical features into numerical representations.
- Apply feature scaling using StandardScaler.
- Build an Artificial Neural Network using TensorFlow and Keras.
- Train the model with validation monitoring.
- Evaluate model performance using classification metrics.

## 📂 Dataset

**Source:** [Credit Card Customer Churn Prediction – Kaggle](https://www.kaggle.com/datasets/rjmanoj/credit-card-customer-churn-prediction)

The dataset contains customer information and a target variable named `Exited`.

| Target value | Meaning |
|---|---|
| 0 | Customer did not exit |
| 1 | Customer exited |

The target variable makes this a binary classification problem.

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- TensorFlow
- Keras
- KaggleHub
- Google Colab

## ⚙️ Project Workflow

### 1. Dataset Loading

Used KaggleHub to download the dataset directly into the notebook environment and Pandas to load the CSV file.

### 2. Exploratory Data Analysis (EDA)

Inspected the dataset using:

- `df.head()` – Preview the dataset.
- `df.info()` – Understand column names and data types.
- `df.shape` – Check dataset dimensions.
- `df['Exited'].value_counts()` – Examine target class distribution.

### 3. Data Preprocessing

Removed unnecessary columns:

- `RowNumber`
- `CustomerId`
- `Surname`

Applied one-hot encoding to categorical features:

- `Geography`
- `Gender`

Used `drop_first=True` to avoid redundant dummy variables.

### 4. Feature and Target Separation

Separated the independent variables and dependent variable:

```python
X = df.drop(columns=['Exited'])
y = df['Exited']
```

- `X`: Input features.
- `y`: Target variable.

### 5. Train-Test Split

Split the dataset into training and testing sets using Scikit-learn.

```python
train_test_split(
    X, y,
    test_size=0.2,
    random_state=1
)
```

- 80% training data.
- 20% testing data.

### 6. Feature Scaling

Applied StandardScaler to normalize the numerical feature distributions.

```python
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

The scaler was fitted only on the training data to prevent data leakage.

### 7. Artificial Neural Network Architecture

Built a feedforward neural network using Keras Sequential API.

| Layer | Neurons | Activation |
|---|---:|---|
| Input | 11 features | — |
| Hidden Layer 1 | 11 | ReLU |
| Hidden Layer 2 | 11 | ReLU |
| Output Layer | 1 | Sigmoid |

**Activation functions:**

- ReLU: Introduces non-linearity in hidden layers.
- Sigmoid: Produces an output probability between 0 and 1 for binary classification.

### 8. Model Compilation

The model uses:

| Parameter | Configuration |
|---|---|
| Optimizer | Adam |
| Loss Function | Binary Crossentropy |
| Evaluation Metric | Accuracy |

### 9. Model Training

The model is trained using scaled training features.

Validation data is used to monitor the model's performance during training.

EarlyStopping can be used to stop training when validation loss stops improving and restore the best model weights.

### 10. Model Evaluation

The trained model generates probability predictions on the test dataset.

A threshold of 0.5 is used to convert probabilities into binary predictions.

Evaluation metrics include:

- Accuracy
- Confusion Matrix
- Precision
- Recall
- F1-score
- ROC-AUC

Training and validation loss and accuracy can also be visualized using Matplotlib.

## 📊 Model Evaluation

The model is evaluated on unseen test data.

The following metrics are considered:

- **Accuracy:** Overall proportion of correct predictions.
- **Precision:** Proportion of predicted churn cases that are actually churn cases.
- **Recall:** Proportion of actual churn cases correctly identified.
- **F1-score:** Harmonic mean of precision and recall.
- **ROC-AUC:** Measures the model's ability to distinguish between the two classes.

Since the dataset has an imbalanced target distribution, accuracy alone may not provide a complete picture of model performance.

## 📈 Visualizations

The project includes:

1. Training Loss
2. Training Accuracy
3. Validation Loss
4. Validation Accuracy
5. Confusion Matrix

These visualizations help understand model learning and identify potential overfitting.

## 🚀 How to Run

### Option 1: Google Colab

1. Open [Google Colab](https://colab.research.google.com/).
2. Upload the notebook or open it from your Google Drive.
3. Run the cells sequentially.
4. Install the required dependencies when prompted.
5. Train and evaluate the ANN model.

### Option 2: Local Environment

Clone the repository:

```bash
git clone <your-repository-url>
cd customer-churn-prediction
```

Install dependencies:

```bash
pip install pandas numpy matplotlib scikit-learn tensorflow kagglehub
```

Run the notebook or Python script in your environment.

## 🔮 Future Improvements

- Handle class imbalance using class weights or resampling techniques.
- Experiment with different ANN architectures and hyperparameters.
- Compare ANN performance with Logistic Regression, Random Forest and XGBoost.
- Tune the classification threshold to improve churn detection.
- Add a customer churn prediction interface using Streamlit.
- Save the trained model and scaler for deployment.

## 📚 Key Learnings

Through this project, I explored:

- End-to-end machine learning workflows.
- Data preprocessing and categorical encoding.
- Training and testing data separation.
- Feature scaling and data leakage prevention.
- Artificial Neural Network architecture.
- Activation functions and binary crossentropy.
- Model validation and overfitting.
- Classification evaluation metrics.

## 👨‍💻 Author

**Amarthya Sabu**

B.Tech Computer Science and Engineering Student

---

*This project was developed as a practical exploration of Artificial Neural Networks and binary classification using TensorFlow and Scikit-learn.*
