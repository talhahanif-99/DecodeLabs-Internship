# DecodeLabs-Internship
# Iris Flower Classification using K-Nearest Neighbors (KNN)

## Project Overview

This project was developed as part of the **DecodeLabs Artificial Intelligence Internship - Project 2: Data Classification Using AI**.

The objective is to build a machine learning model capable of classifying Iris flowers into different species based on their physical characteristics. The project demonstrates the complete supervised learning workflow, including data preprocessing, model training, hyperparameter tuning, evaluation, and visualization.

---

## Objectives

* Load and analyze a real-world dataset
* Perform data preprocessing and feature scaling
* Split data into training and testing sets
* Train a K-Nearest Neighbors (KNN) classification model
* Optimize model performance using Cross-Validation
* Evaluate the model using multiple metrics
* Visualize results using graphs and plots

---

## Dataset Information

The project uses the famous **Iris Dataset**, which contains:

* 150 flower samples
* 4 input features:

  * Sepal Length
  * Sepal Width
  * Petal Length
  * Petal Width
* 3 target classes:

  * Setosa
  * Versicolor
  * Virginica

Dataset Source: Scikit-Learn Built-in Dataset

---

## Technologies Used

* Python 3.12
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-Learn

---

## Project Structure

```text
DecodeLabs-Internship/
│
├── iris_ai_classifier.py
├── README.md
├── requirements.txt
├── pairplot.png
├── best_k_graph.png
├── confusion_matrix.png
└── feature_distribution.png
```

---

## Machine Learning Workflow

### 1. Data Loading

The Iris dataset is loaded using Scikit-Learn.

### 2. Data Exploration

Basic dataset information, class distribution, and sample records are displayed.

### 3. Data Visualization

A pairplot is generated to visualize relationships between features.

### 4. Train-Test Split

The dataset is divided into:

* 80% Training Data
* 20% Testing Data

### 5. Feature Scaling

StandardScaler is used to normalize feature values.

### 6. Hyperparameter Tuning

Cross-validation is performed to determine the optimal K value for KNN.

### 7. Model Training

A K-Nearest Neighbors classifier is trained using the best K value.

### 8. Model Evaluation

The model is evaluated using:

* Accuracy Score
* Precision Score
* Recall Score
* F1 Score
* Confusion Matrix
* Classification Report

---

## Generated Outputs

The program automatically generates:

### Pair Plot

Visualizes relationships among all features.

### Best K Graph

Shows cross-validation accuracy for different K values.

### Confusion Matrix

Displays prediction performance for each class.

### Feature Distribution Graph

Shows average feature values across the dataset.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/DecodeLabs-Internship.git
```

Move into the project folder:

```bash
cd DecodeLabs-Internship
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the project:

```bash
python iris_ai_classifier.py
```

---

## Requirements

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
```

---

## Sample Results

The model typically achieves accuracy between **95% and 100%** depending on the train-test split and optimal K value selected through cross-validation.

---

## Learning Outcomes

Through this project, I learned:

* Supervised Learning Fundamentals
* Data Preprocessing Techniques
* Feature Scaling
* Classification Algorithms
* Hyperparameter Tuning
* Cross Validation
* Model Evaluation Metrics
* Data Visualization
* Machine Learning Workflow

---

## Author

**Talha Hanif**

Artificial Intelligence Intern at DecodeLabs

---

## License

This project is created for educational and internship learning purposes.
