# =====================================================
# DecodeLabs Project 2
# Iris Flower Classification using KNN
# Author: Talha Hanif
# =====================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# =====================================================
# LOAD DATASET
# =====================================================

print("=" * 60)
print("IRIS FLOWER CLASSIFICATION USING KNN")
print("=" * 60)

iris = load_iris()

X = iris.data
y = iris.target

df = pd.DataFrame(
    X,
    columns=iris.feature_names
)

df["Species"] = y

print("\nDataset Shape:")
print(df.shape)

print("\nFirst Five Records:")
print(df.head())

print("\nClass Distribution:")
print(df["Species"].value_counts())

# =====================================================
# DATA VISUALIZATION
# =====================================================

species_names = {
    0: "Setosa",
    1: "Versicolor",
    2: "Virginica"
}

plot_df = df.copy()
plot_df["Species"] = plot_df["Species"].map(species_names)

# Pair Plot
sns.pairplot(
    plot_df,
    hue="Species",
    diag_kind="hist"
)

plt.savefig("pairplot.png")
plt.close()

print("\nPairplot saved as pairplot.png")

# =====================================================
# TRAIN TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))

# =====================================================
# FEATURE SCALING
# =====================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("\nFeature Scaling Applied Successfully")

# =====================================================
# FIND BEST K USING CROSS VALIDATION
# =====================================================

print("\nSearching for Best K Value...\n")

best_k = 1
best_score = 0

k_scores = []

for k in range(1, 21):

    model = KNeighborsClassifier(n_neighbors=k)

    scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=5
    )

    avg_score = scores.mean()

    k_scores.append(avg_score)

    print(f"K = {k:<2} | Accuracy = {avg_score:.4f}")

    if avg_score > best_score:
        best_score = avg_score
        best_k = k

print("\nBest K Found:", best_k)
print("Best Cross Validation Accuracy:", round(best_score * 100, 2), "%")

# =====================================================
# VISUALIZE K VS ACCURACY
# =====================================================

plt.figure(figsize=(8, 5))

plt.plot(
    range(1, 21),
    k_scores,
    marker="o"
)

plt.title("Choosing Optimal K")
plt.xlabel("K Value")
plt.ylabel("Cross Validation Accuracy")

plt.grid(True)

plt.savefig("best_k_graph.png")
plt.close()

print("\nBest K graph saved as best_k_graph.png")

# =====================================================
# TRAIN FINAL MODEL
# =====================================================

final_model = KNeighborsClassifier(
    n_neighbors=best_k
)

final_model.fit(
    X_train,
    y_train
)

# =====================================================
# PREDICTIONS
# =====================================================

predictions = final_model.predict(X_test)

# =====================================================
# EVALUATION METRICS
# =====================================================

accuracy = accuracy_score(
    y_test,
    predictions
)

precision = precision_score(
    y_test,
    predictions,
    average="weighted"
)

recall = recall_score(
    y_test,
    predictions,
    average="weighted"
)

f1 = f1_score(
    y_test,
    predictions,
    average="weighted"
)

print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")

# =====================================================
# CLASSIFICATION REPORT
# =====================================================

print("\nClassification Report")
print("-" * 60)

print(
    classification_report(
        y_test,
        predictions,
        target_names=iris.target_names
    )
)

# =====================================================
# CONFUSION MATRIX
# =====================================================

cm = confusion_matrix(
    y_test,
    predictions
)

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=iris.target_names,
    yticklabels=iris.target_names
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

plt.savefig(
    "confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Confusion Matrix saved as confusion_matrix.png")

# =====================================================
# FEATURE DISTRIBUTION
# =====================================================

plt.figure(figsize=(10, 6))

df.iloc[:, 0:4].mean().plot(
    kind="bar"
)

plt.title("Average Feature Values")
plt.ylabel("Value")

plt.savefig(
    "feature_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Feature Distribution graph saved as feature_distribution.png")

# =====================================================
# SAMPLE PREDICTIONS
# =====================================================

print("\nSample Predictions")

for i in range(5):

    actual = iris.target_names[y_test[i]]
    predicted = iris.target_names[predictions[i]]

    print(
        f"Sample {i+1}: "
        f"Actual = {actual}, "
        f"Predicted = {predicted}"
    )

# =====================================================
# END
# =====================================================

print("\nProject Completed Successfully!")
print("Generated Files:")
print("1. pairplot.png")
print("2. best_k_graph.png")
print("3. confusion_matrix.png")
print("4. feature_distribution.png")