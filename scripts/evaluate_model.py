import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    classification_report
)

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from data_loader import load_dataset
from preprocessing import preprocess


# -------------------------------------------------
# Load Dataset
# -------------------------------------------------
df = load_dataset()

print("\nClass Distribution:")
print(df["label"].value_counts())


# -------------------------------------------------
# SPEED OPTIMIZATION (Use smaller sample)
# -------------------------------------------------
df = df.sample(20000, random_state=42)

print("\nUsing Sample Size:", len(df))


# -------------------------------------------------
# Feature Correlation Heatmap
# -------------------------------------------------
numeric_df = df.select_dtypes(include=['number'])

corr = numeric_df.iloc[:, :15].corr()

plt.figure(figsize=(10,8))
sns.heatmap(corr, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()


# -------------------------------------------------
# Preprocess Data
# -------------------------------------------------
X_train, X_test, y_train, y_test = preprocess(df)


# -------------------------------------------------
# Define Models
# -------------------------------------------------
models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=50,
        n_jobs=-1,
        random_state=42
    ),

    "SVM": SVC(
        kernel="linear",
        probability=True
    )
}

results = []
training_times = []


# -------------------------------------------------
# Train & Evaluate Models
# -------------------------------------------------
for name, model in models.items():

    print(f"\nTraining {name}...")

    start = time.time()

    model.fit(X_train, y_train)

    end = time.time()

    training_time = end - start
    training_times.append([name, training_time])

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:,1]

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    results.append([name, acc, prec, rec, f1])

    print("\nPerformance Metrics")
    print("Accuracy :", acc)
    print("Precision:", prec)
    print("Recall   :", rec)
    print("F1 Score :", f1)

    print("\nClassification Report")
    print(classification_report(y_test, preds))


    # -------------------------------------------------
    # Confusion Matrix
    # -------------------------------------------------
    cm = confusion_matrix(y_test, preds)

    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()


    # -------------------------------------------------
    # ROC Curve
    # -------------------------------------------------
    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0,1],[0,1],'--')
    plt.title(f"{name} ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.show()


    # -------------------------------------------------
    # Precision Recall Curve
    # -------------------------------------------------
    precision, recall, _ = precision_recall_curve(y_test, probs)
    ap = average_precision_score(y_test, probs)

    plt.figure()
    plt.plot(recall, precision, label=f"AP = {ap:.3f}")
    plt.title(f"{name} Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.show()


# -------------------------------------------------
# Model Comparison Table
# -------------------------------------------------

comparison = pd.DataFrame(
    results,
    columns=["Model", "Accuracy", "Precision", "Recall", "F1 Score"]
)

print("\n==============================")
print("MODEL COMPARISON TABLE")
print("==============================")
print(comparison.to_string(index=False))


# -------------------------------------------------
# Model Comparison Graph
# -------------------------------------------------

comparison_melt = comparison.melt(id_vars="Model",
                                  var_name="Metric",
                                  value_name="Score")

plt.figure(figsize=(8,5))
sns.barplot(data=comparison_melt, x="Metric", y="Score", hue="Model")

plt.title("Model Performance Comparison")
plt.ylim(0,1)
plt.show()

# -------------------------------------------------
# Training Time Comparison
# -------------------------------------------------
time_df = pd.DataFrame(
    training_times,
    columns=["Model","Training Time (seconds)"]
)

print("\nTraining Time Comparison")
print(time_df)

plt.figure(figsize=(6,4))
sns.barplot(x="Model", y="Training Time (seconds)", data=time_df)
plt.title("Training Time Comparison")
plt.show()


# -------------------------------------------------
# Feature Importance (Random Forest)
# -------------------------------------------------
rf_model = models["Random Forest"]

importances = rf_model.feature_importances_

# Create generic feature names to match importance length
feature_names = [f"Feature_{i}" for i in range(len(importances))]

feat_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
})

feat_df = feat_df.sort_values(by="Importance", ascending=False).head(15)

plt.figure(figsize=(10,6))
sns.barplot(x="Importance", y="Feature", data=feat_df)
plt.title("Top 15 Feature Importance (Random Forest)")
plt.show()