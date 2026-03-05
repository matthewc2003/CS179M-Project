import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
# roc_auc_score is a common metric for evaluating binary classification models, especially when dealing with imbalanced datasets. It measures the model's ability to distinguish between the two classes across all possible classification thresholds.
# source for roc_auc_score: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib

df = pd.read_csv("Data/prepro_data/nhanes_diet_risk.csv")

# Separate features, target, and weights
X = df.drop(columns=["SEQN", "Obese", "WTDRD1"])
y = df["Obese"]
weights = df["WTDRD1"]

# normalize weights for numerical stability
weights = weights / weights.mean()

# Train/test split
X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
    X, y, weights,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Build pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("logreg", LogisticRegression(max_iter=1000)) # We aren't doing class weight balancing since we have survey weights, which should already account for class imbalance in the population.
])

# Fit with survey weights
pipeline.fit(
    X_train,
    y_train,
    logreg__sample_weight=w_train
)

# Predictions and evaluation
y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]

# Weighted evaluation
print("Weighted Classification Report:")
print(classification_report(y_test, y_pred, sample_weight=w_test))

print("Weighted ROC AUC:",
      roc_auc_score(y_test, y_prob, sample_weight=w_test))

# Coefficients
feature_names = X.columns
coefficients = pipeline.named_steps["logreg"].coef_[0]

print("\nModel Coefficients:")
for name, coef in zip(feature_names, coefficients):
    print(f"{name}: {coef:.4f}")

# Save model
joblib.dump(pipeline, "Data/prepro_data/diet_risk_model.pkl")

print("Model saved to Data/prepro_data/diet_risk_model.pkl")