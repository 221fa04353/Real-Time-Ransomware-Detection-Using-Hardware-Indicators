import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from data_loader import load_dataset
from preprocessing import preprocess

df = load_dataset()
X_train, X_test, y_train, y_test = preprocess(df)

print("Training Random Forest...")

rf = RandomForestClassifier(
    n_estimators=150,
    n_jobs=-1   # use all CPU cores
)
rf.fit(X_train, y_train)
joblib.dump(rf, "../models/random_forest.pkl")

print("Random Forest completed.")

# ---------- FAST SVM (subset only) ----------
print("Training SVM on subset...")

subset = 20000
X_small = X_train[:subset]
y_small = y_train[:subset]

svm = SVC(probability=True)
svm.fit(X_small, y_small)

joblib.dump(svm, "../models/svm.pkl")

print("Models trained successfully")