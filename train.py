import pandas as pd
import numpy as np
import os
import glob
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_score,
    f1_score,
    confusion_matrix
)

# ---------- Dynamic File Loading ----------
data_dir = r'OriginalRedditData\raw_data\2022'

# Collect all CSV files from all month folders
all_files = []
for month_dir in os.listdir(data_dir):
    month_path = os.path.join(data_dir, month_dir)
    if os.path.isdir(month_path):
        csv_files = glob.glob(os.path.join(month_path, '*.csv'))
        all_files.extend(csv_files)

print(f"📂 Found {len(all_files)} CSV files to analyze")

# ---------- Load and Label ----------
def infer_subreddit_from_filename(filename):
    name = os.path.basename(filename).lower()
    if "anx" in name:
        return "Anxiety"
    elif "dep" in name:
        return "Depression"
    elif "lon" in name:
        return "Loneliness"
    elif "mh" in name and "sw" not in name:
        return "Mentalhealth"
    elif "sw" in name or "suicide" in name:
        return "SuicideWatch"
    else:
        return "Unknown"

def load_and_label(path):
    df = pd.read_csv(path)
    df = df[["title", "selftext"]].dropna(subset=["selftext"])
    df["text"] = df["title"].fillna("") + " " + df["selftext"].fillna("")
    df["subreddit"] = infer_subreddit_from_filename(path)
    return df[["text", "subreddit"]]

# ---------- Process and Combine ----------
all_data = []
for file_path in all_files:
    try:
        df = load_and_label(file_path)
        if df["subreddit"].iloc[0] != "Unknown":
            all_data.append(df)
    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")

df_all = pd.concat(all_data, ignore_index=True)

# ---------- Check class counts ----------
print("\n📊 Instance count per subreddit:")
print(df_all["subreddit"].value_counts())

# ---------- Label Encoding ----------
le = LabelEncoder()
df_all["label"] = le.fit_transform(df_all["subreddit"])
X = df_all["text"].values
y = df_all["label"].values
labels = le.classes_

# ---------- Split Data ----------
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42, stratify=y
)

# ---------- Pipeline and Grid Search ----------
custom_stop_words = [
    "said", "t", "see", "i", "im", "m", "like", "get", "s", "go", "lot", "got", "to", "and", "the", "a",
    "put", "did", "went", "didn", "don", "ll", "something", "thing", "of", "that", "in", "but", "for", "have", "is",
    "https", "http", "re", "ve", "etc", "ism", "you", "it", "we", "they", "he", "she",
    "my", "me", "us", "them", "him", "her", "ours", "yours", "their", "theirs", "ourselves",
    "just", "this", "so", "with", "was", "be", "feel", "can", "not", "on", "do", "or", "about", "all", "at",
    "because", "what", "been", "up", "as", "when", "am", "even", "had", "from", "are"
]

# English stop words'ü frozenset'ten listeye dönüştür
english_stop_words = list(TfidfVectorizer(stop_words="english").get_stop_words())

# Birleştir ve tekrar set'e çevirerek tekrar list'e dönüştür (çiftleri engellemek için)
combined_stop_words = list(set(custom_stop_words + english_stop_words))

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words=combined_stop_words)),
    ("clf", LogisticRegression(max_iter=1000 , class_weight="balanced"))
])


param_grid = {
    "tfidf__max_df": [0.75, 1.0],
    "tfidf__ngram_range": [(1, 1), (1, 2)],
    "clf__C": [0.01, 0.1, 1, 10],
    "clf__penalty": ["l2"],
    "clf__solver": ["liblinear"]
}

for idx, label in enumerate(labels):
    print(f"Class {idx}: {label}")



print("\n🔍 Starting Grid Search for Logistic Regression...")
grid = GridSearchCV(
    pipeline,
    param_grid,
    cv=4,
    scoring="f1_weighted",
    verbose=2,
    n_jobs=-1
)
grid.fit(X_train_val, y_train_val)

# ---------- Best Parameters ----------
print("\n✅ Best Parameters Found:")
print(grid.best_params_)
print(f"🎯 Best Validation (CV) F1 Score: {grid.best_score_:.4f}")

# ---------- Top 5 Hyperparameter Results ----------
cv_results = pd.DataFrame(grid.cv_results_)
print("\n📋 Top 5 Grid Search Results:")
print(cv_results[["params", "mean_test_score"]]
      .sort_values(by="mean_test_score", ascending=False)
      .head(5)
      .to_string(index=False))

# ---------- Final Evaluation ----------
y_test_pred = grid.predict(X_test)
print("\n📊 Final Test Set Evaluation:")
print(classification_report(y_test, y_test_pred, target_names=labels))

# ---------- Final Metrics ----------
acc = accuracy_score(y_test, y_test_pred)
prec = precision_score(y_test, y_test_pred, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_test_pred, average='weighted', zero_division=0)

print(f"\nFinal Accuracy:  {acc:.4f}")
print(f"Final Precision: {prec:.4f}")
print(f"Final F1-Score:  {f1:.4f}")

# ---------- Confusion Matrix ----------
cm = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.title("Confusion Matrix: Logistic Regression")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()

# ---------- Save Model ----------
joblib.dump(grid.best_estimator_, "logistic_model_tuned2.pkl")
print("\n💾 Model saved as 'logistic_model_tuned2.pkl'")
