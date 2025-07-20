import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import KFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
)

# Load and combine datasets
def load_and_label(path, name):
    df = pd.read_csv(path)
    df = df[["title", "selftext"]].dropna(subset=["selftext"])
    df["text"] = df["title"].fillna("") + " " + df["selftext"].fillna("")
    df["subreddit"] = name  # Overwrite with desired label
    return df[["text", "subreddit"]]

print("🔹 Loading datasets...")
df_anx = load_and_label(r"OriginalRedditData\raw_data\2022\May22\anximay22.csv", "Anxiety")
df_sw = load_and_label(r"OriginalRedditData\raw_data\2022\May22\Swmay22.csv", "SuicideWatch")
df_mh = load_and_label(r"OriginalRedditData\raw_data\2022\May22\mhmay22.csv", "Mentalhealth")
df_dep = load_and_label(r"OriginalRedditData\raw_data\2022\May22\depmay22.csv", "Depression")
df_lone = load_and_label(r"OriginalRedditData\raw_data\2022\May22\lonemay22.csv", "Loneliness")

df_all = pd.concat([df_anx, df_sw, df_mh, df_dep, df_lone])
df_all = df_all[df_all["subreddit"].isin(["Anxiety", "SuicideWatch", "Mentalhealth", "Depression", "Loneliness"])]


# Show instance counts per subreddit
print("\n📊 Instance count per subreddit:")
print(df_all["subreddit"].value_counts())

# Encode labels
le = LabelEncoder()
df_all["label"] = le.fit_transform(df_all["subreddit"])
X = df_all["text"].values
y = df_all["label"].values
labels = le.classes_

# Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Naive Bayes": MultinomialNB(),
    "Linear SVM": LinearSVC(),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5)
}

kf = KFold(n_splits=5, shuffle=True, random_state=42)
vectorizer = TfidfVectorizer(max_features=10000, stop_words="english")

results = {}

print("🔹 Starting model evaluation...\n")

for name, model in models.items():
    print(f"▶️  Training: {name}")
    start = time.time()
    pipeline = make_pipeline(vectorizer, model)
    y_pred = cross_val_predict(pipeline, X, y, cv=kf)
    end = time.time()

    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y, y_pred, average='weighted', zero_division=0)

    results[name] = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "time": end - start
    }

    print(f"    ✔️  Accuracy:  {acc:.4f}")
    print(f"    ✔️  Precision: {prec:.4f}")
    print(f"    ✔️  Recall:    {rec:.4f}")
    print(f"    ✔️  F1 Score:  {f1:.4f}")
    print(f"    ⏱️  Time taken: {end - start:.2f} seconds\n")

# Convert to DataFrame
results_df = pd.DataFrame(results).T
print("✅ Final Results:")
print(results_df)

# 🔍 Plotting
metrics = ['accuracy', 'precision', 'recall', 'f1_score']
for metric in metrics:
    results_df[metric].plot(kind='barh', title=f"{metric.capitalize()} per Model", legend=False)
    plt.xlabel(metric.capitalize())
    plt.tight_layout()
    plt.show()

# 🔎 Confusion matrices for top 2 models
top_models = results_df.sort_values("f1_score", ascending=False).head(2)

for model_name in top_models.index:
    print(f"\n🔍 Confusion Matrix for: {model_name}")
    
    model = models[model_name]
    pipeline = make_pipeline(vectorizer, model)
    
    y_pred = cross_val_predict(pipeline, X, y, cv=kf)
    cm = confusion_matrix(y, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title(f'Confusion Matrix: {model_name}')
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()
