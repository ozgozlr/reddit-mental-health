import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# --- Configurations ---
MODEL_PATH = 'logistic_model_tuned.pkl'
DATA_PATH = "OriginalRedditData/raw_data/2022/May22"
SAMPLE_PER_LABEL = 100

label_map = {
    'anx': 'Anxiety',
    'dep': 'Depression',
    'lone': 'Loneliness',
    'mh': 'MentalHealth',
    'Sw': 'SuicideWatch'
}

# --- Load and prepare data ---
print("[INFO] Loading May 2022 data...")
all_dfs = []
for filename in os.listdir(DATA_PATH):
    for prefix, label in label_map.items():
        if filename.startswith(prefix):
            df = pd.read_csv(os.path.join(DATA_PATH, filename))
            df['label'] = label
            df_limited = df.head(SAMPLE_PER_LABEL)
            all_dfs.append(df_limited)

data = pd.concat(all_dfs, ignore_index=True)
data['text'] = data['title'].fillna('') + ' ' + data['selftext'].fillna('')
X = data['text']
y = data['label']

le = LabelEncoder()
y_encoded = le.fit_transform(y)

_, X_test, _, y_test = train_test_split(
    X, y_encoded, test_size=0.1, stratify=y_encoded, random_state=42
)
print(f"[INFO] Test samples: {len(X_test)}")

# --- Load model ---
print("[INFO] Loading pipeline...")
pipeline = joblib.load(MODEL_PATH)
print("[INFO] Pipeline loaded.")

# --- Transform test data ---
X_tfidf_test = pipeline.named_steps['tfidf'].transform(X_test)
sample_size = min(100, X_tfidf_test.shape[0])
sample_indices = np.random.choice(range(X_tfidf_test.shape[0]), sample_size, replace=False)
X_tfidf_sample = X_tfidf_test[sample_indices]
y_sample = y_test[sample_indices]

feature_names = pipeline.named_steps['tfidf'].get_feature_names_out()
clf = pipeline.named_steps['clf']

# --- SHAP explanation with modern API ---
print("[INFO] Creating SHAP global summary plots (per class)...")
masker = shap.maskers.Independent(X_tfidf_sample)
explainer = shap.Explainer(clf, masker)
shap_values = explainer(X_tfidf_sample)

# shap_values.values shape = (n_samples, n_features, n_classes)
for class_idx, class_label in enumerate(le.classes_):
    print(f"[INFO] Generating SHAP plots for class: {class_label}")
    class_indices = np.where(y_sample == class_idx)[0]
    if len(class_indices) == 0:
        continue

    shap.summary_plot(
        shap_values[..., class_idx][class_indices],
        features=X_tfidf_sample[class_indices].toarray(),
        feature_names=feature_names,
        show=False
    )
    plt.title(f"SHAP Dot Plot - {class_label}")
    plt.tight_layout()
    plt.savefig(f"shap_dot_{class_label}.png")
    plt.clf()

print("[INFO] SHAP class-wise plots saved.")
