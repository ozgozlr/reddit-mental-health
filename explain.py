import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib
from lime.lime_text import LimeTextExplainer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# --- Configurations ---
MODEL_PATH = 'logistic_model_tuned2.pkl'
DATA_PATH = "OriginalRedditData/raw_data/2022/May22"
SAMPLE_PER_LABEL = 100

label_map = {
    'anx': 'Anxiety',
    'dep': 'Depression',
    'lone': 'Loneliness',
    'mh': 'MentalHealth',
    'Sw': 'SuicideWatch'
}

# --- Step 1: Load and prepare test data ---
print("[INFO] Loading May 2022 data for testing...")

all_dfs = []
for filename in os.listdir(DATA_PATH):
    for prefix, label in label_map.items():
        if filename.startswith(prefix):
            file_path = os.path.join(DATA_PATH, filename)
            df = pd.read_csv(file_path)
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

# --- Step 2: Load pipeline ---
print("[INFO] Loading trained pipeline from disk...")
pipeline = joblib.load(MODEL_PATH)
print("[INFO] Pipeline loaded.")

# --- Step 3: SHAP summary plots ---
print("[INFO] Transforming test set for SHAP...")
X_tfidf_test = pipeline.named_steps['tfidf'].transform(X_test)

sample_size = min(100, X_tfidf_test.shape[0])
print(f"[INFO] Sampling {sample_size} instances for SHAP...")
sample_indices = np.random.choice(range(X_tfidf_test.shape[0]), sample_size, replace=False)
X_tfidf_sample = X_tfidf_test[sample_indices]

print("[INFO] Creating SHAP LinearExplainer...")
explainer = shap.LinearExplainer(pipeline.named_steps['clf'], X_tfidf_sample, feature_perturbation="interventional")

print("[INFO] Calculating SHAP values...")
shap_values = explainer.shap_values(X_tfidf_sample)

feature_names = pipeline.named_steps['tfidf'].get_feature_names_out()

print("[INFO] Plotting SHAP summary plots...")
shap.summary_plot(shap_values, features=X_tfidf_sample.toarray(), feature_names=feature_names, show=False)
plt.title("SHAP Summary (Dot) Plot")
plt.tight_layout()
plt.savefig("shap_summary_dot.png")
plt.clf()

shap.summary_plot(shap_values, features=X_tfidf_sample.toarray(), feature_names=feature_names, plot_type="bar", show=False)
plt.title("SHAP Summary (Bar) Plot")
plt.tight_layout()
plt.savefig("shap_summary_bar.png")
plt.clf()

print("[INFO] Script completed.")
