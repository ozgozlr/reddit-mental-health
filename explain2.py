import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import pickle
import joblib
import os
from glob import glob
import nltk
from nltk.corpus import stopwords

# Download stopwords if not already downloaded
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Path to your model and data
MODEL_PATH = 'logistic_model_tuned.pkl'  # Update this with your model path
DATA_DIR = 'OriginalRedditData/raw_data/2022/May22'  # Update if needed

# Load your pre-trained model
print("Loading model...")
try:
    # Method 1: Using pickle
    model = pickle.load(open(MODEL_PATH, 'rb'))
except:
    try:
        # Method 2: Using joblib (which handles large objects better)
        model = joblib.load(MODEL_PATH)
    except Exception as e:
        print(f"Error loading model: {e}")
        exit(1)

# Load and combine all CSV files from May22 directory
print("Loading data...")
all_csv_files = glob(os.path.join(DATA_DIR, '*.csv'))
print(f"Found {len(all_csv_files)} CSV files: {[os.path.basename(f) for f in all_csv_files]}")

# Load and combine all CSVs
dataframes = []
for csv_file in all_csv_files:
    try:
        df = pd.read_csv(csv_file)
        # Add a source column to track which file each row came from
        df['source_file'] = os.path.basename(csv_file)

        # Optional: remove stopwords from text column if it exists
        if 'selftext' in df.columns:
            df['selftext'] = df['selftext'].astype(str).apply(
        lambda x: ' '.join([word for word in x.split() if word.lower() not in stop_words])
    )


        dataframes.append(df)
        print(f"Loaded {os.path.basename(csv_file)} with {df.shape[0]} rows and {df.shape[1]} columns")
    except Exception as e:
        print(f"Error loading {csv_file}: {e}")

if not dataframes:
    print("No data could be loaded. Exiting.")
    exit(1)

# Combine all dataframes
combined_df = pd.concat(dataframes, ignore_index=True)
print(f"Combined data shape: {combined_df.shape}")

# Display the columns to understand the data structure
print("Columns in the dataset:")
print(combined_df.columns.tolist())

# Ask for input to select which columns to use as features
print("\nYou need to specify which columns to use as features for SHAP analysis.")
print("Options:")
print("1. Use all columns except specific ones (you'll be asked which to exclude)")
print("2. Specify columns to include manually")

# For demonstration purposes, let's assume option 1 and exclude common non-feature columns
possible_non_features = ['id', 'title', 'text', 'author', 'created_utc', 'score',
                         'subreddit', 'url', 'source_file', 'label', 'target', 'timestamp']
existing_non_features = [col for col in possible_non_features if col in combined_df.columns]
X = combined_df.drop(columns=existing_non_features, errors='ignore')

print(f"\nUsing {X.shape[1]} columns as features: {X.columns.tolist()}")

# Handle missing values if any
if X.isna().any().any():
    print("Data contains missing values. Filling with appropriate values...")
    numeric_cols = X.select_dtypes(include=['number']).columns
    X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].mean())
    non_numeric_cols = X.select_dtypes(exclude=['number']).columns
    for col in non_numeric_cols:
        X[col] = X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else "MISSING")

# Convert any categorical features to numeric
print("Processing categorical features...")
categorical_columns = X.select_dtypes(include=['object', 'category']).columns
for col in categorical_columns:
    print(f"Converting categorical column: {col}")
    try:
        X[col] = pd.to_numeric(X[col])
    except:
        try:
            one_hot = pd.get_dummies(X[col], prefix=col, drop_first=False)
            X = pd.concat([X.drop(columns=[col]), one_hot], axis=1)
            print(f"  One-hot encoded {col} into {one_hot.shape[1]} columns")
        except Exception as e:
            print(f"  Error processing column {col}: {e}")
            print(f"  Dropping column {col}")
            X = X.drop(columns=[col])

print(f"Final feature set shape: {X.shape}")

# Create a SHAP explainer object based on the model type
print("Creating SHAP explainer...")
try:
    explainer = shap.TreeExplainer(model)
    print("Using TreeExplainer")
except:
    try:
        explainer = shap.Explainer(model, X)
        print("Using standard Explainer")
    except Exception as e:
        print(f"Error creating explainer: {e}")
        print("Falling back to Kernel explainer (slower but works with any model)")
        background_data = shap.sample(X, min(100, X.shape[0]))

        def model_predict(x):
            try:
                return model.predict(x)
            except:
                try:
                    return model.predict_proba(x)[:, 1]
                except:
                    return np.array([0] * len(x))
        explainer = shap.KernelExplainer(model_predict, background_data)

# Calculate SHAP values
print("Computing SHAP values (this may take a while for large datasets)...")
sample_size = min(100, X.shape[0])
sample_indices = np.random.choice(X.shape[0], sample_size, replace=False)
X_sample = X.iloc[sample_indices]

try:
    shap_values = explainer.shap_values(X_sample)
    print("SHAP values successfully calculated")
    if isinstance(shap_values, list) and len(shap_values) > 1:
        print(f"Multi-class model detected with {len(shap_values)} classes")
        print(f"Using SHAP values for class 1 (positive class) for the plot")
        shap_values_plot = shap_values[1]
    else:
        shap_values_plot = shap_values
except Exception as e:
    print(f"Error computing SHAP values: {e}")
    print("Trying with a smaller sample...")
    sample_size = min(100, X.shape[0])
    sample_indices = np.random.choice(X.shape[0], sample_size, replace=False)
    X_sample = X.iloc[sample_indices]
    try:
        shap_values = explainer.shap_values(X_sample)
        print("SHAP values successfully calculated with smaller sample")
        if isinstance(shap_values, list) and len(shap_values) > 1:
            shap_values_plot = shap_values[1]
        else:
            shap_values_plot = shap_values
    except Exception as e:
        print(f"Failed to compute SHAP values: {e}")
        exit(1)

# Create SHAP plots
print("Creating SHAP plots...")
output_dir = "shap_plots"
os.makedirs(output_dir, exist_ok=True)

# 1. Summary dot plot
plt.figure(figsize=(14, 10))
shap.summary_plot(
    shap_values_plot,
    X_sample,
    plot_type="dot",
    show=True,
    max_display=20
)
plt.title("SHAP Summary (Dot) Plot - Reddit Data May 2022", fontsize=16)
plt.tight_layout()
dot_plot_path = os.path.join(output_dir, "shap_summary_dot_plot.png")
plt.savefig(dot_plot_path, dpi=300, bbox_inches='tight')
print(f"Dot plot saved to {dot_plot_path}")
plt.close()

# 2. Bar plot
plt.figure(figsize=(14, 10))
shap.summary_plot(
    shap_values_plot,
    X_sample,
    plot_type="bar",
    show=False,
    max_display=20
)
plt.title("SHAP Feature Importance (Bar) Plot - Reddit Data May 2022", fontsize=16)
plt.tight_layout()
bar_plot_path = os.path.join(output_dir, "shap_feature_importance_bar.png")
plt.savefig(bar_plot_path, dpi=300, bbox_inches='tight')
print(f"Bar plot saved to {bar_plot_path}")
plt.close()

# 3. Individual feature dependence plots
print("Creating individual feature dependence plots for top features...")
if isinstance(shap_values_plot, np.ndarray):
    mean_abs_shap = np.abs(shap_values_plot).mean(axis=0)
    feature_importance = pd.DataFrame({
        'Feature': X_sample.columns,
        'Importance': mean_abs_shap
    }).sort_values('Importance', ascending=False)

    for i, (feature, _) in enumerate(feature_importance.iloc[:5].itertuples(index=False)):
        plt.figure(figsize=(10, 7))
        shap.dependence_plot(
            feature,
            shap_values_plot,
            X_sample,
            show=False
        )
        plt.title(f"SHAP Dependence Plot - {feature}", fontsize=14)
        plt.tight_layout()
        dep_plot_path = os.path.join(output_dir, f"shap_dependence_{i+1}_{feature.replace('/', '_')}.png")
        plt.savefig(dep_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Dependence plot for {feature} saved to {dep_plot_path}")

    importance_path = os.path.join(output_dir, "feature_importance.csv")
    feature_importance.to_csv(importance_path, index=False)
    print(f"Feature importance saved to {importance_path}")

    print("\nTop 20 Features by Importance:")
    print(feature_importance.head(20).to_string())
else:
    print("Could not create feature dependence plots - unexpected SHAP values format")

print(f"\nAll SHAP plots have been saved to the '{output_dir}' directory")
print("\nSummary of files created:")
for file in os.listdir(output_dir):
    print(f"- {file}")
