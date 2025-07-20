# Early Mental Health Detection Based on Reddit Posts

This project uses Natural Language Processing (NLP) and Machine Learning techniques to classify Reddit posts according to potential mental health concerns. The goal is to develop an AI-powered system that can help identify early signs of mental health conditions based on social media text data.

## Project Overview

-  **Problem**: Mental health conditions are often underdiagnosed or diagnosed late. Social media platforms like Reddit provide an opportunity to analyze early indicators through user-generated content.
-  **Goal**: To classify Reddit posts into categories such as depression, anxiety, PTSD, and control using ML models trained on labeled data.
-  **Dataset**: Reddit Mental Health Dataset (from r/depression, r/Anxiety, r/PTSD, r/SuicideWatch, r/mentalhealth, and control posts)

##  Technologies Used

-  Python
-  Pandas, NumPy
-  Scikit-learn, XGBoost, LightGBM
-  NLTK (for text preprocessing)
-  Matplotlib, Seaborn (for visualization)
-  SHAP &  LIME (for model interpretability)

##  Features

- Cleaned and preprocessed Reddit posts (lowercasing, stopword removal, lemmatization)
- TF-IDF and word embedding features (with optional BERT integration)
- Trained multiple classifiers (Logistic Regression, Random Forest, XGBoost, LightGBM)
- Evaluated models using accuracy, F1-score, confusion matrix
- Visualized model interpretability with:
  - SHAP (SHapley Additive exPlanations)
  - Partial Dependence Plots (PDPs)

##  Disclaimer

This project is for academic/research purposes only and not intended for real-world mental health diagnosis.


