import joblib

# Load the trained model
model = joblib.load("logistic_model_tuned.pkl")

# Predict on new sample
text = ["I'm feeling overwhelmed with work and life."]
prediction = model.predict(text)

# Print result
print("🔍 Predicted subreddit label:", prediction[0])
