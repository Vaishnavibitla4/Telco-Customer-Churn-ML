import mlflow.pyfunc
import joblib

# 👉 Replace this path with your actual model path
MODEL_PATH = "./mlruns/236788551061488637/c4a74d68502e49dfaefb1712c5b64ed9/artifacts/model"

# Load MLflow model
model = mlflow.pyfunc.load_model(MODEL_PATH)

# Save as .pkl
joblib.dump(model, "model.pkl")

print("✅ Model saved as model.pkl")