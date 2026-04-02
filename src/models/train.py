from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier
import joblib
import pandas as pd

def train_model(df, target_col):

    X = df.drop(columns=[target_col])
    y = df[target_col]
    y = y.map({"No": 0, "Yes": 1})

    categorical_cols = X.select_dtypes(include=['object']).columns
    numeric_cols = X.select_dtypes(exclude=['object']).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", "passthrough", numeric_cols)
        ]
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", XGBClassifier(
            n_estimators=300,
            learning_rate=0.1,
            max_depth=6,
            random_state=42,
            n_jobs=-1,
            eval_metric="logloss"
        ))
    ])

    pipeline.fit(X, y)

    # ✅ Save full pipeline
    joblib.dump(pipeline, "pipeline.pkl")

    print("✅ Pipeline saved as pipeline.pkl")



if __name__ == "__main__":

    # Load dataset
    df = pd.read_csv("/Users/bitla/OneDrive/Desktop/Telco-Customer-Churn-ML/data/raw/Telco-Customer-Churn.csv")

    # Call training
    train_model(df, target_col="Churn")