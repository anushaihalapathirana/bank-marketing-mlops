from pathlib import Path
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, roc_auc_score, f1_score, average_precision_score)
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.base import clone

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "bank-additional-full.csv"

mlflow.set_tracking_uri(f"sqlite:///{ROOT/'mlflow.db'}")
mlflow.set_experiment("bank_marketing") 

# Load the dataset
df = pd.read_csv(DATA_PATH, sep=';')    

X = df.drop(columns=['y', 'duration'])  # Exclude 'duration' as it is not known before the call
y = df['y'].map({'yes': 1, 'no': 0})

# Split the dataset into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# Define preprocessing for numeric and categorical features
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()     

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=
    [
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# define models to evaluate
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, max_depth=6, n_estimators=100, learning_rate=0.1, eval_metric='logloss', random_state=42)
}   

# train and track models
results = []
trained_models = {}
for name, classifier in models.items():
    with mlflow.start_run(run_name=name):
        pipeline = Pipeline(steps=[('preprocessor', clone(preprocessor)),
                                   ('classifier', classifier)])
        pipeline.fit(X_train, y_train)

        # Predict on validation data
        y_pred = pipeline.predict(X_valid)
        y_prob = pipeline.predict_proba(X_valid)[:, 1]

        # Evaluate
        metrics = {
            "accuracy": accuracy_score(y_valid, y_pred),
            "f1": f1_score(y_valid, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_valid, y_prob),
            "pr_auc": average_precision_score(y_valid, y_prob),
        }

        # Log parameters and metrics to MLflow
        mlflow.log_param("model_name", name)
        mlflow.log_param("split", "chronological_70_15_15")
        mlflow.log_metrics(metrics)
        mlflow.log_params({
            key: str(value)
            for key, value in classifier.get_params().items()
            if key in {
                "max_iter",
                "n_estimators",
                "max_depth",
                "min_samples_leaf",
                "learning_rate",
                "random_state",
            }
        })
        
        # save the pipeline as an artifact
        model_info = mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="model",
            input_example=X_train.head(3),
            serialization_format="cloudpickle",
        )
        trained_models[name] = pipeline
        results.append({"model": name, **metrics})
        
        print(f"\n{name}")
        print(metrics)
        print("Saved model:", model_info.model_uri)

# compare models
results_df = pd.DataFrame(results)
print("\nModel Comparison:")
print(results_df.sort_values(by="roc_auc", ascending=False))