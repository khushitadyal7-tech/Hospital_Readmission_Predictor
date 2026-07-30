import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report


# ==============================
# Load Dataset
# ==============================

print("Loading dataset...")

df = pd.read_csv("diabetic_data.csv")

print("Dataset Shape:", df.shape)


# ==============================
# Data Cleaning
# ==============================

# Replace ? with NaN
df = df.replace("?", np.nan)


# Remove columns with too many missing values
drop_columns = [
    "encounter_id",
    "patient_nbr",
    "weight",
    "payer_code",
    "medical_specialty"
]

df = df.drop(columns=drop_columns)


# ==============================
# Target Conversion
# ==============================

# Predict 30-day readmission
# <30 = 1
# Others = 0

df["readmitted"] = df["readmitted"].apply(
    lambda x: 1 if x == "<30" else 0
)


# ==============================
# Split Data
# ==============================

X = df.drop("readmitted", axis=1)
y = df["readmitted"]


# Separate columns

categorical_columns = X.select_dtypes(
    include=["object"]
).columns


numerical_columns = X.select_dtypes(
    exclude=["object"]
).columns


print("Categorical Features:", len(categorical_columns))
print("Numerical Features:", len(numerical_columns))


# ==============================
# Preprocessing
# ==============================

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        )
    ]
)


categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_pipeline,
            numerical_columns
        ),
        (
            "cat",
            categorical_pipeline,
            categorical_columns
        )
    ]
)


# ==============================
# Model Pipeline
# ==============================

model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            DecisionTreeClassifier(
                max_depth=10,
                random_state=42
            )
        )
    ]
)


# ==============================
# Train Test Split
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ==============================
# Training
# ==============================

print("Training model...")

model.fit(
    X_train,
    y_train
)


# ==============================
# Evaluation
# ==============================

y_pred = model.predict(X_test)


accuracy = accuracy_score(
    y_test,
    y_pred
)


print("\nModel Accuracy:", accuracy)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred
    )
)


# ==============================
# Save Model
# ==============================

with open(
    "model.pkl",
    "wb"
) as file:

    pickle.dump(
        model,
        file
    )


print("\n================================")
print("Model saved successfully!")
print("File created: model.pkl")
print("================================")