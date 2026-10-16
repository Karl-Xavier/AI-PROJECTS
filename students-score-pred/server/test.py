
import pandas as pd
import json
import joblib

model = joblib.load("artifacts/student_performance_model.pkl")

with open("artifacts/columns.json") as f:
    columns = json.load(f)

with open("artifacts/template.json") as f:
    template = json.load(f)

tests = [
    {"paid": "yes", "romantic": "yes"},
    {"paid": "no", "romantic": "yes"},
    {"paid": "yes", "romantic": "no"},
    {"paid": "no", "romantic": "no"},
]

for changes in tests:
    row = template.copy()
    row.update(changes)
    row["G1"] = 15
    row["G2"] = 15

    X = pd.DataFrame(
        [{col: row[col] for col in columns}],
        columns=columns
    )

    print(changes, "->", model.predict(X)[0])



for g1, g2 in [(0, 0), (5, 5), (10, 10), (15, 15), (20, 20)]:
    row = template.copy()

    row["G1"] = g1
    row["G2"] = g2
    row["paid"] = "yes"
    row["romantic"] = "yes"

    X = pd.DataFrame(
        [{col: row[col] for col in columns}],
        columns=columns
    )

    print(
        f"G1={g1}, G2={g2} -> "
        f"{model.predict(X)[0]:.4f}"
    )



import pandas as pd

preprocessor = model.named_steps["preprocessor"]
regressor = model.named_steps["regressor"]

feature_names = preprocessor.get_feature_names_out()

importance = pd.Series(
    regressor.feature_importances_,
    index=feature_names
)

print("\nG1 and G2 feature importances:")

for name, value in importance.items():
    if "G1" in name or "G2" in name:
        print(f"{name}: {value:.6f}")

print("\nTotal trees:", len(regressor.estimators_))
print("Training features:", model.n_features_in_)



preprocessor = model.named_steps["preprocessor"]
regressor = model.named_steps["regressor"]

feature_names = preprocessor.get_feature_names_out()

print("\nFeatures containing G1 or G2:")

for name in feature_names:
    if "G1" in name or "G2" in name:
        print(name)

importance = pd.Series(
    regressor.feature_importances_,
    index=feature_names
).sort_values(ascending=False)

print("\nG1/G2 importances, if present:")

for name, value in importance.items():
    if "G1" in name or "G2" in name:
        print(f"{name}: {value:.8f}")

print("\nLowest 10 feature importances:")
print(importance.tail(10).to_string())



preprocessor = model.named_steps["preprocessor"]

print("\nColumnTransformer configuration:")

for name, transformer, cols in preprocessor.transformers_:
    print(f"\nTransformer: {name}")
    print(f"Type: {type(transformer).__name__}")
    print(f"Columns: {cols}")


import pandas as pd
import numpy as np

# Load the saved template and feature order
df = pd.read_json("artifacts/template.json", typ="series")
columns = pd.read_json("artifacts/columns.json", typ="series").tolist()

# Create two otherwise identical students
student_1 = df.copy()
student_2 = df.copy()

student_1["G1"] = 0
student_1["G2"] = 0

student_2["G1"] = 20
student_2["G2"] = 20

X_test = pd.DataFrame([
    student_1[columns],
    student_2[columns]
])

# Transform the inputs using the saved preprocessing step
preprocessor = model.named_steps["preprocessor"]
X_transformed = preprocessor.transform(X_test)

feature_names = preprocessor.get_feature_names_out()

# Convert transformed data into a DataFrame for inspection
X_transformed_df = pd.DataFrame(
    X_transformed.toarray()
    if hasattr(X_transformed, "toarray")
    else X_transformed,
    columns=feature_names
)


# Print the actual feature names produced by preprocessing
print("\nAll transformed feature names:")
print(preprocessor.get_feature_names_out().tolist())

# Find the actual names containing G1 or G2
grade_features = [
    name for name in feature_names
    if "G1" in name or "G2" in name
]

print("\nGrade features found:", grade_features)

# Display the values using their actual column names
if grade_features:
    print("\nTransformed G1/G2 values:")
    print(X_transformed_df[grade_features])

    print("\nAre the transformed rows identical?")
    print(np.allclose(
        X_transformed_df.iloc[0].to_numpy(),
        X_transformed_df.iloc[1].to_numpy()
    ))
else:
    print("\nG1 and G2 were not found in the transformed feature names.")
