# %% [markdown]
# ## WORKING WITH A NEW DATASET LAST NOTEBOOK COULDN'T DELIVER.

# %%
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt

# %%
df = pd.read_csv('./student2.csv', sep=';')
df

# %%
df['studytime']

# %%
X = df.drop(columns=['G3'])
y = df['G3']

category_columns = X.select_dtypes(include=['object', 'str', 'category']).columns.tolist()

category_columns

# %%

category_columns = X.select_dtypes(
    include=["object", "category", "str"]
).columns.tolist()

numerical_columns = X.select_dtypes(
    include=["number"]
).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore"),
            category_columns
        ),
        (
            "numeric",
            "passthrough",
            numerical_columns
        )
    ],
    remainder="drop"
)


# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %%
param_grid = {
    'regressor__n_estimators': [100, 200, 300],
    'regressor__max_depth': [None, 5, 10, 20],
    'regressor__min_samples_split': [2, 5, 10],
    'regressor__min_samples_leaf': [6, 15, 30],
    # 'regressor__max_features': [1.0, 'sqrt', 0.5]
}

# %%
def train_model(X_train, X_test, y_train, y_test):

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor())
    ])

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    prediction = grid_search.predict(X_test)

    mae = mean_absolute_error(y_test, prediction)
    mse = mean_squared_error(y_test, prediction)

    print('Best Parameters:', grid_search.best_params_)
    print(f'Best Cross Validation MAE: {-grid_search.best_score_:.2f}')
    print(f'Test MAE: {mae:.2f}')
    print(f'Test RMSE: {mse ** 0.5:.2f}')

    return best_model, grid_search

# %%
model, search = train_model(X_train, X_test, y_train, y_test)

# %%
model_pred = model.predict(X_test)

results = pd.DataFrame({
    'G3': y_test, 'Predicted G3': model_pred.round(2)
})

results.head(15)

# %%
plt.figure(figsize=(8, 6))

plt.scatter(y_test, model_pred, alpha=0.7)

plt.plot([0, 20], [0, 20], 'r--', label='PREDICTION')

plt.xlabel('Actual G3')
plt.ylabel('Predicted G3')
plt.title('Actual vs Predicted G3')
plt.legend()
plt.grid(True)

plt.xlim(0, 20)
plt.ylim(0, 20)

plt.show()

# %% [markdown]
# ### SAVE THE MODEL

# %%
import joblib

joblib.dump(model, 'student_performance_model.pkl')

# %%
import json

columns = list(X_train.columns)

with open('columns.json', 'w') as f:
    json.dump(columns, f, indent=4)

print(columns)

# %%
template = df[
    (df['Mjob'] == 'teacher') & (df['Fjob'] == 'teacher')
].iloc[0]

template

# %%
template_data = template.to_dict()

with open('template.json', 'w') as f:
    json.dump(template_data, f, indent=4)

print(template_data)


