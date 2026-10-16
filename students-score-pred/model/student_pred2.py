# %% [markdown]
# # SECOND NOTEBOOK FOR STUDENT SCORE PREDICTION

# %%
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

# %%
df = pd.read_csv('./student1.csv')

# %%
df.head()

# %%
df = df.drop(['race/ethnicity'], axis=1)
df

# %% [markdown]
# ### PREPARE DATA

# %%
X = df.drop(columns=[
    'math score',
    'reading score',
    'writing score'
])

y_math = df['math score']
y_reading = df['reading score']
y_writing = df['writing score']

# %%
X_train, X_test, y_math_train, y_math_test = train_test_split(
    X, y_math,
    test_size=0.2,
    random_state=42
)

_, _, y_reading_train, y_reading_test = train_test_split(
    X, y_reading,
    test_size=0.2,
    random_state=42
)

_, _, y_writing_train, y_writing_test = train_test_split(
    X, y_writing,
    test_size=0.2,
    random_state=42
)

# %% [markdown]
# ### ENCODE CATEGORICAL COLUMNS

# %%
categorical_columns = X.columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        (
            'encoder',
            OneHotEncoder(handle_unknown='ignore'),
            categorical_columns
        )
    ]
)

# %%
param_grid = {
    'regressor__n_estimators': [100, 200, 300],
    'regressor__max_depth': [None, 10, 20],
    'regressor__min_samples_split': [2, 5, 10],
    'regressor__min_samples_leaf': [1, 2, 4]
}

# %%
def train_model(X_train, X_test, y_train, y_test, score_name):

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
    predictions = best_model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)

    print(f'\n--- {score_name} Model Results ---')
    print('Best Parameters:', grid_search.best_params_)
    print(f'Best Cross Validation MAE: {-grid_search.best_score_:.2f}')
    print(f'Test MAE: {mae:.2f}')
    print(f'Test RMSE: {mse ** 0.5:.2f}')

    return best_model, grid_search

# %%
math_model, math_search = train_model(X_train, X_test, y_math_train, y_math_test, 'Math')
reading_model, reading_search = train_model(X_train, X_test, y_reading_train, y_reading_test, 'Reading')
writing_model, writing_search = train_model(X_train, X_test, y_writing_train, y_writing_test, 'Writing')

# %%
maths_predictions = math_model.predict(X_test)

results = pd.DataFrame({
    'Actual Math Score': y_math_test.values, 'Predicted Math Score': maths_predictions.round(2)
})

results.head(50)

# %%



