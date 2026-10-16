# %% [markdown]
# ## THE DATASET USED HERE COULD NOT PRODUCE A GOOD MODEL CHECK OUT THE OTHER ONE

# %% [markdown]
# # MY FIRST PROJECT (WITHOUT TUTORIAL)

# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

# %%
df = pd.read_csv('./student.csv')
df.head()

# %%
df.shape

# %%
df = df.drop(['Name', 'Gender', 'ParentalSupport', 'Online Classes Taken'], axis=1)

# %%
df

# %%
print(df.isnull().sum())

# %%
print('OG Shape: ', df.shape)
print('Dropped Shape: ', df.dropna().shape)

# %% [markdown]
# ### REMOVE ROWS WITH 2 OR MORE NAN VALUES

# %%
df = df[df.isnull().sum(axis = 1) < 2].copy()

# %%
df.shape

# %% [markdown]
# ### FILL MISSING VALUES WITH THEIR MEDIANS

# %%
input_cols = [
    'StudyHoursPerWeek',
    'PreviousGrade',
    'StudentID', 'ExtracurricularActivities', 'Attendance (%)', 'AttendanceRate', 'Study Hours'
]

for col in input_cols:
    if df[col].dtype in ['int64', 'float64']:
        df[col] = df[col].fillna(df[col].median())

# %%
df

# %% [markdown]
# ### DROP NAN VALUES IN FINAL GRADE COLUMN

# %%
df = df.dropna(subset=['FinalGrade'])

# %%
df

# %% [markdown]
# ### SPLIT THE DATASET INTO X AND Y

# %%
X = df.drop(columns=['FinalGrade'])
y = df['FinalGrade']

# %%
print('X shape: ', X.shape)
print('y shape: ', y.shape)

print(X.head())
print(y.head())

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# %%
model = LinearRegression()

model.fit(X_train, y_train)

# %%
y_pred = model.predict(X_test)

print(y_pred[:10])

# %%
comparison = pd.DataFrame({
    'Actual Grade': y_test,
    'Predicted Grade': y_pred
})

comparison.head(10)

# %%
plt.figure(figsize=(8, 5))

plt.scatter(y_test, y_pred)

plt.xlabel('Actual Grade')
plt.ylabel('Predicted Grade')
plt.title('Actual vs Predicted Grade')

plt.show()

# %%
print(mean_absolute_error(y_test, y_pred),
mean_squared_error(y_test, y_pred),
r2_score(y_test, y_pred))

# %%
df.corr(numeric_only=True)['FinalGrade'].sort_values(ascending=False)

# %%
df['FinalGrade'].describe()

# %%
df[['StudyHoursPerWeek', 'Study Hours']].corr()

# %% [markdown]
# ## DEFINE THE HYPER PARAMETERS TO TEST

# %%
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# %%
rf = RandomForestRegressor(random_state=42)

# %%
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    scoring='neg_mean_absolute_error',
    n_jobs=-1,
    verbose=1
)

# %%
grid_search.fit(X_train, y_train)

# %%
-grid_search.best_score_

# %%
best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test)

test_mae = mean_absolute_error(y_test, y_pred)
test_r = r2_score(y_test, y_pred)

print(
    'Test MAE: ', test_mae,
    'Test R2: ', test_r
)

# %%
comparison = pd.DataFrame({
    'Actual Grade': y_test,
    'Predicted Grade': y_pred
})

comparison.head(10)

# %%
plt.figure(figsize=(8, 5))

plt.scatter(y_test, y_pred)

plt.xlabel('Actual Grade')
plt.ylabel('Predicted Grade')
plt.title('Actual vs Predicted Grade')
