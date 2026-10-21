# %%
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, make_scorer, f1_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import numpy as np

# %%
df = pd.read_csv('./churd.csv')
df.head()

# %%
df = df.drop(['customerID'], axis=1)

# %%
df

# %%
df['TotalCharges'] = pd.to_numeric(
    df['TotalCharges'], errors='coerce'
).astype(np.float64)

# %%
df.isna().sum()

# %%
df = df.dropna()

# %%
print(df.isnull().sum())

# %%
X = df.drop(columns=['Churn'])
y = df['Churn']

# %%
categorical_columns = X.select_dtypes(include=['object', 'str', 'category']).columns.tolist()

numerical_columns = X.select_dtypes(include=['number']).columns.tolist()

# %%
preprocessor = ColumnTransformer(
    transformers=[
        (
            'encoder',
            OneHotEncoder(handle_unknown='ignore'),
            categorical_columns
        ),
        (
            'numeric',
            StandardScaler(),#'passthrough',
            numerical_columns
        )
    ],
    remainder='drop'
)

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %%
param_grid = {
    'svm': {
        # 'model': svm.SVC(gamma='auto', probability=True),
        'params': {
            'svc__C': [0.1, 1, 10, 100],
            'svc__kernel': ['linear', 'rbf'],
            'svc__class_weight': [None, 'balanced']
        }
    },
    'random_forest': {
        'model': RandomForestClassifier(),
        'params': {
            'randomforestclassifier__n_estimators': [100, 200, 300],
            'randomforestclassifier__max_depth': [None, 5, 10, 20],
            'randomforestclassifier__min_samples_split': [2, 5, 10],
            'randomforestclassifier__min_samples_leaf': [6, 15, 30],
        },
    },
    'logistic_regression': {
        'model': LogisticRegression(solver='lbfgs'),
        'params': {
            'logisticregression__C': [0.01, 0.1, 1, 10],
        }
    }
}

params_grid = {
    'logisticregression__C': [0.01, 0.1, 1, 10]
}

# %%
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


def train_model(Xt, yt, Xte, yte):

    # scores = []
    # best_estimator = {}

    # # for loop

    # for algorithm, mp in param_grid.items():
        #  pipeline

    pipeline = make_pipeline(
        preprocessor,
        LogisticRegression()
    )

    # print(pipeline.named_steps.keys())

    # gridsearchcv but iterate throughh it

    f1_scorer = make_scorer(
        f1_score,
        pos_label='Yes'
    )

    clf = GridSearchCV(
        estimator=pipeline,
        param_grid=params_grid,
        cv=5,
        n_jobs=-1,
        verbose=1,
        scoring='accuracy',
        error_score='raise'
    )
    clf.fit(Xt, yt)
    # scores.append({
    #     'model': algorithm,
    #     'best_score': clf.best_score_,
    #     'best_params': clf.best_params_
    # })

    # best_estimator[algorithm] = clf.best_estimator_

    best_model = clf.best_estimator_

    prediction = clf.predict(Xte)
    

    print('Accuracy:', accuracy_score(yte, prediction))
    print('F1 Score:', f1_score(yte, prediction, pos_label='Yes'))
    print('\nClassification Report:\n',
        classification_report(yte, prediction))

    print('\nConfusion Matrix:\n',
      confusion_matrix(yte, prediction, labels=['No', 'Yes']))
    

    # y_pred = best_model.predict(Xte)

    # print("Selected model:", model_name)
    # print("Accuracy:", accuracy_score(yte, y_pred))
    # print("F1 score:", f1_score(yte, y_pred, pos_label='Yes'))
    # print("\nClassification report:")
    # print(classification_report(yte, y_pred))
    # print("\nConfusion matrix:")
    # print(confusion_matrix(yte, y_pred))

    # results = pd.DataFrame(scores)
    # print("\nGrid search results:")
    # print(results.to_string(index=False))

        # fitting and get mae and mse

    return best_model



# %%
pred = train_model(X_train, y_train, X_test, y_test)

# ('preprocessor', preprocessor),

# %%
import joblib
# from pathlib import Path

# %%
# Path('artifacts').mkdir(exist_ok=True)

joblib.dump(
    pred,
    'artifacts/logisticregression_churn.pkl'
)

print('LOGISTIC REGRESSION Model Saved')

# %%



