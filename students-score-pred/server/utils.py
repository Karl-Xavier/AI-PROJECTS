import json
import joblib
import pandas as pd


__data_columns = None
__templates = None
__model = None


def load_artifacts():
    print('Loading Artifacts....')

    global __data_columns, __templates, __model

    # BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # ARTIFACTS_DIR = os.path.join(BASE_DIR, 'artifacts')

    with open('./artifacts/columns.json', 'r') as f:
        __data_columns = json.load(f)

    with open('./artifacts/student_performance_model.pkl', 'rb') as f:
        __model = joblib.load(f)

    with open('./artifacts/template.json', 'r') as f:
            __templates = json.load(f)

    print('Artifacts Sucessfully loaded')

# def getdata_column():
#     return __data_columns

# def get_model():
#     return __model

def predicted_score(data):
    if __model is None:
        raise RuntimeError('model has not been loaded')

    input_data = __templates.copy()

    input_data.update(data)

    missing = [
        col for col in __data_columns
        if col not in input_data
    ]

    if missing:
        raise ValueError(f"Missing features: {missing}")

    input_data = {
        col: input_data[col]
        for col in __data_columns
    }

    df = pd.DataFrame([input_data])

    print("Received data:", data)
    print("G1:", df["G1"].iloc[0])
    print("G2:", df["G2"].iloc[0])
    print("Studytime:", df["studytime"].iloc[0])
    print("All model inputs:")
    print(df.to_string(index=False))

    prediction = __model.predict(df)

    print(prediction[0])

    return round(float(prediction[0]))