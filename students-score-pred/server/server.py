from flask import Flask, request, jsonify
from flask_cors import CORS
app = Flask( __name__ )

import utils

CORS(app)

utils.load_artifacts()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(silent=True)

        if not isinstance(data, dict):
            return jsonify({
                'error': 'No JSON Data recieved'
            }), 400

        prediction = utils.predicted_score(data)

        return jsonify({
            'prediction': prediction
        }),200
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500

if __name__ == "__main__":
    print("Starting python Flask Server")
    app.run(debug=True)