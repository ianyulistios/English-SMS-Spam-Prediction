from flask import Flask,jsonify,request
from flasgger import Swagger
from sklearn.externals import joblib
import numpy as np
from flask_cors import CORS

import pandas as pd

app = Flask(__name__)
Swagger(app)
CORS(app)

@app.route('/input/task', methods=['POST'])
def predict():
    """
    Ini Adalah Endpoint Untuk Mengklasifikasi Jamur
    ---
    tags:
        - Rest Controller
    parameters:
      - name: body
        in: body
        required: true
        schema:
          id: Petal
          required:
            - odor
            - gill-size
            - gill-color
            - spore-print-color

          properties:
            odor:
              type: int
              description: Please input with valid Magnesium.
              default: 0
            gill-size:
              type: int
              description: Please input with valid Refractive Index.
              default: 0
            gill-color:
              type: int
              description: Please input with valid Almunium.
              default: 0
            spore-print-color:
              type: int
              description: Please input with valid Calcium.
              default: 0

    responses:
        200:
            description: Success Input
    """
    new_task = request.get_json()

    odor = new_task['odor']
    gill_s = new_task['gill-size']
    gill_c = new_task['gill-color']
    spore_p_c = new_task['spore-print-color']

    X_New = np.array([[odor,gill_c,gill_s,spore_p_c]])
    x_new = X_New.reshape(1,-1)

    knn = joblib.load('MushroomClassifier.pkl')

    resultPredict = knn[0].predict(x_new)

    return jsonify({'message': format(resultPredict)})

if __name__ == '__main__' :
 app.run(debug=True)