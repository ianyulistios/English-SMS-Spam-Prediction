from flask import Flask,jsonify,request
from flasgger import Swagger
from sklearn.externals import joblib
from flask_cors import CORS
import numpy as np
import pandas as pd

app = Flask(__name__)
Swagger(app)

@app.route('/input/task', methods=['POST'])
def classifier():
    """
    Ini Adalah Endpoint Untuk Mengklasifikasi KACA
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
            - Mg
            - RI
            - Al
            - Ca
            - Na
          properties:
            Mg:
              type: float
              description: Please input with valid Magnesium.
              default: 0
            RI:
              type: float
              description: Please input with valid Refractive Index.
              default: 0
            Al:
              type: float
              description: Please input with valid Almunium.
              default: 0
            Ca:
              type: float
              description: Please input with valid Calcium.
              default: 0
            Na:
              type: float
              description: Please input with valid Sodium.
              default: 0
    responses:
        200:
            description: Success Input
    """
    new_task = request.get_json()

    Mg = new_task['Mg']
    RI = new_task['RI']
    Al = new_task['Al']
    Ca = new_task['Ca']
    Na = new_task['Na']

    X_New = np.array([[Mg,RI,Al,Ca,Na]])
    x_new = X_New.reshape(1,-1)

    clf = joblib.load('GlassKnnClassifier.pkl')

    resultPredict = clf[0].predict(x_new)

    return jsonify({'message': format(resultPredict)})

app.run(debug=True)