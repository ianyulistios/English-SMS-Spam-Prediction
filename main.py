from flask import Flask,jsonify,request
from flasgger import Swagger
from sklearn.externals import joblib
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
Swagger(app)
CORS(app)

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

    clf = joblib.load('GlassknnClassifierbaru.pkl')

    resultPredict = clf[0].classifier(X_New)

    return jsonify({'message': format(clf[1].target_names[resultPredict])})


app.run()