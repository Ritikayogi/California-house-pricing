import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd

print("FILE EXECUTED")  # test line

app = Flask(__name__)

# Load the model
rfmodel = pickle.load(open('rfmodel.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']
    print(data)
    final_data = np.array(list(data.values())).reshape(1, -1)
    print(final_data)

    output = rfmodel.predict(final_data)[0]
    return jsonify({"prediction": str(output)})

# ---- THIS WAS THE PROBLEM ----
if __name__ == "__main__":
    print("STARTING SERVER")
    app.run(debug=True)
