import numpy as np
from flask import Flask, request, render_template, jsonify
import pickle
import gdown
import os

app = Flask(__name__)

# -----------------------------
# DOWNLOAD MODEL FROM GOOGLE DRIVE
# -----------------------------
model_url = "https://drive.google.com/uc?export=download&id=1wuu4pZYFi7zOJ1jQ1TL28L4QUzfqkuZb"
model_path = "rfmodel.pkl"

if not os.path.exists(model_path):
    print("Downloading model...")
    gdown.download(model_url, model_path, quiet=False)

# -----------------------------
# LOAD MODEL
# -----------------------------
print("Loading model...")
rfmodel = pickle.load(open(model_path, "rb"))
print("Model loaded successfully!")


# -----------------------------
# HOME PAGE
# -----------------------------
@app.route("/")
def home():
    return render_template("home.html")


# -----------------------------
# PREDICT FROM HTML FORM
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    data = [float(x) for x in request.form.values()]
    final_input = np.array(data).reshape(1, -1)

    output = rfmodel.predict(final_input)[0]
    return render_template("home.html",
                           prediction_text=f"Predicted California House Price: {output}")


# -----------------------------
# PREDICT FROM JSON
# -----------------------------
@app.route("/predict_api", methods=["POST"])
def predict_api():
    data = request.json["data"]
    final_input = np.array(list(data.values())).reshape(1, -1)
    output = rfmodel.predict(final_input)[0]
    return jsonify({"prediction": str(output)})


# -----------------------------
# LOCAL RUN
# -----------------------------
if __name__ == "__main__":
    print("FILE EXECUTED")
    app.run(debug=True)
