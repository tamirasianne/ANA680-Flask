from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load your trained model
model = pickle.load(open("knn_model.pkl", "rb"))

# Features must EXACTLY match the input names in index.html
FEATURES = [
    "ClumpThickness",
    "UniformityCellSize",
    "UniformityCellShape",
    "MarginalAdhesion",
    "SingleEpithelialCellSize",
    "BareNuclei",
    "BlandChromatin",
    "NormalNucleoli",
    "Mitoses"
]

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    values = [float(request.form[f]) for f in FEATURES]
    X = np.array(values).reshape(1, -1)

    pred = model.predict(X)[0]

    # Dataset uses 2 = benign, 4 = malignant
    label = "Benign (2)" if pred == 2 else "Malignant (4)"

    return render_template("index.html", result=label)

if __name__ == "__main__":
    app.run(debug=True)

