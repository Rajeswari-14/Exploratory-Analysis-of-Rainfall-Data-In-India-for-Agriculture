from flask import Flask, render_template, request
import pickle
import numpy as np

model = pickle.load(open("rainfall_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        MinTemp = float(request.form["MinTemp"])
        MaxTemp = float(request.form["MaxTemp"])
        Rainfall = float(request.form["Rainfall"])

        input_data = np.array([[MinTemp, MaxTemp, Rainfall]])

        input_scaled = scaler.transform(input_data)

        prediction = model.predict(input_scaled)

        if prediction[0] == 1:
            return render_template("chance.html")
        else:
            return render_template("noChance.html")

    except Exception as e:
        return str(e)

if __name__ == "__main__":
    app.run(debug=True)
