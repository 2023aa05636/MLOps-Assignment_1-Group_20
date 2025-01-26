import pickle
from flask import Flask, request, jsonify

# Load the model
with open("models/iris_model.pkl", "rb") as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    features = data["features"]
    prediction = model.predict([features])
    return jsonify({"prediction": prediction.tolist()})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
