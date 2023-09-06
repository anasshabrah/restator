from flask import Flask, jsonify
import model  # This will be your machine learning script.

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    # Here you will handle the prediction.
    # This requires refactoring your model.py script.
    data = request.get_json(force=True)
    prediction = model.predict(data)
    return jsonify({"prediction": prediction})

if __name__ == "__main__":
    app.run(debug=True)
