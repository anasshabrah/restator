from flask import Flask, jsonify, request
import model  # This will be your machine learning script.
import logging

app = Flask(__name__)

# Set logging level after the app instance is created.
app.logger.setLevel(logging.DEBUG)

@app.route("/predict", methods=["POST"])
def predict():
    # Here you will handle the prediction.
    # This requires refactoring your model.py script.
    data = request.get_json(force=True)
    prediction = model.predict(data)
    return jsonify({"prediction": prediction})

if __name__ == "__main__":
    app.run(debug=True)
