from flask import Flask, jsonify, request
import model  # This will be your machine learning script.
import logging

app = Flask(__name__)

# Set up logging configuration
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Set logging level after the app instance is created.
app.logger.setLevel(logging.DEBUG)

# Add stream handler to send logs to STDOUT.
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.DEBUG)
app.logger.addHandler(stream_handler)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Here you will handle the prediction.
        # This requires refactoring your model.py script.
        data = request.get_json(force=True)
        prediction = model.predict(data)
        app.logger.info('Prediction successfully made.')
        return jsonify({"prediction": prediction})
    except Exception as e:
        app.logger.error(f"Error during prediction: {e}")
        return jsonify({"error": "Error during prediction", "details": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
