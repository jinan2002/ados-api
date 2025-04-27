from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
import numpy as np
import os  # ← أضفناها هنا عشان نقرأ متغيرات البيئة

app = Flask(__name__)
CORS(app)  # ← عشان نحل مشكلة CORS

# Load the trained model WITHOUT compiling (to avoid errors with custom losses)
model = load_model('final_model.h5', compile=False)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    print("Received data:", data)
    sequence = data.get('sequence')

    if sequence is None or not isinstance(sequence, list):
        return jsonify({'error': 'Invalid input. Please provide a "sequence" of features.'}), 400

    try:
        sequence_array = np.array(sequence).reshape(1, len(sequence), len(sequence[0]))
        prediction = model.predict(sequence_array)
        predicted_class = int(np.argmax(prediction, axis=1)[0])
        confidence = float(np.max(prediction))

        return jsonify({
            'predicted_class': predicted_class,
            'confidence': confidence
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # ← نقرأ البورت من البيئة
    app.run(host='0.0.0.0', port=port)  # ← نفتح السيرفر للعالم كله



