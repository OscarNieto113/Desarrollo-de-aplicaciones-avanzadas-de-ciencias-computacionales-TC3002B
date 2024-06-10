import os
import pickle
import features
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'queries'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Cargar el modelo
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Puerto de Flask
PORT = 5000

def convert_to_serializable(obj):
    if isinstance(obj, np.int32):
        return int(obj)
    elif isinstance(obj, np.float32):
        return float(obj)
    return obj

@app.route('/api/upload', methods=['POST'])
def upload_files():
    if 'file1' not in request.files or 'file2' not in request.files:
        return jsonify({'error': 'Both files are required'}), 400

    file1 = request.files['file1']
    file2 = request.files['file2']

    file1.save(os.path.join(app.config['UPLOAD_FOLDER'], file1.filename))
    file2.save(os.path.join(app.config['UPLOAD_FOLDER'], file2.filename))

    data = features.table_generator()
    print(data)

    # Hacer una predicción
    prediction = model.predict([list(data[0].values())])[0]
    response = {
        'data': data,
        'prediction': convert_to_serializable(prediction)
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, port=PORT)
