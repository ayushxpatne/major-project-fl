from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd
from federated_fraud_detection import FederatedClient, FederatedServer
import tensorflow as tf

app = Flask(__name__)

# Load trained model and scaler (update path accordingly)
SERVER = None  # Should be initialized with trained model
SCALER = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    global SERVER, SCALER
    transaction_data = request.get_json()
    
    # Get input shape from model's input layer
    input_shape = SERVER.global_model.input_shape[1]  # Changed from layers[0] to model.input_shape
    
    # Create client with proper initialization
    client = FederatedClient(client_id=0, data=pd.DataFrame(), scaler=SCALER)
    client.create_model(input_shape)
    client.set_model_weights(SERVER.global_model.get_weights())
    
    probability, result = client.simulate_transaction(transaction_data)
    return jsonify({
        'probability': float(probability),
        'result': result,
        'confidence': f"{probability*100:.2f}%"
    })

def init_server(model_path='models/global_model_v10.h5', scaler_path='scaler.save'):
    global SERVER, SCALER
    SERVER = FederatedServer()
    SERVER.global_model = tf.keras.models.load_model(model_path)
    SCALER = joblib.load(scaler_path)

# Change the port number in the app.run() call
if __name__ == '__main__':
    init_server()
    app.run(host='0.0.0.0', port=5001, debug=True)  # Changed from 5000 to 5001