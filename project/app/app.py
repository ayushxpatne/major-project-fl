import os
import sys
import json
import time
import random
import threading
import numpy as np
import pandas as pd
import torch
import joblib
from flask import Flask, render_template, request, jsonify
from datetime import datetime
import plotly
import plotly.graph_objs as go

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from project modules
from fl_system.model import FraudDetectionNet, get_model_input_features
from scripts.simulate_tx import generate_transaction, generate_batch_transactions

# Constants
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'global_model.pth')
SIMULATED_TX_PATH = os.path.join(DATA_DIR, 'simulated_tx.csv')

# Initialize Flask app
app = Flask(__name__)

# Global variables to store state
client_status = {
    'client_1': {'id': 1, 'fraud_ratio': 0.05, 'samples': 0, 'accuracy': 0.0, 'last_update': None},
    'client_2': {'id': 2, 'fraud_ratio': 0.10, 'samples': 0, 'accuracy': 0.0, 'last_update': None},
    'client_3': {'id': 3, 'fraud_ratio': 0.20, 'samples': 0, 'accuracy': 0.0, 'last_update': None}
}

# Load metrics history if available
fl_metrics = {'round': [], 'auc': [], 'f1': [], 'loss': [], 'accuracy': []}
centralized_metrics = {'auc': 0, 'f1': 0, 'precision': 0, 'recall': 0}

# Load model and preprocessing components
def load_model():
    try:
        # Load feature columns and scaler
        feature_columns = joblib.load(os.path.join(DATA_DIR, 'feature_columns.joblib'))
        scaler = joblib.load(os.path.join(DATA_DIR, 'scaler.joblib'))
        
        # Load model
        num_features = get_model_input_features()
        model = FraudDetectionNet(num_features)
        
        if os.path.exists(MODEL_PATH):
            model.load_state_dict(torch.load(MODEL_PATH))
            model.eval()
            print(f"Model loaded from {MODEL_PATH}")
        else:
            print(f"Warning: Model file not found at {MODEL_PATH}. Using untrained model.")
        
        return model, feature_columns, scaler
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None, None

# Load client data statistics
def load_client_stats():
    try:
        for client_id in range(1, 4):
            client_name = f'client_{client_id}'
            train_path = os.path.join(DATA_DIR, f'{client_name}_train.csv')
            test_path = os.path.join(DATA_DIR, f'{client_name}_test.csv')
            
            if os.path.exists(train_path):
                train_df = pd.read_csv(train_path)
                client_status[client_name]['samples'] = len(train_df)
                client_status[client_name]['fraud_ratio'] = train_df['FLAG'].mean()
            
            # Try to load latest metrics if available
            try:
                metrics_path = os.path.join(RESULTS_DIR, f'{client_name}_metrics.json')
                if os.path.exists(metrics_path):
                    with open(metrics_path, 'r') as f:
                        metrics = json.load(f)
                    client_status[client_name]['accuracy'] = metrics.get('accuracy', 0.0)
                    client_status[client_name]['last_update'] = datetime.fromtimestamp(
                        os.path.getmtime(metrics_path)).strftime('%Y-%m-%d %H:%M:%S')
            except Exception as e:
                print(f"Error loading metrics for {client_name}: {e}")
    except Exception as e:
        print(f"Error loading client stats: {e}")

# Load FL and centralized metrics
def load_metrics():
    global fl_metrics, centralized_metrics
    try:
        # Load FL metrics
        fl_metrics_path = os.path.join(RESULTS_DIR, 'fl_metrics.json')
        if os.path.exists(fl_metrics_path):
            with open(fl_metrics_path, 'r') as f:
                fl_metrics = json.load(f)
        
        # Load centralized metrics
        centralized_metrics_path = os.path.join(RESULTS_DIR, 'centralized_metrics.joblib')
        if os.path.exists(centralized_metrics_path):
            centralized_metrics = joblib.load(centralized_metrics_path)
    except Exception as e:
        print(f"Error loading metrics: {e}")

# Initialize model and data
model, feature_columns, scaler = load_model()
load_client_stats()
load_metrics()

# Simulation state
simulation_running = False
simulation_thread = None

# Routes
@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('dashboard.html', 
                          client_status=client_status,
                          fl_metrics=fl_metrics,
                          centralized_metrics=centralized_metrics)

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint to predict fraud for a transaction"""
    try:
        # Get input data
        data = request.form.to_dict()
        
        # Validate inputs
        required_fields = ['amount', 'sender', 'receiver']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Convert to appropriate types
        try:
            amount = float(data['amount'])
            if amount <= 0:
                return jsonify({'error': 'Amount must be positive'}), 400
        except ValueError:
            return jsonify({'error': 'Amount must be a number'}), 400
        
        # Generate a complete transaction from the basic inputs
        transaction = generate_transaction(
            amount=amount,
            sender_id=data['sender'],
            receiver_id=data['receiver'],
            force_fraud=False  # Let the model decide
        )
        
        # Prepare for prediction
        if model is None or feature_columns is None or scaler is None:
            return jsonify({'error': 'Model not loaded properly'}), 500
        
        # Extract features and scale
        tx_df = pd.DataFrame([transaction])
        if 'FLAG' in tx_df.columns:
            tx_df = tx_df.drop(columns=['FLAG'])
        
        # Ensure all required features are present
        for col in feature_columns:
            if col not in tx_df.columns:
                tx_df[col] = 0  # Default value for missing features
        
        # Select only the features the model knows about
        tx_df = tx_df[feature_columns]
        
        # Scale features
        numeric_features = tx_df.select_dtypes(include=np.number).columns
        tx_df[numeric_features] = scaler.transform(tx_df[numeric_features])
        
        # Convert to tensor and predict
        tx_tensor = torch.tensor(tx_df.values, dtype=torch.float32)
        with torch.no_grad():
            prediction = model(tx_tensor).item()
        
        # Return prediction
        return jsonify({
            'transaction': transaction,
            'fraud_probability': prediction,
            'is_fraud': prediction > 0.5
        })
    
    except Exception as e:
        print(f"Error in prediction: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/simulate', methods=['POST'])
def start_simulation():
    """Start generating synthetic transactions"""
    global simulation_running, simulation_thread
    
    if simulation_running:
        return jsonify({'status': 'Simulation already running'})
    
    # Start simulation in a separate thread
    simulation_running = True
    simulation_thread = threading.Thread(target=run_simulation)
    simulation_thread.daemon = True
    simulation_thread.start()
    
    return jsonify({'status': 'Simulation started'})

@app.route('/stop_simulation', methods=['POST'])
def stop_simulation():
    """Stop the transaction simulation"""
    global simulation_running
    simulation_running = False
    return jsonify({'status': 'Simulation stopped'})

@app.route('/api/client_status')
def get_client_status():
    """API endpoint to get client status for async updates"""
    return jsonify(client_status)

@app.route('/api/fl_metrics')
def get_fl_metrics():
    """API endpoint to get FL metrics for async updates"""
    # Reload metrics from disk to get latest
    load_metrics()
    
    # Create plotly figure
    fig = go.Figure()
    
    # Add FL model AUC line
    if fl_metrics['round'] and fl_metrics['auc']:
        fig.add_trace(go.Scatter(
            x=fl_metrics['round'],
            y=fl_metrics['auc'],
            mode='lines+markers',
            name='FL Model AUC'
        ))
    
    # Add centralized model reference line if available
    if centralized_metrics['auc'] > 0:
        fig.add_trace(go.Scatter(
            x=[min(fl_metrics['round']) if fl_metrics['round'] else 0, 
               max(fl_metrics['round']) if fl_metrics['round'] else 10],
            y=[centralized_metrics['auc'], centralized_metrics['auc']],
            mode='lines',
            name='Centralized Model AUC',
            line=dict(color='red', dash='dash')
        ))
    
    # Update layout
    fig.update_layout(
        title='AUC-ROC over Rounds',
        xaxis_title='Round',
        yaxis_title='AUC-ROC',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    # Convert to JSON
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    return jsonify({
        'fl_metrics': fl_metrics,
        'centralized_metrics': centralized_metrics,
        'plot': graphJSON
    })

@app.route('/api/recent_transactions')
def get_recent_transactions():
    """API endpoint to get recent transactions for display"""
    try:
        if os.path.exists(SIMULATED_TX_PATH):
            df = pd.read_csv(SIMULATED_TX_PATH)
            # Get last 10 transactions
            recent = df.tail(10).to_dict('records')
            return jsonify(recent)
        else:
            return jsonify([])
    except Exception as e:
        print(f"Error getting recent transactions: {e}")
        return jsonify([])

def run_simulation():
    """Run the transaction simulation (10 tx/sec)"""
    global simulation_running
    
    print("Starting transaction simulation...")
    
    # Create or load the simulated transactions file
    if not os.path.exists(SIMULATED_TX_PATH):
        # Create initial file with headers
        initial_transactions = generate_batch_transactions(10)
        pd.DataFrame(initial_transactions).to_csv(SIMULATED_TX_PATH, index=False)
    
    while simulation_running:
        try:
            # Generate 10 transactions (1 second worth)
            transactions = generate_batch_transactions(10)
            
            # Append to CSV
            pd.DataFrame(transactions).to_csv(SIMULATED_TX_PATH, mode='a', header=False, index=False)
            
            # Sleep for 1 second (10 tx/sec)
            time.sleep(1)
        except Exception as e:
            print(f"Error in simulation: {e}")
            time.sleep(1)
    
    print("Transaction simulation stopped.")

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Start the Flask app
    app.run(debug=True, host='0.0.0.0', port=5001)