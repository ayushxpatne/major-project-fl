import numpy as np
import pandas as pd
import os  # Add this import
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, Sequential
from typing import List, Dict
# # Add to FederatedClient __init__
import joblib
import tensorflow as tf
import json
from datetime import datetime

# Add new class method to FederatedClient
class FederatedClient:
    def __init__(self, client_id, data, model_fn):
        self.client_id = client_id
        self.model = model_fn()
        self.scaler = StandardScaler()
        # self.feature_names_in_ will be set by preprocess_data
        self.X_train, self.y_train, self.X_test, self.y_test = self.preprocess_data(data)

    def save_scaler(self, path="scaler.save"):
        """Properly save the fitted scaler"""
        joblib.dump(self.scaler, path)      
        
    def preprocess_data(self, data):
        df = pd.DataFrame(data)
        
        if 'Class' not in df.columns:
            # This is critical for training. If 'Class' is missing, data might be malformed.
            raise ValueError(f"Target column 'Class' not found in data for client {self.client_id}.")
            
        X_unscaled_df = df.drop('Class', axis=1)
        y_series = df['Class']
        
        # Store the feature names in the order they appear in the unscaled data
        self.feature_names_in_ = list(X_unscaled_df.columns)
        
        # Fit the scaler ONLY on the training portion of this client's data
        # For simplicity here, fitting on all of X_unscaled_df for this client.
        # In a more rigorous setup, you might fit only on the client's training split
        # if you were to split before fitting. However, fitting on all client's X is common.
        self.scaler.fit(X_unscaled_df)
        
        # Transform the features using the fitted scaler
        X_scaled_np = self.scaler.transform(X_unscaled_df)
        
        # Split the scaled data and the original y_series (target)
        # Using y_series.values to ensure it's a NumPy array for train_test_split
        X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(
            X_scaled_np, y_series.values, test_size=0.2, random_state=42
        )
        
        return X_train_np, y_train_np, X_test_np, y_test_np
    
    def create_model(self, input_shape: int):
        model = Sequential([
            layers.Input(shape=(input_shape,)),
            layers.Dense(64, activation='relu', kernel_regularizer='l2'),
            layers.Dropout(0.5),
            layers.Dense(32, activation='relu', kernel_regularizer='l2'),
            layers.Dropout(0.3),
            layers.Dense(16, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy', 
                              tf.keras.metrics.Precision(),
                              tf.keras.metrics.Recall()])
        
        self.model = model
        return model
    
    def train_local_model(self, epochs=5, batch_size=32):
        print(f"\nClient {self.client_id} training on local data...")
        X, y = self.preprocess_data()
        if not self.model:
            self.create_model(X.shape[1])
        
        history = self.model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=1)
        self.training_history.append(history.history)
        return history.history
    
    def get_weights(self) -> List:
        return self.model.get_weights()
    
    def set_model_weights(self, weights: List):
        """Set the weights of the local model."""
        self.model.set_weights(weights)
    
    def simulate_transaction(self, transaction_data):
        # transaction_data is a dictionary from the Flask app, e.g.,
        # {'Transaction Frequency (per hour)': 10, 'Average Time Between Transactions (min)': 5, ...}

        if not hasattr(self, 'feature_names_in_') or not self.feature_names_in_:
            # This should not happen if the client was initialized correctly.
            # Log an error or handle as appropriate.
            print(f"Error: Client {self.client_id} feature names not initialized for scaling.")
            # Fallback: try to use keys from transaction_data, but this is risky
            # as order and completeness are not guaranteed.
            # For robustness, this case should lead to an error or a default prediction.
            # For now, let's attempt to construct based on transaction_data keys if feature_names_in_ is missing.
            # This is NOT recommended for production.
            print("Warning: Falling back to using transaction_data keys for feature names. This may be unreliable.")
            current_features_from_input = list(transaction_data.keys())
            X_for_transform = pd.DataFrame([transaction_data], columns=current_features_from_input)
        else:
            # Create a DataFrame with columns in the order self.scaler expects (self.feature_names_in_)
            # Populate with values from transaction_data, using None for any missing features.
            ordered_transaction_values = {
                feature: transaction_data.get(feature) for feature in self.feature_names_in_
            }
            X_for_transform = pd.DataFrame([ordered_transaction_values], columns=self.feature_names_in_)

            # Optional: Check for and handle NaNs if any feature was missing in transaction_data
            # and transaction_data.get(feature) returned None (its default).
            if X_for_transform.isnull().values.any():
                missing_cols = X_for_transform.columns[X_for_transform.isnull().any()].tolist()
                print(f"Warning: Missing values for features: {missing_cols} in transaction data for client {self.client_id}. Prediction may be unreliable.")
                # Depending on requirements, you might impute here, or raise an error,
                # or let scaler.transform handle it (it might raise an error if it wasn't fit with NaNs).
                # For now, we'll proceed, but this is a point of potential failure or inaccuracy.

        # Transform the data using the fitted scaler
        X_scaled = self.scaler.transform(X_for_transform)
        
        # Make prediction
        # Assuming binary classification and model has predict_proba
        proba_class_1 = self.model.predict_proba(X_scaled)[:, 1] 
        probability = proba_class_1[0]  # Get the probability for the single transaction

        # Determine result based on a threshold (e.g., 0.5)
        threshold = 0.5 
        result = "Fraud" if probability > threshold else "Not Fraud"

        return probability, result

# Add new imports at the top
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
import seaborn as sns

class FederatedServer:
    def __init__(self):
        self.global_model = None
        self.clients: List[FederatedClient] = []
        self.model_version = 0
        self.training_history = []
        self.metrics = {
            'global_metrics': {'rounds': [], 'accuracy': [], 'loss': []},
            'node_metrics': {},
            'dataset_stats': {}
        }

    def add_client(self, client: FederatedClient):
        self.clients.append(client)
        
    def initialize_global_model(self, input_shape: int):
        model = Sequential([
            layers.Dense(64, activation='relu', input_shape=(input_shape,)),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        
        self.global_model = model
        
    def aggregate_weights(self, client_weights: List[List]) -> List:
        # Simple averaging of weights
        avg_weights = []
        for weights_list_tuple in zip(*client_weights):
            avg_weights.append(
                np.array([np.array(w).mean(axis=0) for w in zip(*weights_list_tuple)])
            )
        return avg_weights
    
    def save_model(self, path="models"):
        if not os.path.exists(path):
            os.makedirs(path)
        self.global_model.save(f"{path}/global_model_v{self.model_version}.h5")
        print(f"Global model saved: version {self.model_version}")
        
    def train_round(self, epochs=5, batch_size=32):
        print("\n=== Starting New Federated Round ===")
        print(f"Current Model Version: {self.model_version}")
        
        # Train local models
        local_weights = []
        local_performances = []
        
        for client in self.clients:
            print(f"\nTraining Client {client.client_id}")
            client.train_local_model(epochs=epochs, batch_size=batch_size)
            local_weights.append(client.get_weights())  # Changed from get_model_weights to get_weights
            
            # Evaluate client's performance
            X, y = client.preprocess_data()
            metrics = client.model.evaluate(X, y, verbose=0)
            local_performances.append({
                'client_id': client.client_id,
                'loss': metrics[0],
                'accuracy': metrics[1]
            })
            print(f"Client {client.client_id} - Loss: {metrics[0]:.4f}, Accuracy: {metrics[1]:.4f}")
        
        # Aggregate weights
        print("\nAggregating models from all clients...")
        global_weights = self.aggregate_weights(local_weights)
        
        # Update global model
        self.global_model.set_weights(global_weights)
        self.model_version += 1
        
        # Save updated model
        self.save_model()
        
        # Distribute new weights to clients
        print("\nDistributing updated model to all clients...")
        for client in self.clients:
            client.set_model_weights(global_weights)
            
        # Update metrics
        self._update_metrics(local_performances)
        return global_weights, local_performances

    def _update_metrics(self, performances):
        self.metrics['global_metrics']['rounds'].append(self.model_version)
        self.metrics['global_metrics']['accuracy'].append(
            sum(p['accuracy'] for p in performances)/len(performances))
        self.metrics['global_metrics']['loss'].append(
            sum(p['loss'] for p in performances)/len(performances))
        
        # Add visualization
        self._plot_training_progress()
    
    def _plot_training_progress(self):
        """Generate and save training progress visualization"""
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(self.metrics['global_metrics']['rounds'], 
                self.metrics['global_metrics']['accuracy'], 'b-', label='Global Accuracy')
        plt.xlabel('Federated Rounds')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(self.metrics['global_metrics']['rounds'], 
                self.metrics['global_metrics']['loss'], 'r--', label='Global Loss')
        plt.xlabel('Federated Rounds')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.savefig(f'figures/training_progress_v{self.model_version}.png')
        plt.close()
        
        # Save to file
        with open('metrics.json', 'w') as f:
            json.dump(self.metrics, f)

    def generate_research_visuals(self):
        """Generate all visuals needed for research paper"""
        self._plot_class_distribution()
        self._plot_roc_curve()
        self._plot_confusion_matrix()
    
    def _plot_class_distribution(self):
        """Plot data distribution across clients"""
        plt.figure(figsize=(10, 6))
        client_data = [len(client.data) for client in self.clients]
        plt.bar(range(len(client_data)), client_data)
        plt.title('Data Distribution Across Clients')
        plt.xlabel('Client ID')
        plt.ylabel('Number of Transactions')
        plt.savefig('figures/class_distribution.png')
        plt.close()
    
    def _plot_roc_curve(self):
        """Generate ROC curve for final model"""
        all_y = []
        all_preds = []
        for client in self.clients:
            X, y = client.preprocess_data()
            preds = self.global_model.predict(X).ravel()
            all_y.extend(y)
            all_preds.extend(preds)
        
        fpr, tpr, _ = roc_curve(all_y, all_preds)
        roc_auc = auc(fpr, tpr)
        
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.savefig('figures/roc_curve.png')
        plt.close()

def create_federated_setup(data_path: str, num_clients: int = 3):
    # Load and shuffle data
    df = pd.read_csv(data_path)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Split data into chunks for each client
    chunk_size = len(df) // num_clients
    client_data = [df[i:i + chunk_size] for i in range(0, len(df), chunk_size)]
    
    # Create clients
    clients = []
    scaler = StandardScaler()
    for i, data in enumerate(client_data[:num_clients]):
        client = FederatedClient(i, data, scaler)
        clients.append(client)
    
    # Setup server
    server = FederatedServer()
    for client in clients:
        server.add_client(client)
    
    # Initialize global model
    X = df.select_dtypes(include=['float64', 'int64']).fillna(0)
    y = X.pop('FLAG')
    input_shape = X.shape[1]
    server.initialize_global_model(input_shape)
    
    # After creating clients
    if clients:  # Save scaler after first client processes data
        clients[0].save_scaler()
    
    return server, clients


def train_federated_model(server: FederatedServer, clients: List[FederatedClient], num_rounds: int = 10, epochs_per_round: int = 5, batch_size: int = 32):
    history = {
        'rounds': [],
        'global_loss': [],
        'global_accuracy': []
    }
    
    for round_num in range(num_rounds):
        print(f"\nFederated Round {round_num + 1}/{num_rounds}")
        
        # Train one round and get updated weights
        weights, performances = server.train_round(epochs=epochs_per_round, batch_size=batch_size)  # Unpack both values
        
        # Evaluate global model on each client's data
        total_loss = 0
        total_accuracy = 0
        for client in clients:
            X, y = client.preprocess_data()
            metrics = client.model.evaluate(X, y, verbose=0)
            total_loss += metrics[0]
            total_accuracy += metrics[1]
        
        # Calculate average metrics
        avg_loss = total_loss / len(clients)
        avg_accuracy = total_accuracy / len(clients)
        
        print(f"Average Loss: {avg_loss:.4f}")
        print(f"Average Accuracy: {avg_accuracy:.4f}")
        
        # Store metrics
        history['rounds'].append(round_num + 1)
        history['global_loss'].append(avg_loss)
        history['global_accuracy'].append(avg_accuracy)
    
    return history


def update_metrics(self, metrics):
    """Update metrics file with current training progress."""
    current_metrics = {
        'global_metrics': {
            'accuracy': metrics['global_accuracy'],
            'loss': metrics['global_loss'],
            'rounds': metrics['rounds'],
            'timestamp': datetime.now().isoformat()
        },
        'node_metrics': {},
        'dataset_stats': {
            'node_distribution': [
                {'Node': f'Node {i}', 'Samples': len(client.data)} 
                for i, client in enumerate(self.clients)
            ],
            'class_distribution': [
                {'Class': 'Normal', 'Count': sum(client.data['FLAG'] == 0 for client in self.clients)},
                {'Class': 'Fraud', 'Count': sum(client.data['FLAG'] == 1 for client in self.clients)}
            ]
        }
    }
    
    # Update node metrics
    for i, client in enumerate(self.clients):
        current_metrics['node_metrics'][f'node_{i}'] = {
            'accuracy': client.training_history[-1]['accuracy'][-1] if client.training_history else 0,
            'loss': client.training_history[-1]['loss'][-1] if client.training_history else 0
        }
    
    with open('metrics.json', 'w') as f:
        json.dump(current_metrics, f)

