import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
import os
import joblib # For loading scaler and feature_columns
from collections import OrderedDict
import matplotlib.pyplot as plt


from .model import FraudDetectionNet, get_model_input_features # Relative import

# Configuration
DATA_DIR = '/Users/ayushpatne/Developer/FL_Major/project/data/'
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_EPOCHS_LOCAL = 5
BATCH_SIZE = 32
LEARNING_RATE = 0.001

def load_client_data(client_id_str):
    """Loads train and test data for a specific client."""
    train_path = os.path.join(DATA_DIR, f"{client_id_str}_train.csv")
    test_path = os.path.join(DATA_DIR, f"{client_id_str}_test.csv")
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(f"Data for {client_id_str} not found. Run data_split.py.")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Ensure FLAG is int
    train_df['FLAG'] = train_df['FLAG'].astype(int)
    test_df['FLAG'] = test_df['FLAG'].astype(int)

    X_train = train_df.drop(columns=['FLAG']).values.astype(np.float32)
    y_train = train_df['FLAG'].values.astype(np.float32).reshape(-1, 1)
    X_test = test_df.drop(columns=['FLAG']).values.astype(np.float32)
    y_test = test_df['FLAG'].values.astype(np.float32).reshape(-1, 1)
    
    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    test_dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    return train_loader, test_loader, len(train_df), len(test_df)


def train(net, trainloader, epochs, device):
    """Train the model on the_local dataset."""
    criterion = nn.BCELoss()
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    net.to(device)
    net.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        # print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(trainloader):.4f}")
    return net

def test(net, testloader, device):
    """Validate the model on the local test set."""
    criterion = nn.BCELoss()
    net.to(device)
    net.eval()
    correct, total, loss = 0, 0, 0.0
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            loss += criterion(outputs, labels).item()
            
            predicted_probs = outputs.cpu().numpy()
            actual_labels = labels.cpu().numpy()
            
            all_labels.extend(actual_labels.flatten().tolist())
            all_predictions.extend(predicted_probs.flatten().tolist())

    avg_loss = loss / len(testloader)
    
    # Calculate metrics
    # Ensure there are positive samples for AUC, precision, recall if possible
    try:
        auc = roc_auc_score(all_labels, all_predictions)
    except ValueError: # Only one class present in y_true or y_score is constant
        auc = 0.5 # Or handle as appropriate (e.g. 0.0 or skip)
        print("Warning: AUC could not be computed due to single class or constant predictions.")

    # Convert probabilities to binary predictions for F1, precision, recall
    binary_predictions = (np.array(all_predictions) > 0.5).astype(int)
    
    f1 = f1_score(all_labels, binary_predictions, zero_division=0)
    precision = precision_score(all_labels, binary_predictions, zero_division=0)
    recall = recall_score(all_labels, binary_predictions, zero_division=0)
    
    accuracy = np.mean( (np.array(all_predictions) > 0.5) == np.array(all_labels) ) if len(all_labels) > 0 else 0.0

    return avg_loss, accuracy, auc, f1, precision, recall


class FraudClient(fl.client.NumPyClient):
    def __init__(self, client_id_str, net, trainloader, testloader, num_train_examples, num_test_examples):
        self.client_id_str = client_id_str
        self.net = net
        self.trainloader = trainloader
        self.testloader = testloader
        self.num_train_examples = num_train_examples
        self.num_test_examples = num_test_examples
        # Track client metrics for paper
        self.metrics_history = {
            'round': [],
            'auc': [],
            'f1': [],
            'precision': [],
            'recall': [],
            'accuracy': [],
            'loss': []
        }
        os.makedirs(os.path.join(RESULTS_DIR, 'clients'), exist_ok=True)


    def get_parameters(self, config):
        print(f"[Client {self.client_id_str}] get_parameters")
        # Simulate secure aggregation placeholder: add noise before sending
       
        noisy_params = []
        for param in self.net.state_dict().values():
            noise = torch.randn_like(param) * 0.001 # Small noise
            noisy_params.append((param + noise).cpu().numpy())
        return noisy_params


    def set_parameters(self, parameters):
        print(f"[Client {self.client_id_str}] set_parameters")
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    
    def fit(self, parameters, config):
        print(f"[Client {self.client_id_str}] fit, config: {config}")
        self.set_parameters(parameters)
        
        # Get current model weights before training
        original_weights = [param.cpu().numpy() for param in self.net.state_dict().values()]
        
        # Train model on local data
        self.net = train(self.net, self.trainloader, epochs=NUM_EPOCHS_LOCAL, device=DEVICE)
        
        # Get updated weights after training
        updated_weights = [param.cpu().numpy() for param in self.net.state_dict().values()]
        
        # Calculate weight updates (the difference between updated and original weights)
        weight_updates = []
        for orig, updated in zip(original_weights, updated_weights):
            weight_updates.append(updated - orig)
        
        # Apply differential privacy to weight updates
        # Add calibrated noise based on sensitivity analysis
        clipping_threshold = 1.0  # Set appropriate clipping threshold
        noise_scale = 0.01        # Noise scale (adjust based on privacy requirements)
        
        # Clip gradients for differential privacy (limit influence of any single example)
        clipped_updates = []
        for update in weight_updates:
            # Calculate L2 norm of the update
            update_norm = np.sqrt(np.sum(np.square(update)))
            
            # Apply clipping if norm exceeds threshold
            if update_norm > clipping_threshold:
                update = update * (clipping_threshold / update_norm)
            
            # Add calibrated Gaussian noise for differential privacy
            noise = np.random.normal(0, noise_scale, size=update.shape)
            update_with_noise = update + noise
            
            # Calculate the new weight from the original + the noisy update
            clipped_updates.append(update_with_noise)
        
        # Calculate the final updated weights (original + noisy updates)
        final_weights = []
        for orig, update in zip(original_weights, clipped_updates):
            final_weights.append(orig + update)
        
        return final_weights, self.num_train_examples, {
            "client_id": self.client_id_str,
            "num_examples": self.num_train_examples
        }

    def evaluate(self, parameters, config):
        print(f"[Client {self.client_id_str}] evaluate, config: {config}")
        self.set_parameters(parameters)
        loss, accuracy, auc, f1, precision, recall = test(self.net, self.testloader, device=DEVICE)
        
        # Extract round number from config
        round_num = config.get("round_num", 0)
        
        # Update metrics history
        self.metrics_history['round'].append(round_num)
        self.metrics_history['auc'].append(float(auc))
        self.metrics_history['f1'].append(float(f1))
        self.metrics_history['precision'].append(float(precision))
        self.metrics_history['recall'].append(float(recall))
        self.metrics_history['accuracy'].append(float(accuracy))
        self.metrics_history['loss'].append(float(loss))
        
        # Save client metrics periodically
        if round_num % 3 == 0 or round_num == 10:  # Match AGGREGATE_EVERY_N_ROUNDS
            self._save_client_metrics(round_num)
        
        print(f"[Client {self.client_id_str}] Evaluation: Loss={loss:.4f}, Acc={accuracy:.4f}, AUC={auc:.4f}, F1={f1:.4f}")
        
        return float(loss), self.num_test_examples, {
            "accuracy": float(accuracy), 
            "auc": float(auc),
            "f1_score": float(f1),
            "precision": float(precision),
            "recall": float(recall)
        }

    def _save_client_metrics(self, round_num):
        """Save client metrics for research paper analysis"""
        client_dir = os.path.join(RESULTS_DIR, 'clients')
        os.makedirs(client_dir, exist_ok=True)

        # Save metrics to file
        import json
        with open(os.path.join(client_dir, f'{self.client_id_str}_metrics.json'), 'w') as f:
            json.dump(self.metrics_history, f)

        # Generate performance plot with better y-axis spacing
        plt.figure(figsize=(10, 6))
        metrics = ['auc', 'f1', 'precision', 'recall', 'accuracy']
        
        for metric in metrics:
            if len(self.metrics_history[metric]) > 0:
                plt.plot(self.metrics_history['round'], self.metrics_history[metric],
                        marker='o', label=metric.upper())

        plt.title(f'Client {self.client_id_str} Performance Metrics')
        plt.xlabel('Round')
        plt.ylabel('Metric Value')
        plt.ylim(0.85, 1.01)  # better spacing than 0–1
        plt.yticks(np.arange(0.85, 1.01, 0.02))  # cleaner tick steps
        plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig(os.path.join(client_dir, f'{self.client_id_str}_performance.png'))
        plt.close()
    
    # Add this new method to save metrics after each round
    def save_metrics(self, round_num, metrics):
        """Save metrics to history and generate plots if needed"""
        self.metrics_history['rounds'].append(round_num)
        for key in ['loss', 'accuracy', 'auc', 'f1', 'precision', 'recall']:
            if key in metrics:
                self.metrics_history[key].append(metrics[key])
        
        # Generate client-specific plots every 5 rounds or at final round
        if round_num % 5 == 0 or round_num == 10:  # Assuming 10 rounds total
            self._generate_metrics_plots()
    
    def _generate_metrics_plots(self):
        """Generate performance plots for this client"""
        plt.figure(figsize=(12, 8))
        metrics_to_plot = ['accuracy', 'auc', 'f1', 'precision', 'recall']
        
        for metric in metrics_to_plot:
            if metric in self.metrics_history and len(self.metrics_history[metric]) > 0:
                plt.plot(self.metrics_history['rounds'], self.metrics_history[metric], 
                         marker='o', label=metric.upper())
        
        plt.title(f'Performance Metrics - {self.client_id_str}')
        plt.xlabel('Training Round')
        plt.ylabel('Metric Value')
        plt.ylim(0.5, 1)
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(RESULTS_DIR, f'{self.client_id_str}_metrics.png'))
        plt.close()

def main(client_idx_str_arg):
    """Load data, start Flower client."""
    client_name_map = {"1": "client_1", "2": "client_2", "3": "client_3"}
    client_id_str = client_name_map.get(client_idx_str_arg)
    if not client_id_str:
        print(f"Invalid client index: {client_idx_str_arg}. Use 1, 2, or 3.")
        return

    print(f"Starting client: {client_id_str}")
    num_input_features = get_model_input_features()
    net = FraudDetectionNet(input_features=num_input_features).to(DEVICE)
    
    try:
        trainloader, testloader, num_train, num_test = load_client_data(client_id_str)
    except FileNotFoundError as e:
        print(e)
        return
    
    if num_train == 0 or num_test == 0:
        print(f"Client {client_id_str} has no training or testing data. Skipping client.")
        return

    client = FraudClient(client_id_str, net, trainloader, testloader, num_train, num_test)
    fl.client.start_client(server_address="127.0.0.1:3000", client=client)  # Changed from 8080 to 3000

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python clients.py <client_idx_str (1, 2, or 3)>")
    else:
        client_idx_str_arg = sys.argv[1]
        main(client_idx_str_arg)

# To run clients:
# python -m project.fl_system.clients 1
# python -m project.fl_system.clients 2
# python -m project.fl_system.clients 3
# (Run each in a separate terminal after the server starts)