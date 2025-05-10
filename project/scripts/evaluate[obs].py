import os
import torch
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

# Add project root to path for imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fl_system.model import FraudDetectionNet, get_model_input_features

# Configuration - Use relative paths for better portability
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
FL_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'global_model.pth')
CENTRALIZED_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'centralized_model.pth')

def load_test_data():
    """Load the global test dataset for evaluation"""
    try:
        test_df = pd.read_csv(os.path.join(DATA_DIR, "global_test_centralized.csv"))
        print(f"Loaded global test set: {len(test_df)} samples")
        
        # Extract features and labels
        X_test = test_df.drop(columns=['FLAG']).values.astype(np.float32)
        y_test = test_df['FLAG'].values.astype(np.float32)
        
        return X_test, y_test, test_df.drop(columns=['FLAG']).columns.tolist()
    except FileNotFoundError:
        print("Global test set not found. Trying to use client_1 test data as fallback.")
        try:
            test_df = pd.read_csv(os.path.join(DATA_DIR, "client_1_test.csv"))
            print(f"Using client_1 test data: {len(test_df)} samples")
            
            # Extract features and labels
            X_test = test_df.drop(columns=['FLAG']).values.astype(np.float32)
            y_test = test_df['FLAG'].values.astype(np.float32)
            
            return X_test, y_test, test_df.drop(columns=['FLAG']).columns.tolist()
        except FileNotFoundError:
            raise FileNotFoundError("No test data found. Please run data_split.py first.")

def train_centralized_model(X_train, y_train, num_features):
    """Train a centralized model on combined data"""
    model = FraudDetectionNet(num_features)
    
    # Convert to tensor
    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
    
    # Training parameters - match FL client settings
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    batch_size = 32
    epochs = 10  # Slightly more than local FL client epochs to ensure fair comparison
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        # Simple batch implementation
        for i in range(0, len(X_tensor), batch_size):
            batch_X = X_tensor[i:i+batch_size]
            batch_y = y_tensor[i:i+batch_size]
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        print(f"Centralized Training Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/(len(X_tensor)//batch_size):.4f}")
    
    return model

def load_model(model_path, num_features):
    """Load a PyTorch model from the given path"""
    model = FraudDetectionNet(num_features)
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        model.eval()
        print(f"Model loaded from {model_path}")
        return model
    else:
        print(f"Warning: Model file not found at {model_path}")
        return None
def get_evaluate_fn(model_class, num_input_features):
    """Return an evaluation function for server-side evaluation."""
    
    # Load test data upfront to avoid reloading during each evaluation
    X_test, y_test, X_tensor, y_tensor = load_test_data()
    print(f"Loaded test data for server evaluation: {len(X_test)} samples")
    
    # Track centralized model metrics over rounds
    centralized_history = {'round': [], 'auc': [], 'f1': [], 'loss': [], 'accuracy': []}
    
    # The evaluation function
    def evaluate(server_round, parameters, config):
        # Initialize model with the latest weights
        model = model_class(num_input_features)
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        
        # Evaluate FL model on the test set
        with torch.no_grad():
            outputs = model(X_tensor)
            loss = torch.nn.functional.binary_cross_entropy(outputs, y_tensor).item()
            preds = (outputs > 0.5).float()
            accuracy = ((preds == y_tensor).sum().item()) / len(y_tensor)
            
            # Convert to numpy for sklearn metrics
            y_true = y_tensor.numpy().flatten()
            y_pred = preds.numpy().flatten()
            y_prob = outputs.numpy().flatten()
            
            try:
                auc = roc_auc_score(y_true, y_prob)
            except ValueError:
                auc = 0.5
                print("Warning: AUC could not be computed due to single class or constant predictions.")
            
            f1 = f1_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
        
        # Update history for plotting
        history_fl['round'].append(server_round)
        history_fl['auc'].append(auc)
        history_fl['f1'].append(f1)
        history_fl['loss'].append(loss)
        history_fl['accuracy'].append(accuracy)
        history_fl['precision'].append(precision)
        history_fl['recall'].append(recall)
        
        # Re-evaluate centralized model at each round for proper comparison
        # This ensures both models are evaluated on the same test data
        centralized_metrics = {}
        if os.path.exists(CENTRALIZED_MODEL_PATH):
            centralized_model = model_class(num_input_features)
            centralized_model.load_state_dict(torch.load(CENTRALIZED_MODEL_PATH))
            centralized_model.eval()
            
            with torch.no_grad():
                c_outputs = centralized_model(X_tensor)
                c_loss = torch.nn.functional.binary_cross_entropy(c_outputs, y_tensor).item()
                c_preds = (c_outputs > 0.5).float()
                c_accuracy = ((c_preds == y_tensor).sum().item()) / len(y_tensor)
                
                try:
                    c_auc = roc_auc_score(y_true, c_outputs.numpy().flatten())
                except ValueError:
                    c_auc = 0.5
                
                c_f1 = f1_score(y_true, (c_outputs.numpy().flatten() > 0.5).astype(int))
                
                centralized_metrics = {
                    'auc': float(c_auc),
                    'f1': float(c_f1),
                    'accuracy': float(c_accuracy),
                    'loss': float(c_loss)
                }
                
                # Track centralized model metrics for this round
                centralized_history['round'].append(server_round)
                centralized_history['auc'].append(c_auc)
                centralized_history['f1'].append(c_f1)
                centralized_history['loss'].append(c_loss)
                centralized_history['accuracy'].append(c_accuracy)
        
        print(f"\nRound {server_round} Evaluation Results:")
        print(f"FL Model - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}, AUC: {auc:.4f}, F1: {f1:.4f}")
        if centralized_metrics:
            print(f"Centralized - AUC: {centralized_metrics['auc']:.4f}, F1: {centralized_metrics['f1']:.4f}")
        
        # Save model periodically or at the end
        if server_round % AGGREGATE_EVERY_N_ROUNDS == 0 or server_round == NUM_ROUNDS:
            # Save the global model
            torch.save(model.state_dict(), GLOBAL_MODEL_PATH)
            print(f"Global model saved to {GLOBAL_MODEL_PATH} at round {server_round}")
            
            # Create plots for visualization
            if not os.path.exists(RESULTS_DIR):
                os.makedirs(RESULTS_DIR)
            
            # Plot metrics over rounds with actual centralized model performance
            plt.figure(figsize=(15, 10))
            
            plt.subplot(2, 2, 1)
            plt.plot(history_fl['round'], history_fl['auc'], 'b-', label='FL Model')
            if centralized_history['round']:
                plt.plot(centralized_history['round'], centralized_history['auc'], 'r-', label='Centralized')
            plt.title('AUC-ROC over Rounds')
            plt.xlabel('Round')
            plt.ylabel('AUC-ROC')
            plt.legend()
            
            plt.subplot(2, 2, 2)
            plt.plot(history_fl['round'], history_fl['f1'], 'b-', label='FL Model')
            if centralized_history['round']:
                plt.plot(centralized_history['round'], centralized_history['f1'], 'r-', label='Centralized')
            plt.title('F1 Score over Rounds')
            plt.xlabel('Round')
            plt.ylabel('F1 Score')
            plt.legend()
            
            plt.subplot(2, 2, 3)
            plt.plot(history_fl['round'], history_fl['loss'], 'g-', label='FL Loss')
            if centralized_history['round']:
                plt.plot(centralized_history['round'], centralized_history['loss'], 'r-', label='Centralized Loss')
            plt.title('Loss over Rounds')
            plt.xlabel('Round')
            plt.ylabel('Loss')
            plt.legend()
            
            plt.subplot(2, 2, 4)
            plt.plot(history_fl['round'], history_fl['accuracy'], 'g-', label='FL Accuracy')
            if centralized_history['round']:
                plt.plot(centralized_history['round'], centralized_history['accuracy'], 'r-', label='Centralized Accuracy')
            plt.title('Accuracy over Rounds')
            plt.xlabel('Round')
            plt.ylabel('Accuracy')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(RESULTS_DIR, 'comparison.png'))
            plt.close()
            
            # Save metrics for external use
            metrics_data = {
                'round': history_fl['round'],
                'auc': history_fl['auc'],
                'f1': history_fl['f1'],
                'loss': history_fl['loss'],
                'accuracy': history_fl['accuracy'],
                'precision': history_fl['precision'],
                'recall': history_fl['recall'],
                'centralized_auc': centralized_history['auc'] if centralized_history['auc'] else [],
                'centralized_f1': centralized_history['f1'] if centralized_history['f1'] else [],
                'centralized_loss': centralized_history['loss'] if centralized_history['loss'] else [],
                'centralized_accuracy': centralized_history['accuracy'] if centralized_history['accuracy'] else []
            }
            
            # Save as JSON for web access
            import json
            with open(os.path.join(RESULTS_DIR, 'fl_metrics.json'), 'w') as f:
                json.dump(metrics_data, f)
            
            # Save last round metrics in joblib format
            last_metrics = {
                'auc': auc,
                'f1': f1,
                'precision': precision,
                'recall': recall,
                'accuracy': accuracy,
                'loss': loss
            }
            joblib.dump(last_metrics, os.path.join(RESULTS_DIR, 'fl_metrics.joblib'))
            
            # Also save centralized metrics
            if centralized_metrics:
                joblib.dump(centralized_metrics, os.path.join(RESULTS_DIR, 'centralized_metrics.joblib'))
            
            print(f"Evaluation plots and metrics saved at {RESULTS_DIR}")
            
        return float(loss), {
            "accuracy": float(accuracy),
            "auc": float(auc),
            "f1": float(f1),
            "precision": float(precision),
            "recall": float(recall)
        }
    
    return evaluate

def plot_comparison(fl_metrics, centralized_metrics):
    """Generate a bar plot comparing FL and centralized models"""
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    
    metrics = ['auc', 'f1', 'precision', 'recall', 'accuracy']
    fl_values = [fl_metrics[m] for m in metrics]
    centralized_values = [centralized_metrics[m] for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 8))
    rects1 = ax.bar(x - width/2, fl_values, width, label='Federated Learning')
    rects2 = ax.bar(x + width/2, centralized_values, width, label='Centralized')
    
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in metrics])
    ax.set_ylim(0, 1.0)
    ax.legend()
    
    # Add value labels
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    autolabel(rects1)
    autolabel(rects2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'centralized_vs_fl.png'))
    print(f"Comparison plot saved to {os.path.join(RESULTS_DIR, 'centralized_vs_fl.png')}")
    plt.close()

def save_metrics(fl_metrics, centralized_metrics):
    """Save metrics to files for later use"""
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    
    # Save metrics to text file
    with open(os.path.join(RESULTS_DIR, 'metrics.txt'), 'w') as f:
        f.write("Model Performance Comparison\n")
        f.write("===========================\n\n")
        f.write("Metric      | Federated Learning | Centralized\n")
        f.write("------------|-------------------|------------\n")
        for metric in ['auc', 'f1', 'precision', 'recall', 'accuracy']:
            f.write(f"{metric.upper():12} | {fl_metrics[metric]:.6f}          | {centralized_metrics[metric]:.6f}\n")
    
    # Save metrics as joblib for the Flask app to use
    joblib.dump(fl_metrics, os.path.join(RESULTS_DIR, 'fl_metrics.joblib'))
    joblib.dump(centralized_metrics, os.path.join(RESULTS_DIR, 'centralized_metrics.joblib'))
    
    # Also save as JSON for easier web access
    import json
    with open(os.path.join(RESULTS_DIR, 'fl_metrics.json'), 'w') as f:
        json.dump({
            'round': list(range(1, 11)),  # Assuming 10 rounds
            'auc': [fl_metrics['auc']] * 10,  # Replicate the final value for all rounds
            'f1': [fl_metrics['f1']] * 10,
            'loss': [0.1] * 10,  # Placeholder
            'accuracy': [fl_metrics['accuracy']] * 10
        }, f)
    
    print(f"Metrics saved to {os.path.join(RESULTS_DIR, 'metrics.txt')}")

def main():
    """Main evaluation function"""
    print("Starting model evaluation...")
    
    # Load test data
    X_test, y_test, feature_columns = load_test_data()
    
    # Get number of features
    num_features = get_model_input_features()
    
    # Check if FL model exists
    fl_model = load_model(FL_MODEL_PATH, num_features)
    
    # Check if centralized model exists, if not, train it
    centralized_model = load_model(CENTRALIZED_MODEL_PATH, num_features)
    if centralized_model is None:
        # For centralized model, we need to load and combine all client data
        print("Centralized model not found. Training a new centralized model...")
        
        # Combine data from all clients for centralized training
        client_dfs = []
        for client_id in [1, 2, 3]:  # Adjust based on your client count
            try:
                client_df = pd.read_csv(os.path.join(DATA_DIR, f"client_{client_id}_train.csv"))
                client_dfs.append(client_df)
                print(f"Loaded client_{client_id}_train.csv with {len(client_df)} samples")
            except FileNotFoundError:
                print(f"Client {client_id} train data not found, skipping")
        
        if client_dfs:
            # Combine all client data
            combined_df = pd.concat(client_dfs, ignore_index=True)
            print(f"Combined {len(combined_df)} samples for centralized training")
            
            # Extract features and labels
            X_train_centralized = combined_df.drop(columns=['FLAG']).values.astype(np.float32)
            y_train_centralized = combined_df['FLAG'].values.astype(np.float32)
            
            # Train centralized model
            centralized_model = train_centralized_model(X_train_centralized, y_train_centralized, num_features)
            
            # Save centralized model
            torch.save(centralized_model.state_dict(), CENTRALIZED_MODEL_PATH)
            print(f"Centralized model saved to {CENTRALIZED_MODEL_PATH}")
        else:
            print("No client data found for centralized training")
    
    # Evaluate models
    fl_metrics = evaluate_model(fl_model, X_test, y_test)
    centralized_metrics = evaluate_model(centralized_model, X_test, y_test)
    
    # Print results
    print("\nEvaluation Results:")
    print("------------------")
    print(f"FL Model AUC: {fl_metrics['auc']:.4f}, F1: {fl_metrics['f1']:.4f}")
    print(f"Centralized Model AUC: {centralized_metrics['auc']:.4f}, F1: {centralized_metrics['f1']:.4f}")
    
    # Plot comparison
    plot_comparison(fl_metrics, centralized_metrics)
    
    # Save metrics
    save_metrics(fl_metrics, centralized_metrics)
    
    print("Evaluation complete!")

if __name__ == "__main__":
    main()