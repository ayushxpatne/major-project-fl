from federated_fraud_detection import create_federated_setup, train_federated_model
# from visualize import plot_federated_network, plot_training_progress
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Path to your dataset
DATA_PATH = "transaction_dataset.csv"

# Hyperparameters
NUM_CLIENTS = 3
NUM_ROUNDS = 10
EPOCHS_PER_ROUND = 5
BATCH_SIZE = 32

def evaluate_model(server, clients):
    all_predictions = []
    all_true_labels = []
    
    for client in clients:
        X, y = client.preprocess_data()
        predictions = server.global_model.predict(X)
        predictions = (predictions > 0.5).astype(int)
        
        all_predictions.extend(predictions)
        all_true_labels.extend(y)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(all_true_labels, all_predictions))
    
    # Plot confusion matrix
    cm = confusion_matrix(all_true_labels, all_predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

def run_node_dashboard(client):
    dashboard = NodeDashboard(client)
    dashboard.run()

def run_central_dashboard(server):
    dashboard = FederatedDashboard()
    dashboard.create_central_dashboard()

def main():
    # Create federated setup
    print("Creating federated setup...")
    server, clients = create_federated_setup(DATA_PATH, num_clients=NUM_CLIENTS)
    
    # Add transaction simulators to clients
    for client in clients:
        client.simulator = TransactionSimulator(client)
    
    # Train model
    history = train_federated_model(
        server=server,
        clients=clients,
        num_rounds=NUM_ROUNDS,
        epochs_per_round=EPOCHS_PER_ROUND,
        batch_size=BATCH_SIZE
    )
    
    # Run dashboards
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select Dashboard", 
                           ["Central Dashboard"] + [f"Node {i}" for i in range(len(clients))])
    
    if page == "Central Dashboard":
        run_central_dashboard(server)
    else:
        node_id = int(page.split()[-1])
        run_node_dashboard(clients[node_id])
    
    # Plot network topology
    plot_federated_network(server, clients)
    
    # Plot training progress
    plot_training_progress(server, clients)
    
    # Evaluate final model
    evaluate_model(server, clients)
    
    # Print final results
    print("\nTraining completed!")
    print(f"Final accuracy: {history['global_accuracy'][-1]:.4f}")
    print(f"Final loss: {history['global_loss'][-1]:.4f}")

if __name__ == "__main__":
    main()