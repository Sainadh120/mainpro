"""
Federated Learning Simulation - Working Version
Shows clear improvement over training rounds
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
NUM_CLIENTS = 5
NUM_ROUNDS = 10
LOCAL_EPOCHS = 5
NUM_FEATURES = 10
SAMPLES_PER_CLIENT = 200

# =============================================================================
# SIMPLE MODEL (No Dropout - easier to train)
# =============================================================================
class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

# =============================================================================
# DATA CREATION - Moderately difficult
# =============================================================================
def create_client_data(client_id):
    """Create moderately challenging data for each client"""
    X, y = make_classification(
        n_samples=SAMPLES_PER_CLIENT,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        n_clusters_per_class=2,
        flip_y=0.1,  # 10% noise
        class_sep=0.8,
        random_state=42 + client_id * 10
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42 + client_id
    )
    
    return {
        'X_train': torch.FloatTensor(X_train),
        'X_test': torch.FloatTensor(X_test),
        'y_train': torch.FloatTensor(y_train).unsqueeze(1),
        'y_test': torch.FloatTensor(y_test).unsqueeze(1)
    }

# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================
def train_local(model, data, epochs, lr):
    """Train model on local data"""
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    
    X = data['X_train']
    y = data['y_train']
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
    
    return model.state_dict(), loss.item()

def evaluate(model, data):
    """Evaluate model accuracy"""
    model.eval()
    with torch.no_grad():
        output = model(data['X_test'])
        predictions = (output > 0.5).float()
        accuracy = (predictions == data['y_test']).float().mean().item()
    return accuracy

def federated_average(weights_list):
    """Average model weights from all clients"""
    avg_weights = {}
    for key in weights_list[0].keys():
        avg_weights[key] = torch.stack([w[key].float() for w in weights_list]).mean(0)
    return avg_weights

# =============================================================================
# MAIN SIMULATION
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("         FEDERATED LEARNING SIMULATION")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Clients:            {NUM_CLIENTS}")
    print(f"  Rounds:             {NUM_ROUNDS}")
    print(f"  Local Epochs:       {LOCAL_EPOCHS}")
    print(f"  Samples per Client: {SAMPLES_PER_CLIENT}")
    print("=" * 70)

    # Create client datasets
    print("\n[SETUP] Creating client datasets...")
    clients = []
    for i in range(NUM_CLIENTS):
        data = create_client_data(i)
        clients.append(data)
        print(f"  Client {i+1}: {len(data['X_train'])} train, {len(data['X_test'])} test")

    # Initialize global model
    torch.manual_seed(123)
    global_model = SimpleNN()
    
    # Apply proper weight initialization
    for m in global_model.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            nn.init.constant_(m.bias, 0.01)

    # Check initial accuracy (should be ~50% random)
    print("\n[INIT] Checking initial model (before training)...")
    all_X_test = torch.cat([c['X_test'] for c in clients])
    all_y_test = torch.cat([c['y_test'] for c in clients])
    
    global_model.eval()
    with torch.no_grad():
        init_output = global_model(all_X_test)
        init_pred = (init_output > 0.5).float()
        init_acc = (init_pred == all_y_test).float().mean().item()
    print(f"  Initial Global Accuracy: {init_acc:.4f} (expected ~0.50)")

    # Storage for results
    results = {
        'round': [],
        'loss': [],
        'local_acc': [],
        'global_acc': []
    }

    print("\n" + "=" * 70)
    print("                    TRAINING PROGRESS")
    print("=" * 70)

    # Federated training loop
    for round_num in range(1, NUM_ROUNDS + 1):
        # Learning rate schedule
        lr = 0.01 * (0.9 ** (round_num - 1))
        
        client_weights = []
        client_losses = []
        client_accuracies = []
        
        # Train each client
        for i, client_data in enumerate(clients):
            # Create local model from global
            local_model = SimpleNN()
            local_model.load_state_dict(global_model.state_dict())
            
            # Local training
            weights, loss = train_local(local_model, client_data, LOCAL_EPOCHS, lr)
            client_weights.append(weights)
            client_losses.append(loss)
            
            # Evaluate locally
            local_model.load_state_dict(weights)
            acc = evaluate(local_model, client_data)
            client_accuracies.append(acc)
        
        # Aggregate weights (FedAvg)
        avg_weights = federated_average(client_weights)
        global_model.load_state_dict(avg_weights)
        
        # Evaluate global model
        global_model.eval()
        with torch.no_grad():
            output = global_model(all_X_test)
            pred = (output > 0.5).float()
            global_acc = (pred == all_y_test).float().mean().item()
        
        # Store results
        avg_loss = np.mean(client_losses)
        avg_local_acc = np.mean(client_accuracies)
        
        results['round'].append(round_num)
        results['loss'].append(avg_loss)
        results['local_acc'].append(avg_local_acc)
        results['global_acc'].append(global_acc)
        
        # Print progress
        print(f"\nRound {round_num:2d}/{NUM_ROUNDS} | LR: {lr:.4f}")
        print(f"  Avg Loss:       {avg_loss:.4f}")
        print(f"  Avg Local Acc:  {avg_local_acc:.4f}")
        print(f"  Global Acc:     {global_acc:.4f}")
        
        # Show per-client results
        for i in range(NUM_CLIENTS):
            print(f"    Client {i+1}: acc={client_accuracies[i]:.4f}, loss={client_losses[i]:.4f}")

    # =============================================================================
    # FINAL RESULTS
    # =============================================================================
    print("\n" + "=" * 70)
    print("                    TRAINING COMPLETE")
    print("=" * 70)

    print("\n" + "-" * 60)
    print(f"{'Round':<8}{'Loss':<12}{'Local Acc':<15}{'Global Acc':<15}")
    print("-" * 60)
    for i in range(NUM_ROUNDS):
        print(f"{results['round'][i]:<8}{results['loss'][i]:<12.4f}"
              f"{results['local_acc'][i]:<15.4f}{results['global_acc'][i]:<15.4f}")
    print("-" * 60)

    # Summary
    initial_acc = results['global_acc'][0]
    final_acc = results['global_acc'][-1]
    improvement = (final_acc - initial_acc) * 100
    
    initial_loss = results['loss'][0]
    final_loss = results['loss'][-1]
    loss_reduction = ((initial_loss - final_loss) / initial_loss) * 100

    print(f"\n╔{'═' * 58}╗")
    print(f"║{'FEDERATED LEARNING RESULTS':^58}║")
    print(f"╠{'═' * 58}╣")
    print(f"║  Initial Global Accuracy:   {initial_acc:.2%}{' ' * 25}║")
    print(f"║  Final Global Accuracy:     {final_acc:.2%}{' ' * 25}║")
    print(f"║  Accuracy Improvement:      +{improvement:.2f}%{' ' * 24}║")
    print(f"╠{'═' * 58}╣")
    print(f"║  Initial Loss:              {initial_loss:.4f}{' ' * 26}║")
    print(f"║  Final Loss:                {final_loss:.4f}{' ' * 26}║")
    print(f"║  Loss Reduction:            {loss_reduction:.2f}%{' ' * 25}║")
    print(f"╚{'═' * 58}╝")

    print(f"\n✅ Privacy Preserved:")
    print(f"   • Raw data stayed on client devices")
    print(f"   • Only model weights were shared")
    print(f"   • FedAvg aggregation used")
    
    print("\n" + "=" * 70)