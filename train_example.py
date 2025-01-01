import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

# Assuming tabnet_core.py is in the same directory
from tabnet_core import TabNetEncoder

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Generate synthetic data
def generate_data(n_samples=10000, n_features=50):
    """Generate synthetic data for binary classification"""
    # Generate random features
    X = np.random.randn(n_samples, n_features)
    
    # Generate target: if sum of first 5 features > 0 and product of next 3 features > 0
    # then class 1, else class 0
    y = ((X[:, 0:5].sum(axis=1) > 0) & (X[:, 5:8].prod(axis=1) > 0)).astype(int)
    
    # Split into train and test
    train_size = int(0.8 * n_samples)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test

# Create a complete TabNet model for binary classification
class TabNetBinary(nn.Module):
    def __init__(self, input_dim, n_d=8, n_a=8):
        super(TabNetBinary, self).__init__()
        self.tabnet = TabNetEncoder(
            input_dim=input_dim,
            output_dim=1,  # binary classification
            n_d=n_d,
            n_a=n_a,
            n_steps=3,
            gamma=1.3,
            n_independent=2,
            n_shared=2,
            virtual_batch_size=128,
            momentum=0.02
        )
        self.fc = nn.Linear(n_d, 1)  # Final classification layer
        
    def forward(self, x):
        steps_output, M_loss = self.tabnet(x)
        # Use the last step output for classification
        last_step = steps_output[-1]
        out = self.fc(last_step)
        return torch.sigmoid(out.squeeze()), M_loss

# Training function
def train_model(model, train_loader, val_loader, device, epochs=100):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    criterion = nn.BCELoss()
    
    best_val_auc = 0
    patience = 10
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            y_pred, M_loss = model(batch_X)
            loss = criterion(y_pred, batch_y) + 0.001 * M_loss
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        # Validation
        model.eval()
        val_preds = []
        val_true = []
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device)
                y_pred, _ = model(batch_X)
                val_preds.extend(y_pred.cpu().numpy())
                val_true.extend(batch_y.numpy())
        
        val_auc = roc_auc_score(val_true, val_preds)
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
            
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {total_loss:.4f}, Val AUC = {val_auc:.4f}")
    
    return best_val_auc

def main():
    # Parameters
    BATCH_SIZE = 256
    N_SAMPLES = 10000
    N_FEATURES = 50
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Generate data
    print("Generating synthetic data...")
    X_train, X_test, y_train, y_test = generate_data(N_SAMPLES, N_FEATURES)
    
    # Create data loaders
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test),
        torch.FloatTensor(y_test)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    # Initialize model
    print("Initializing TabNet model...")
    model = TabNetBinary(input_dim=N_FEATURES).to(DEVICE)
    
    # Train model
    print("Starting training...")
    best_auc = train_model(model, train_loader, test_loader, DEVICE)
    print(f"Training completed. Best AUC: {best_auc:.4f}")
    
    # Final evaluation
    model.eval()
    test_preds = []
    with torch.no_grad():
        for batch_X, _ in test_loader:
            batch_X = batch_X.to(DEVICE)
            y_pred, _ = model(batch_X)
            test_preds.extend(y_pred.cpu().numpy())
    
    final_auc = roc_auc_score(y_test, test_preds)
    print(f"Final Test AUC: {final_auc:.4f}")

if __name__ == "__main__":
    main() 