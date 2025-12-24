"""
Grid Search for Traffic Predictor V5 Hyperparameters

Searches over:
- hidden_size: LSTM hidden units
- num_layers: Number of stacked LSTM layers
- dropout: Dropout rate
- seq_length: Input sequence length (lookback window)
- learning_rate: Initial learning rate
- batch_size: Training batch size

Results are saved to models/gridsearch_results.csv
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, mean_absolute_error
import itertools
import os
from datetime import datetime
import json


# ============================================================================
# Model Definition (same as V5)
# ============================================================================

class TrafficPredictorV5(nn.Module):
    def __init__(self, input_size=7, hidden_size=128, num_layers=2, dropout=0.2, output_size=2):
        super(TrafficPredictorV5, self).__init__()

        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.bn = nn.BatchNorm1d(hidden_size)

        self.fc = nn.Sequential(
            nn.Linear(hidden_size + 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_size)
        )

    def forward(self, x):
        last_input = x[:, -1, :2]
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        last_hidden = self.bn(last_hidden)
        combined = torch.cat([last_hidden, last_input], dim=1)
        out = self.fc(combined)
        return out


# ============================================================================
# Data Loading
# ============================================================================

def load_data(data_path, num_days=30):
    """Load and preprocess traffic data."""
    df = pd.read_csv(data_path, parse_dates=['start_time'])
    df = df.head(num_days * 24 * 12)  # num_days of 5-min intervals

    feature_cols = ['cpu', 'memory', 'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'is_weekend']

    scaler = RobustScaler()
    scaled_cpu_mem = scaler.fit_transform(df[['cpu', 'memory']].values)

    features_normalized = np.column_stack([
        scaled_cpu_mem,
        df[['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'is_weekend']].values
    ])

    return features_normalized, scaler


def create_sequences(data, seq_length=6, stride=1):
    """Create sequences for training."""
    X, y = [], []
    for i in range(0, len(data) - seq_length, stride):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length, :2])
    return np.array(X), np.array(y)


# ============================================================================
# Training Function
# ============================================================================

def train_model(
    X_train, y_train, X_test, y_test,
    hidden_size, num_layers, dropout, learning_rate, batch_size,
    device, num_epochs=50, early_stop_patience=15, verbose=False
):
    """Train a single model configuration and return metrics."""

    # Create data loaders
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.FloatTensor(y_train).to(device)
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    y_test_tensor = torch.FloatTensor(y_test).to(device)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    input_size = X_train.shape[2]
    model = TrafficPredictorV5(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout
    ).to(device)

    criterion = nn.HuberLoss(delta=1.0)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)

    # Training loop
    best_test_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for batch_X, batch_y in train_loader:
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()

        scheduler.step()
        avg_train_loss = train_loss / len(train_loader)

        # Eval
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                predictions = model(batch_X)
                loss = criterion(predictions, batch_y)
                test_loss += loss.item()

        avg_test_loss = test_loss / len(test_loader)

        if verbose and (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}: Train={avg_train_loss:.6f}, Test={avg_test_loss:.6f}")

        # Early stopping
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                break

    # Load best model and compute final metrics
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_tensor).cpu().numpy()
        y_true = y_test_tensor.cpu().numpy()

    r2_cpu = r2_score(y_true[:, 0], y_pred[:, 0])
    r2_mem = r2_score(y_true[:, 1], y_pred[:, 1])
    mae_cpu = mean_absolute_error(y_true[:, 0], y_pred[:, 0])
    mae_mem = mean_absolute_error(y_true[:, 1], y_pred[:, 1])

    total_params = sum(p.numel() for p in model.parameters())

    return {
        'best_test_loss': best_test_loss,
        'r2_cpu': r2_cpu,
        'r2_memory': r2_mem,
        'mae_cpu': mae_cpu,
        'mae_memory': mae_mem,
        'total_params': total_params,
        'epochs_trained': epoch + 1
    }


# ============================================================================
# Grid Search
# ============================================================================

def run_gridsearch(
    features_normalized,
    param_grid,
    device,
    num_epochs=50,
    early_stop_patience=15,
    verbose=True
):
    """Run grid search over hyperparameter combinations."""

    # Generate all combinations
    keys = list(param_grid.keys())
    combinations = list(itertools.product(*param_grid.values()))
    total_combinations = len(combinations)

    print(f"Total combinations to test: {total_combinations}")
    print(f"Parameters: {keys}")
    print("=" * 60)

    results = []

    for i, values in enumerate(combinations):
        params = dict(zip(keys, values))

        print(f"\n[{i+1}/{total_combinations}] Testing: {params}")

        # Create sequences with current seq_length
        seq_length = params.get('seq_length', 6)
        X, y = create_sequences(features_normalized, seq_length=seq_length)

        # Split data
        train_size = int(0.8 * len(X))
        indices = np.random.permutation(len(X))
        train_indices = indices[:train_size]
        test_indices = indices[train_size:]

        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]

        # Train model
        try:
            metrics = train_model(
                X_train, y_train, X_test, y_test,
                hidden_size=params.get('hidden_size', 128),
                num_layers=params.get('num_layers', 2),
                dropout=params.get('dropout', 0.2),
                learning_rate=params.get('learning_rate', 0.002),
                batch_size=params.get('batch_size', 64),
                device=device,
                num_epochs=num_epochs,
                early_stop_patience=early_stop_patience,
                verbose=verbose
            )

            # Combine params and metrics
            result = {**params, **metrics}
            results.append(result)

            print(f"  -> R² CPU: {metrics['r2_cpu']:.4f}, R² Mem: {metrics['r2_memory']:.4f}, "
                  f"Loss: {metrics['best_test_loss']:.6f}")

        except Exception as e:
            print(f"  -> FAILED: {e}")
            result = {**params, 'error': str(e)}
            results.append(result)

    return results


# ============================================================================
# Main
# ============================================================================

def main():
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    data_path = 'data/processed/traffic_data_processed.csv'
    print(f"Loading data from {data_path}...")
    features_normalized, scaler = load_data(data_path, num_days=30)
    print(f"Data shape: {features_normalized.shape}")

    # Define parameter grid
    param_grid = {
        'hidden_size': [64, 128, 256],
        'num_layers': [1, 2, 3],
        'dropout': [0.1, 0.2, 0.3],
        'seq_length': [6, 12, 24],
        'learning_rate': [0.001, 0.002, 0.005],
        'batch_size': [32, 64, 128],
    }

    # For quick testing, use a smaller grid:
    # param_grid = {
    #     'hidden_size': [64, 128],
    #     'num_layers': [2],
    #     'dropout': [0.2],
    #     'seq_length': [6, 12],
    #     'learning_rate': [0.002],
    #     'batch_size': [64],
    # }

    # Run grid search
    print("\nStarting grid search...")
    results = run_gridsearch(
        features_normalized,
        param_grid,
        device,
        num_epochs=50,
        early_stop_patience=15,
        verbose=False
    )

    # Save results
    os.makedirs('models', exist_ok=True)
    results_df = pd.DataFrame(results)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = f'models/gridsearch_results_{timestamp}.csv'
    results_df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    # Also save as latest
    results_df.to_csv('models/gridsearch_results.csv', index=False)

    # Print best results
    print("\n" + "=" * 60)
    print("TOP 5 CONFIGURATIONS (by R² CPU):")
    print("=" * 60)

    if 'r2_cpu' in results_df.columns:
        top5 = results_df.nlargest(5, 'r2_cpu')
        for i, row in top5.iterrows():
            print(f"\nRank {top5.index.get_loc(i) + 1}:")
            print(f"  hidden_size={row.get('hidden_size')}, num_layers={row.get('num_layers')}, "
                  f"dropout={row.get('dropout')}, seq_length={row.get('seq_length')}")
            print(f"  learning_rate={row.get('learning_rate')}, batch_size={row.get('batch_size')}")
            print(f"  R² CPU: {row.get('r2_cpu', 'N/A'):.4f}, R² Memory: {row.get('r2_memory', 'N/A'):.4f}")
            print(f"  MAE CPU: {row.get('mae_cpu', 'N/A'):.6f}, MAE Memory: {row.get('mae_memory', 'N/A'):.6f}")
            print(f"  Test Loss: {row.get('best_test_loss', 'N/A'):.6f}, Params: {row.get('total_params', 'N/A'):,}")

    # Save best config
    if len(results_df) > 0 and 'r2_cpu' in results_df.columns:
        best_row = results_df.loc[results_df['r2_cpu'].idxmax()]
        best_config = {
            'hidden_size': int(best_row['hidden_size']),
            'num_layers': int(best_row['num_layers']),
            'dropout': float(best_row['dropout']),
            'seq_length': int(best_row['seq_length']),
            'learning_rate': float(best_row['learning_rate']),
            'batch_size': int(best_row['batch_size']),
            'r2_cpu': float(best_row['r2_cpu']),
            'r2_memory': float(best_row['r2_memory']),
        }

        with open('models/best_config.json', 'w') as f:
            json.dump(best_config, f, indent=2)
        print(f"\nBest config saved to models/best_config.json")

    return results_df


if __name__ == '__main__':
    results = main()
