import numpy as np
import joblib
import torch
from src.modeling.model import TrafficPredictorV2

class ModelWrapper:
    """
    A wrapper class to load a trained PyTorch LSTM model and its scaler,
    and provide a simple interface for making predictions.
    """
    def __init__(self, model_path: str, scaler_path: str, sequence_length: int = 6):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.scaler = None
        self.is_loaded = False
        self.sequence_length = sequence_length 
        print(f"Wrapper initialized with sequence_length={sequence_length}")

    def load(self):
        """Loads the PyTorch model state dictionary and the scikit-learn scaler."""
        try:
            self.model = TrafficPredictorV2()
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval() 
            self.scaler = joblib.load(self.scaler_path)
            self.is_loaded = True
            print("Model and scaler loaded successfully.")
        except FileNotFoundError as e:
            print(f"Error loading artifacts: {e}. Ensure model and scaler paths are correct.")
            raise

    def predict(self, raw_input_sequence: np.ndarray, horizon: int = 1) -> tuple[float, float]:
        """
        Performs recursive multi-step prediction.
        
        Args:
            raw_input_sequence: Shape (seq_len, 2) - The raw historical data
            horizon: How many steps into the future to predict.
        """
        if not self.is_loaded:
            raise RuntimeError("Model and scaler have not been loaded.")

        # logtransform and scale
        log_scaled_input = np.log1p(raw_input_sequence) 
        scaled_sequence = self.scaler.transform(log_scaled_input)

        current_input_tensor = torch.FloatTensor(scaled_sequence).unsqueeze(0).to(self.device)

        predictions_list = []

        # recursive prediction loop for horizon
        with torch.no_grad():
            for _ in range(horizon):
                normalized_prediction = self.model(current_input_tensor)
                pred_np = normalized_prediction.cpu().numpy()[0]
                predictions_list.append(pred_np)
                next_step_input = normalized_prediction.unsqueeze(1)
                current_input_tensor = torch.cat((current_input_tensor[:, 1:, :], next_step_input), dim=1)

        predictions_array = np.array(predictions_list) 

        # choose the peak load predcitions from predictions
        max_cpu_norm = float(np.max(predictions_array[:, 0]))
        max_mem_norm = float(np.max(predictions_array[:, 1]))

        return max_cpu_norm, max_mem_norm
