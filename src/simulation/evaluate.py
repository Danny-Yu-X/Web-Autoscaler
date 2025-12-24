import pandas as pd
import numpy as np
from src.simulation.model_wrapper import ModelWrapper
from src.simulation.simulator import run_streaming_autoscaler, evaluate_history, plot_autoscaler
from src.simulation.policies import ReactivePolicy, PredictivePolicy
import joblib

MODEL_PATH = "models/traffic_predictor_v5.pth"
SCALER_PATH = "models/scaler_v5.pkl"
DATA_PATH = "data/traffic_test.csv" 
SEQUENCE_LENGTH = 12 

scaling_factor = 10000 # for numbers and readibility
NODE_CAPACITY = 25.0      


INITIAL_NODES = 1   

def main():
    print("Starting simulation evaluation...")
    scaler_instance = None
    try:
        scaler_instance = joblib.load(SCALER_PATH)
        print("Global scaler loaded.")
    except FileNotFoundError:
        print(f"Error: Scaler not found at {SCALER_PATH}.")
        return

    try:
        if DATA_PATH.endswith('.csv'):

            print(f"Loading data from CSV: {DATA_PATH}")
            raw_df = pd.read_csv(DATA_PATH)
            # renamed column
            data = raw_df.rename(columns={'cpu': 'cpu_usage', 'memory': 'memory_usage'})
            
            # handles time column
            if 'start_time' in data.columns:
                data['start_time'] = pd.to_datetime(data['start_time'])
            else:
                data['start_time'] = pd.to_datetime(pd.Series(range(len(data))), unit='s')


            print(f"Applying Scaling Factor: {scaling_factor}")
            data['cpu_usage'] = data['cpu_usage'] * scaling_factor
            data['memory_usage'] = data['memory_usage'] * scaling_factor
            
            print(f"Data Loaded. First row CPU (Scaled): {data['cpu_usage'].iloc[0]:.2f}")

        elif DATA_PATH.endswith('.pkl'):
            # old logic using pickle file for backup
            loaded_data = joblib.load(DATA_PATH)
            X_test = loaded_data['X_test']
            num_samples, seq_len, num_features = X_test.shape
            X_test_reshaped = X_test.reshape(num_samples * seq_len, num_features)
            X_test_log_scale = scaler_instance.inverse_transform(X_test_reshaped)
            X_test_tiny_raw = np.expm1(X_test_log_scale)
            X_test_big_raw = X_test_tiny_raw * scaling_factor
            data = pd.DataFrame(X_test_big_raw, columns=['cpu_usage', 'memory_usage'])
            if 'start_time' not in data.columns:
                data['start_time'] = pd.to_datetime(pd.Series(range(len(data))), unit='s')

        else:
            raise ValueError("Unsupported data file type. Use .csv or .pkl")
            
    except Exception as e:
        print(f"Error loading/processing data: {e}")
        return

    # Initialize ModelWrapper
    model_wrapper_instance = None
    try:
        model_wrapper_instance = ModelWrapper(model_path=MODEL_PATH, scaler_path=SCALER_PATH, sequence_length=SEQUENCE_LENGTH)
        model_wrapper_instance.load()
    except Exception as e:
        print(f"Warning: Model wrapper issue: {e}")

    cfg = {
        "cooldown_window": 5,        
        "safety_buffer": 1.45,       
        "prediction_horizon": 6,   
        "boot_time": 3,              
        "target_util": 0.5,         
        "min_instances": 1,
        "max_instances": 40,
        "instance_capacity": NODE_CAPACITY, 
        "sequence_length": SEQUENCE_LENGTH,

    }

    # Policies
    reactive_policy = ReactivePolicy(
        node_capacity=cfg["instance_capacity"], 
        target_util=cfg["target_util"], 
        cooldown_window=cfg["cooldown_window"],
        min_instances=cfg["min_instances"]
    )

    predictive_policy = PredictivePolicy(
        node_capacity=cfg["instance_capacity"], 
        target_util=cfg["target_util"], 
        cooldown_window=cfg["cooldown_window"], 
        safety_buffer=cfg["safety_buffer"],
        min_instances=cfg["min_instances"]
    )
    if model_wrapper_instance:
        predictive_policy.model_wrapper = model_wrapper_instance

    print("\n--- Running Baseline (Reactive) ---")
    _, reactive_metrics, reactive_history = run_streaming_autoscaler(
        data, 
        policy=reactive_policy, 
        autoscaler_config=cfg,
        global_scaler=scaler_instance,
        input_scaling_factor=scaling_factor 
    )
    print("Reactive Results:", reactive_metrics)
    plot_autoscaler(reactive_history, title="Reactive Autoscaler")

    print("\n--- Running Predictive (LSTM) ---")
    if model_wrapper_instance:
        _, predictive_metrics, predictive_history = run_streaming_autoscaler(
            data,
            policy=predictive_policy,
            model_wrapper_instance=model_wrapper_instance,
            autoscaler_config=cfg,
            global_scaler=scaler_instance,
            input_scaling_factor=scaling_factor 
        )
        print("Predictive Results:", predictive_metrics)
        plot_autoscaler(predictive_history, title="Predictive Autoscaler")

if __name__ == "__main__":
    main()