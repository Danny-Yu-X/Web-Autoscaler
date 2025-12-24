import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.simulation.policies import ReactivePolicy, PredictivePolicy, AutoscalerPolicy
from src.simulation.model_wrapper import ModelWrapper
from sklearn.preprocessing import MinMaxScaler 

STEP_SIZE = 10
PREVIEW_STEPS = 5000

def step_through_time(df, step_size=STEP_SIZE):
    for i in range(0, len(df), step_size):
        yield df.iloc[i:i + step_size]

def update_booting_instances(booting_instances):
    finished = 0
    for inst in booting_instances:
        inst["remaining"] -= 1
    finished = sum(1 for inst in booting_instances if inst["remaining"] <= 0)
    booting_instances[:] = [inst for inst in booting_instances if inst["remaining"] > 0]
    return finished

def run_streaming_autoscaler(
    df: pd.DataFrame,
    policy: AutoscalerPolicy, 
    model_wrapper_instance: ModelWrapper = None,
    autoscaler_config: dict = None,
    step_size: int = STEP_SIZE,
    preview_steps: int = PREVIEW_STEPS,
    global_scaler: MinMaxScaler = None,
    input_scaling_factor: float = 1.0  
):
    cfg = autoscaler_config or {}
    print("\n=== Starting Streaming Autoscaler Simulation ===\n")
    
    BOOT_TIME = cfg.get("boot_time", 5)
    INITIAL_NODES = cfg.get("min_instances", 1)

    active_instances = INITIAL_NODES
    booting_instances = []

    policy.last_scale_change_step = -1
    policy.last_action_type = "NONE"

    history = []

    for chunk_idx, chunk in enumerate(step_through_time(df, step_size)):
        for row_idx, row in chunk.iterrows():
            current_step = row_idx 
            
            current_cpu_load_raw = float(row["cpu_usage"])
            current_mem_load_raw = float(row["memory_usage"])

            newly_ready = update_booting_instances(booting_instances)
            active_instances += newly_ready

            predicted_cpu_load_raw = 0.0 
            
            if isinstance(policy, PredictivePolicy) and model_wrapper_instance:
                sequence_length = cfg.get("sequence_length", 6)
                
                prediction_horizon = cfg.get("prediction_horizon", 5) 

                if current_step >= sequence_length:
                    # get historical data
                    historical_data_big = df.loc[current_step - sequence_length:current_step - 1, ["cpu_usage", "memory_usage"]].values
                    historical_data_for_model = historical_data_big / input_scaling_factor

                    try:
                        # predict with horizon
                        pred_norm_cpu, pred_norm_mem = model_wrapper_instance.predict(
                            historical_data_for_model, 
                            horizon=prediction_horizon
                        )
                        
                        # inverse transform and scale up
                        if global_scaler is not None:
                            pred_norm_combined = np.array([[pred_norm_cpu, pred_norm_mem]])
                            pred_log_scale = global_scaler.inverse_transform(pred_norm_combined)
                            predicted_tiny_raw = np.expm1(pred_log_scale[0, 0])
                            predicted_cpu_load_raw = predicted_tiny_raw * input_scaling_factor
                        else:
                            predicted_cpu_load_raw = pred_norm_cpu * input_scaling_factor 

                    except Exception as e:
                        # print(f"Prediction error: {e}") 
                        predicted_cpu_load_raw = current_cpu_load_raw 
                else:
                    predicted_cpu_load_raw = current_cpu_load_raw

            # policy logic
            desired_instances_from_policy = policy.get_decision(
                current_step=current_step,
                current_load=current_cpu_load_raw, 
                predicted_load=predicted_cpu_load_raw, 
                current_nodes=active_instances 
            )
            
            total_available_including_booting = active_instances + len(booting_instances)

            if desired_instances_from_policy > total_available_including_booting:
                to_add = desired_instances_from_policy - total_available_including_booting
                for _ in range(to_add):
                    booting_instances.append({"remaining": BOOT_TIME})
            elif desired_instances_from_policy < active_instances:
                to_remove = active_instances - desired_instances_from_policy
                active_instances -= min(to_remove, active_instances - policy.min_instances)

            actual_capacity = active_instances * policy.node_capacity
            utilization = current_cpu_load_raw / max(1e-6, actual_capacity) 

            history.append({
                'step': current_step,
                'current_cpu_load': current_cpu_load_raw, 
                'predicted_cpu_load': predicted_cpu_load_raw, 
                'active_nodes': active_instances,
                'utilization': utilization
            })
            if current_step % 50 == 0:
                print(f"Step {current_step}: Load={current_cpu_load_raw:.2f}, Active={active_instances}")

        if chunk_idx >= preview_steps: 
            break

    history_df = pd.DataFrame(history)
    history_df['node_capacity'] = policy.node_capacity
    metrics = evaluate_history(history_df)

    return policy, metrics, history_df

def evaluate_history(history_df, overprovision_margin=0.3):
    if history_df.empty:
        return {}

    demands = history_df['current_cpu_load'].values
    instances = history_df['active_nodes'].values
    node_capacity = history_df['node_capacity'].iloc[0]
    capacity = instances * node_capacity

    utilization_ratio = demands / np.maximum(capacity, 1e-6)
    
    # simulated latency curve: base 20ms + expoenntial growth as we near 100%
    latencies = 20 + (1000 * np.maximum(0, utilization_ratio - 0.8)**2)
    # cap latency at 5000ms (timeout) for crashed steps
    latencies[utilization_ratio > 1.0] = 5000 

    unserved_load = np.sum(np.maximum(0, demands - capacity))
    p99_latency = np.percentile(latencies, 99)
    util_std_dev = np.std(utilization_ratio)
    avg_utilization = float(np.mean(utilization_ratio))
    scaling_actions = int(np.sum(np.abs(np.diff(instances))))
    total_cost = float(np.sum(instances))
    underprovision_count = int(np.sum(demands > capacity))

    metrics = {
        'avg_utilization': round(avg_utilization, 4),
        'scaling_actions': scaling_actions,
        'total_cost': round(total_cost, 2),
        'avg_instances': round(float(np.mean(instances)), 3),
        'total_unserved_load': round(unserved_load, 2), 
        'p99_latency_ms': round(p99_latency, 2),        
        'stability_std_dev': round(util_std_dev, 4),    
        'crash_steps': underprovision_count             
    }
    return metrics

def plot_autoscaler(history_df, title="Simulation"):
    cpu_usages = history_df['current_cpu_load'].values
    instances = history_df['active_nodes'].values
    steps = range(len(cpu_usages))

    fig, ax1 = plt.subplots(figsize=(12, 6))

    color_traffic = 'tab:blue'
    ax1.set_xlabel('Simulation Step', fontsize=12)
    ax1.set_ylabel('Traffic Load (Requests)', color=color_traffic, fontsize=12)
    ax1.fill_between(steps, cpu_usages, alpha=0.3, color=color_traffic, label="Traffic Load")
    ax1.plot(steps, cpu_usages, color=color_traffic, linewidth=1.5)
    ax1.tick_params(axis='y', labelcolor=color_traffic)
    ax1.grid(True, which='both', linestyle='--', alpha=0.6)

    ax2 = ax1.twinx()
    color_instances = 'tab:orange'
    ax2.set_ylabel('Server Instances', color=color_instances, fontsize=12)
    ax2.step(steps, instances, where="post", color=color_instances, linewidth=2.5, label="Active Servers")
    ax2.tick_params(axis='y', labelcolor=color_instances)

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper left', fontsize=10)

    plt.title(title, fontsize=16)
    fig.tight_layout()
    plt.show()