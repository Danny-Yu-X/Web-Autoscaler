import numpy as np
import matplotlib.pyplot as plt

# Latency / QoS proxy
def latency_from_utilization(per_instance_util, base_latency_ms=100.0, clamp_threshold=0.95, eps=1e-6):

    u = float(per_instance_util)
    if u < 0:
        u = 0.0
    if u >= clamp_threshold:
        # penalty factor (tunable)
        penalty = 5.0
        return base_latency_ms / max(eps, (1 - clamp_threshold)) * penalty
    return base_latency_ms / max(eps, (1.0 - u))

# Reactive autoscaler
# Scales up when average CPU exceeds threshold
# Scales down when average CPU falls below threshold
class ReactiveAutoscaler:
    def __init__(self, scale_up_threshold=0.7, scale_down_threshold=0.3,scale_step=1, min_instances=1,max_instances=10, instance_capacity=1.0):
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.scale_step = scale_step
        self.min_instances = min_instances
        self.max_instances = max_instances
        self.instance_capacity = instance_capacity
        self.instances = self.min_instances
        self.history = []

    def capacity(self):
        return self.instances * self.instance_capacity

# Updates autoscaler state based on curr CPU and mem use
    def update(self, cpu_usage, mem_usage=None):
        if mem_usage is None:
            demand = float(cpu_usage)
        else:
            demand = float((cpu_usage + mem_usage) / 2.0)
        per_instance_util = demand / max(1e-6, self.capacity())

        if per_instance_util > self.scale_up_threshold and self.instances < self.max_instances:
            self.instances = min(self.max_instances, self.instances + self.scale_step)
        elif per_instance_util < self.scale_down_threshold and self.instances > self.min_instances:
            self.instances = max(self.min_instances, self.instances - self.scale_step)

        # QoS / provisioning metrics
        latency_ms = latency_from_utilization(per_instance_util)
        underprovisioning_flag = per_instance_util >= 1.0

        record = {
            'demand': demand,
            'cpu': float(cpu_usage),
            'mem': (None if mem_usage is None else float(mem_usage)),
            'instances': int(self.instances),
            'per_instance_util': float(per_instance_util),
            'latency_ms': float(latency_ms),
            'underprovisioning_flag': bool(underprovisioning_flag)
        }
        self.history.append(record)
        return self.instances

# runs autoscaler over a sequence of predicted usage vals
    def simulate(self, cpu_series, mem_series=None):
        self.history = []
        self.instances = self.min_instances
        n = len(cpu_series)
        if mem_series is None:
            for t in range(n):
                self.update(cpu_series[t], None)
        else:
            for t in range(n):
                self.update(cpu_series[t], mem_series[t])
        return self.history

# metric tracking
def evaluate_history(history, overprovision_margin=0.3):
    if not history:
        return {}

    demands = np.array([h['demand'] for h in history], dtype=float)
    instances = np.array([h['instances'] for h in history], dtype=float)
    latencies = np.array([h['latency_ms'] for h in history], dtype=float)
    underprov_flags = np.array([1 if h.get('underprovisioning_flag', False) else 0 for h in history], dtype=int)

    capacity = instances

    # original metrics
    avg_utilization = float(np.mean(demands / np.maximum(capacity, 1e-6)))
    scaling_actions = int(np.sum(np.abs(np.diff(instances))))
    total_cost = float(np.sum(instances))
    # new metrics: QoS + provisioning
    avg_latency_ms = float(np.mean(latencies))
    p95_latency_ms = float(np.percentile(latencies, 95))
    underprovision_count = int(np.sum(underprov_flags))
    underprovision_rate = float(np.mean(underprov_flags))
    overprov_mask = (capacity - demands) / np.maximum(capacity, 1e-6) > overprovision_margin
    overprovision_count = int(np.sum(overprov_mask))
    overprovision_rate = float(np.mean(overprov_mask))
    avg_instances = float(np.mean(instances))

    metrics = {
        # average demand/capacity across timesteps
        'avg_utilization': round(avg_utilization, 4),

        # count of instance changes (stability indicator)
        'scaling_actions': scaling_actions,

        # total instance-time usage
        'total_cost': round(total_cost, 4),

        # average response latency
        'avg_latency_ms': round(avg_latency_ms, 3),

        # 95th percentile latency (QoS)
        'p95_latency_ms': round(p95_latency_ms, 3),

        # how often demand exceeded capacity
        'underprovision_count': underprovision_count,

        # fraction of time underprovisioned
        'underprovision_rate': round(underprovision_rate, 4),

        # how often capacity far exceeded demand (waste)
        'overprovision_count': overprovision_count,

        # fraction of time over-provisioned
        'overprovision_rate': round(overprovision_rate, 4),

        # mean nuumber of instances
        'avg_instances': round(avg_instances, 3)
    }
    return metrics

# predicted_cpu / predicted_mem: both are arrays
# config: optional dict to override thresholds and instance limits
# use_predictions: If True, currently still runs reactive but documents intent

def run_autoscaler_simulation(predicted_cpu, predicted_mem=None, config=None, use_predictions=False):
    cfg = config or {}
    autoscaler = ReactiveAutoscaler(scale_up_threshold=cfg.get('scale_up_threshold', 0.7), scale_down_threshold=cfg.get('scale_down_threshold', 0.3), scale_step=cfg.get('scale_step', 1), min_instances=cfg.get('min_instances', 1), max_instances=cfg.get('max_instances', 10), instance_capacity=cfg.get('instance_capacity', 1.0))

    # treat predicted_cpu as observed_cpu for reactive
    observed_cpu = np.array(predicted_cpu, dtype=float)
    observed_mem = None if predicted_mem is None else np.array(predicted_mem, dtype=float)

    history = autoscaler.simulate(observed_cpu, observed_mem)
    metrics = evaluate_history(history, overprovision_margin=cfg.get('overprovision_margin', 0.3))

    return metrics, history

def plot(history, title="Reactive Autoscaler Simulation"):
    cpu_usages = [h['cpu'] for h in history]
    instances = [h['instances'] for h in history]
    steps = range(len(cpu_usages))

    plt.figure(figsize=(10, 5))
    plt.plot(steps, cpu_usages, label='CPU usage (observed)')
    plt.step(steps, instances, label='Instances', where='post')
    plt.xlabel('Time step')
    plt.ylabel('CPU / Instances')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

# When modeling team finishes up, we need to incorporate their observed cpu + mem usage for this reactive autoscaler
# We also need their predicted cpu + prediced mem usuage for the prediced autoscaler so we can compare later

# Example test case for our reactive autoscaler for testing purposes (we will remove this test and incorporate actual data later):
if __name__ == "__main__":
    np.random.seed(1)
    observed_cpu = np.clip(np.random.normal(0.5, 0.2, 100), 0, 1)
    observed_mem = np.clip(np.random.normal(0.45, 0.25, 100), 0, 1)

    cfg = {
        'scale_up_threshold': 0.7,
        'scale_down_threshold': 0.3,
        'scale_step': 1,
        'min_instances': 1,
        'max_instances': 10,
        'instance_capacity': 1.0,
        'overprovision_margin': 0.3
    }

    metrics, history = run_autoscaler_simulation(observed_cpu, observed_mem, config=cfg)
    print("Simulation Metrics:")
    for k, v in metrics.items():
        print(f"  {k:>20}: {v}")

    plot(history, title="Reactive Autoscaler")
