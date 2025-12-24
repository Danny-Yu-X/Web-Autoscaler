import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_traffic_data(
    start_date='2019-04-30 23:55:00',
    num_days=30,
    interval_minutes=5,
    output_path='data/traffic_data.csv'
):
    """
    Generate dummy traffic data for class registration portal simulation.

    patterns:
    - higher traffic 8am-8pm
    - peak 11am-2pm
    - lower at extreme times
    - random spikes to simulate unexpected surges

    params :

    start_date : str
        Starting datetime for data generation
    num_days : int
        Number of days to generate data for
    interval_minutes : int
        Time interval between measurements (should be always 5 minutes)
    output_path : str
        Path to save the CSV file
    """

    points_per_day = (24 * 60) // interval_minutes
    total_points = points_per_day * num_days

    start = pd.to_datetime(start_date)
    timestamps = [start + timedelta(minutes=i*interval_minutes) for i in range(total_points)]

    cpu_usage = np.zeros(total_points)
    memory_usage = np.zeros(total_points)

    np.random.seed(42)

    for i, ts in enumerate(timestamps):
        hour = ts.hour
        day_of_week = ts.weekday()

        base_cpu = 0.003
        base_memory = 0.0006

        # Time-based patterns
        if 0 <= hour < 6:  # night - very low traffic
            traffic_multiplier = np.random.uniform(0.5, 2.0)
        elif 6 <= hour < 8:  # morning increasing
            traffic_multiplier = np.random.uniform(2.0, 8.0)
        elif 8 <= hour < 11:  # morning to noon - moderate
            traffic_multiplier = np.random.uniform(8.0, 15.0)
        elif 11 <= hour < 14:  # peak - high traffic
            traffic_multiplier = np.random.uniform(15.0, 35.0)
        elif 14 <= hour < 17: 
            traffic_multiplier = np.random.uniform(10.0, 20.0)
        elif 17 <= hour < 20:
            traffic_multiplier = np.random.uniform(8.0, 15.0)
        elif 20 <= hour < 22:
            traffic_multiplier = np.random.uniform(4.0, 10.0)
        else:
            traffic_multiplier = np.random.uniform(1.0, 5.0)

        if day_of_week >= 5:
            traffic_multiplier *= 0.3

        if np.random.random() < 0.01:
            traffic_multiplier *= np.random.uniform(2.0, 4.0)

        cpu = base_cpu * traffic_multiplier
        cpu += np.random.normal(0, cpu * 0.1)
        cpu = np.clip(cpu, 0.0, 1.0)

        memory = base_memory * traffic_multiplier * np.random.uniform(0.8, 1.2)
        memory += np.random.normal(0, memory * 0.08)
        memory = np.clip(memory, 0.0, 1.0)

        cpu_usage[i] = cpu
        memory_usage[i] = memory

    df = pd.DataFrame({
        'start_time': timestamps,
        'cpu': cpu_usage,
        'memory': memory_usage
    })

    # save
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} data points")
    print(f"Date range: {df['start_time'].min()} to {df['start_time'].max()}")
    print(f"\nData saved to: {output_path}")
    print(f"\nFirst 10 rows:")
    print(df.head(10))
    print(f"\nBasic statistics:")
    print(df[['cpu', 'memory']].describe())

    return df

if __name__ == "__main__":
    # 7 days of data with 5-minute intervals
    df = generate_traffic_data(
        start_date='2019-04-30 23:55:00',
        num_days=7,
        interval_minutes=5,
        output_path='data/traffic_data.csv'
    )