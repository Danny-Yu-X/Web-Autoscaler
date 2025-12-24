import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import ast
import os

def parse_dict_string(dict_str):
    if pd.isna(dict_str) or dict_str == '':
        return None, None
    try:
        if isinstance(dict_str, str):
            dict_str = dict_str.replace("'", '"')
            dict_str = dict_str.replace('None', 'null')
            parsed = ast.literal_eval(dict_str.replace('"', "'"))
        else:
            parsed = dict_str
        cpu = parsed.get('cpus', None) if isinstance(parsed, dict) else None
        memory = parsed.get('memory', None) if isinstance(parsed, dict) else None
        return cpu, memory
    except:
        return None, None

def preprocess_csv(input_file, output_file=None, remove_zero_values=True, cap_outliers_iqr=True, apply_normalization=True):

    df = pd.read_csv(input_file)

    if 'average_usage' not in df.columns:
        raise ValueError("'average_usage' column not found in the dataset")

    cpu_values = []
    memory_values = []
    for _, row in df.iterrows():
        cpu, memory = parse_dict_string(row['average_usage'])
        cpu_values.append(cpu)
        memory_values.append(memory)

    numeric_times = pd.to_numeric(df['start_time'], errors='coerce')
    MAXINT = 2**63 - 1
    valid_mask = (numeric_times > 0) & (numeric_times < MAXINT - 1000)

    time_minutes = numeric_times / (60 * 1e6)
    min_minutes = time_minutes[valid_mask].min()
    start_times = time_minutes - min_minutes

    start_times_rounded = (
        .fillna(-1)
        .astype(int)
    )

    cluster_values = df['cluster'].values if 'cluster' in df.columns else None

    processed_df = pd.DataFrame({
        'start_time': start_times_rounded,
        'cpu': cpu_values,
        'memory': memory_values
    })

    if cluster_values is not None:
        processed_df['cluster'] = cluster_values

    processed_df = processed_df.dropna(subset=['cpu', 'memory'])

    if cluster_values is not None:
        processed_df = processed_df.sort_values(['cluster', 'start_time']).reset_index(drop=True)
    else:
        processed_df = processed_df.sort_values('start_time').reset_index(drop=True)

    processed_df['cpu'] = pd.to_numeric(processed_df['cpu'], errors='coerce')
    processed_df['memory'] = pd.to_numeric(processed_df['memory'], errors='coerce')
    processed_df = processed_df.dropna(subset=['cpu', 'memory'])

    if cluster_values is not None:
        processed_df = processed_df.groupby(
            ['cluster', 'start_time'], as_index=False
        ).agg({'cpu': 'mean', 'memory': 'mean'})
    else:
        processed_df = processed_df.groupby(
            'start_time', as_index=False
        ).agg({'cpu': 'mean', 'memory': 'mean'})

    processed_df['cpu'] = np.log1p(processed_df['cpu'])
    processed_df['memory'] = np.log1p(processed_df['memory'])

    if cap_outliers_iqr:
        if cluster_values is not None and 'cluster' in processed_df.columns:

            def cap_outliers_cluster(group):
                Q1_cpu, Q3_cpu = group['cpu'].quantile([0.25, 0.75])
                Q1_mem, Q3_mem = group['memory'].quantile([0.25, 0.75])
                IQR_cpu, IQR_mem = Q3_cpu - Q1_cpu, Q3_mem - Q1_mem

                group['cpu'] = group['cpu'].clip(
                    Q1_cpu - 1.5 * IQR_cpu,
                    Q3_cpu + 1.5 * IQR_cpu
                )
                group['memory'] = group['memory'].clip(
                    Q1_mem - 1.5 * IQR_mem,
                    Q3_mem + 1.5 * IQR_mem
                )
                return group

            processed_df = (
                processed_df
                .groupby('cluster', group_keys=False)
                .apply(cap_outliers_cluster, include_groups=False)
                .reset_index(drop=True)
            )
        else:
            Q1_cpu, Q3_cpu = processed_df['cpu'].quantile([0.25, 0.75])
            Q1_mem, Q3_mem = processed_df['memory'].quantile([0.25, 0.75])

            processed_df['cpu'] = processed_df['cpu'].clip(
                Q1_cpu - 1.5 * (Q3_cpu - Q1_cpu),
                Q3_cpu + 1.5 * (Q3_cpu - Q1_cpu)
            )
            processed_df['memory'] = processed_df['memory'].clip(
                Q1_mem - 1.5 * (Q3_mem - Q1_mem),
                Q3_mem + 1.5 * (Q3_mem - Q1_mem)
            )

    if apply_normalization:
        if cluster_values is not None and 'cluster' in processed_df.columns:

            def normalize_cluster(group):
                scaler = MinMaxScaler()
                group[['cpu', 'memory']] = scaler.fit_transform(group[['cpu', 'memory']])
                return group

            processed_df = (
                processed_df
                .groupby('cluster', group_keys=False)
                .apply(normalize_cluster, include_groups=False)
                .reset_index(drop=True)
            )
        else:
            scaler = MinMaxScaler()
            processed_df[['cpu', 'memory']] = scaler.fit_transform(
                processed_df[['cpu', 'memory']]
            )

    if remove_zero_values:
        processed_df = processed_df[
            ~((processed_df['cpu'] == 0) | (processed_df['memory'] == 0))
        ]

    return processed_df


if __name__ == "__main__":
    input_file = '/Users/darrenchoe/dev/170project/borg_traces_data.csv'
    processed_df = preprocess_csv(input_file, apply_normalization=True)

    if 'cluster' in processed_df.columns:
        base_dir = os.path.dirname(input_file) or '.'
        base_name = 'preprocessed_data'

        for cluster_id in sorted(processed_df['cluster'].dropna().unique()):
            cluster_df = processed_df[processed_df['cluster'] == cluster_id]
            cluster_file = os.path.join(base_dir, f"{base_name}_cluster_{int(cluster_id)}.csv")
            cluster_df.to_csv(cluster_file, index=False)
    print('Done!')
