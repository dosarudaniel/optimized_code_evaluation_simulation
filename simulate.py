# simulate.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import List, Tuple
from datetime import datetime

# ---------------------------
# Helper: load data
# ---------------------------
def load_submissions(path: str = None) -> pd.DataFrame:
    if path:
        df = pd.read_csv(path)
        # ensure columns exist
        assert {'timestamp','eval_time'}.issubset(df.columns), \
            "CSV must contain 'timestamp' and 'eval_time' columns"
        df = df.sort_values('timestamp').reset_index(drop=True)
        return df[['timestamp','eval_time','student_id']] if 'student_id' in df.columns else df[['timestamp','eval_time']]
    # synthetic demo data
    np.random.seed(42)
    n = 500
    timestamps = np.sort(np.random.uniform(0, 3600, size=n))   # times in seconds within one hour
    eval_times = np.random.exponential(scale=30, size=n) + 2   # evaluation secs
    student_ids = np.random.randint(1, 200, size=n)
    df = pd.DataFrame({'timestamp': timestamps, 'eval_time': eval_times, 'student_id': student_ids})
    return df

# ---------------------------
# Simulation core
# ---------------------------
def simulate_nodes(df: pd.DataFrame, num_nodes: int) -> Tuple[List[float], List[float]]:
    """
    Simulate processing with `num_nodes` identical processors.
    For each submission (sorted by arrival timestamp), we assign it to the node which becomes
    available the earliest (min node_available time). We track wait_time = start - arrival.
    Returns (wait_times, finish_times)
    """
    # track availability time for each node (initially zero)
    node_available = [0.0] * num_nodes
    wait_times = []
    finish_times = []

    # ensure sorted
    df_sorted = df.sort_values('timestamp').reset_index(drop=True)

    for _, row in df_sorted.iterrows():
        
        arrival_dt = datetime.strptime(row['timestamp'], '%d/%m/%Y %H:%M:%S')
        arrival = arrival_dt.timestamp()  # float in seconds since epoch
        #timestamp_str = row['timestamp']
        #arrival = datetime.strptime(timestamp_str, '%d/%m/%Y %H:%M:%S')
        #arrival = float(row['timestamp'])
        service = float(row['eval_time'])

        # find node that will be free earliest
        n_idx = min(range(num_nodes), key=lambda i: node_available[i])
        start_time = max(arrival, node_available[n_idx])
        wait = start_time - arrival
        finish = start_time + service

        wait_times.append(wait)
        finish_times.append(finish)

        # update that node availability
        node_available[n_idx] = finish

    return wait_times, finish_times


# ---------------------------
# Main: run experiments and plot
# ---------------------------
def main():
    
    colors = {
        1: "C0",
        2: "C1",
        4: "C2",
        8: "C3",   # always red-ish
        12: "C4",  # always purple-ish
    }
    # change this path if you want to load your CSV
    path = "dataset.csv"   # set to None to use synthetic data
    try:
        df = load_submissions(path if __name__ == "__main__" else None)
    except Exception as e:
        print("Couldn't load CSV:", e)
        df = load_submissions(None)

    # experiment node counts
    node_list = [1, 2, 4, 8, 12] 
    results = {}

    for n in node_list:
        waits, finishes = simulate_nodes(df, n)
        results[n] = {
            'waits': np.array(waits),
            'finishes': np.array(finishes),
            'drain_time': float(np.max(finishes))  # last finish time
        }

    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Create time bins (e.g., 1-minute intervals)
    time_bin_size = '1T'  # 'T' = minute, 'S' = second, 'H' = hour
    freq_series = df.set_index('timestamp').resample(time_bin_size).size()

    # Plot submission frequency
    fig, ax1 = plt.subplots(figsize=(12,5))

    ax1.bar(freq_series.index, freq_series.values, width=0.0008, color='skyblue', alpha=0.6)
    ax1.set_xlabel("Time", fontsize=14)
    ax1.set_ylabel("Submissions/minute", fontsize=14, color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    # If you want to overlay wait times for 8 and 12 nodes on the same x-axis:
    ax2 = ax1.twinx()  # second y-axis
    for n in [12]:    # Change here to [1,2,4,8,12] to get Figure 3.
        if n not in results: continue
        # Convert submission indices to timestamps
        times = df['timestamp'].iloc[:len(results[n]['waits'])]
        ax2.plot(times, results[n]['waits'], label=f"{n} nodes", linewidth=2, color=colors[n])

    ax2.set_ylabel("Wait time (seconds)", fontsize=14, color='purple')
    ax2.tick_params(axis='y', labelcolor='purple')
    ax2.legend(fontsize=12)

    # plt.title("Submission frequency and wait times", fontsize=16)
    plt.grid(alpha=0.3)
    time_format = mdates.DateFormatter('%H:%M')
    ax1.xaxis.set_major_formatter(time_format)
    ax2.xaxis.set_major_formatter(time_format)

    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    # Histogram of wait times (stacked)
    plt.figure(figsize=(10,5))
    plt.hist([results[q]['waits'] for q in [1,2,4,8,12]], bins=40, label=[f"{q} nodes" for q in [1,2,4,8,12]], stacked=False)
    plt.xlabel("Wait time (s)")
    plt.ylabel("Count")
    plt.title("Wait time distribution")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Drain time vs node count
    plt.figure(figsize=(8,5))
    xs = sorted(results.keys())
    ys = [results[x]['drain_time'] for x in xs]
    plt.plot(xs, ys, marker='o')
    plt.xlabel("Number of nodes")
    plt.ylabel("Time when last submission finished (s)")
    plt.title("System drain time vs number of nodes")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Print summary stats
    print("Summary (per node configuration):")
    for n in xs:
        w = results[n]['waits']
        print(f"Nodes={n}: drain_time={results[n]['drain_time']:.1f}s, mean_wait={w.mean():.2f}s, p95_wait={np.percentile(w,95):.1f}s")

if __name__ == "__main__":
    main()
