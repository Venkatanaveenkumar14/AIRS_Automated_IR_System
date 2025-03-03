import pandas as pd
import os

def load_splunk_logs(directory):
    """
    Loads multiple Splunk log files from the specified directory.
    Returns a combined DataFrame of all logs.
    """
    all_logs = []
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            filepath = os.path.join(directory, filename)
            df = pd.read_csv(filepath, low_memory=False)
            all_logs.append(df)
    return pd.concat(all_logs, ignore_index=True) if all_logs else pd.DataFrame()
