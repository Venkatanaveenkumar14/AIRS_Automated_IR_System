'''def extract_features(df):
    """
    Extracts features for the ML model from the dataset.
    """
    # Avoid division by zero issues
    df['flow_duration'] = df['flow_duration'].replace(0, 1)

    df['packet_rate'] = df['total_fwd_packets'] / df['flow_duration']
    df['byte_rate'] = df['total_length_of_fwd_packets'] / df['flow_duration']

    print("Feature extraction completed.")  # Debugging output
    return df
    '''

# ======================
# 📁 data_preprocessing/feature_engineering.py (Updated)
# ======================

def extract_features(df):
    """
    Extracts only the features used in the trained model.
    Ensures consistency between training and real-time predictions.
    """
    print("\n Step 4: Extracting Features")

    # Ensure the label column is not removed
    if 'label' not in df.columns:
        print("ERROR: 'label' column is missing before feature extraction! Check dataset format.")
        return df  # Return original dataset for debugging

    # Define the exact feature list used during training
    required_features = [
        'flow_duration', 'total_fwd_packets', 'total_backward_packets',
        'total_length_of_fwd_packets', 'total_length_of_bwd_packets',
        'fwd_packet_length_max', 'fwd_packet_length_min', 'fwd_packet_length_mean',
        'fwd_packet_length_std', 'bwd_packet_length_max', 'bwd_packet_length_min',
        'bwd_packet_length_mean', 'bwd_packet_length_std', 'flow_bytes/s',
        'flow_packets/s', 'flow_iat_mean', 'flow_iat_std', 'flow_iat_max',
        'flow_iat_min', 'fwd_iat_total', 'fwd_iat_mean', 'fwd_iat_std', 'fwd_iat_max',
        'fwd_iat_min', 'bwd_iat_total', 'bwd_iat_mean', 'bwd_iat_std', 'bwd_iat_max',
        'bwd_iat_min', 'fwd_psh_flags', 'bwd_psh_flags', 'fwd_urg_flags',
        'bwd_urg_flags', 'fwd_header_length', 'bwd_header_length', 'fwd_packets/s',
        'bwd_packets/s', 'min_packet_length', 'max_packet_length',
        'packet_length_mean', 'packet_length_std', 'packet_length_variance',
        'fin_flag_count', 'syn_flag_count', 'rst_flag_count', 'psh_flag_count',
        'ack_flag_count', 'urg_flag_count', 'cwe_flag_count', 'ece_flag_count',
        'down/up_ratio', 'average_packet_size', 'avg_fwd_segment_size',
        'avg_bwd_segment_size', 'fwd_header_length.1', 'fwd_avg_bytes/bulk',
        'fwd_avg_packets/bulk', 'fwd_avg_bulk_rate', 'bwd_avg_bytes/bulk',
        'bwd_avg_packets/bulk', 'bwd_avg_bulk_rate', 'subflow_fwd_packets',
        'subflow_fwd_bytes', 'subflow_bwd_packets', 'subflow_bwd_bytes',
        'init_win_bytes_forward', 'init_win_bytes_backward', 'act_data_pkt_fwd',
        'min_seg_size_forward', 'active_mean', 'active_std', 'active_max',
        'active_min', 'idle_mean', 'idle_std', 'idle_max', 'idle_min', 'packet_rate',
        'byte_rate'
    ]

    # Ensure label is retained
    if 'label' in df.columns:
        required_features.append('label')

    # Fill missing columns with 0 to match training data
    for col in required_features:
        if col not in df.columns:
            df[col] = 0

    df = df[required_features]  # Keep only the required features
    print(f"Feature extraction completed! Shape: {df.shape}")

    return df