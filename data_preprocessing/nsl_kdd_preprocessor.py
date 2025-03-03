import pandas as pd
import numpy as np
import gzip
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

def load_kdd_dataset(dataset_path):
    """
    Loads the KDD Cup 1999 dataset from a compressed .gz file and preprocesses it.
    """
    # Define column names based on dataset description
    columns = [
        'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
        'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins',
        'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root',
        'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds',
        'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate',
        'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
        'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
        'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
        'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
        'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
        'dst_host_srv_rerror_rate', 'target'
    ]

    try:
        # Open and read the compressed file
        with gzip.open(dataset_path, 'rt', encoding='ISO-8859-1') as file:
            df = pd.read_csv(file, names=columns)

        # Mapping attack types to categories
        attack_types = {
            'normal': 'normal', 'back': 'dos', 'buffer_overflow': 'u2r', 'ftp_write': 'r2l',
            'guess_passwd': 'r2l', 'imap': 'r2l', 'ipsweep': 'probe',
            'land': 'dos', 'loadmodule': 'u2r', 'multihop': 'r2l', 'neptune': 'dos',
            'nmap': 'probe', 'perl': 'u2r', 'phf': 'r2l', 'pod': 'dos',
            'portsweep': 'probe', 'rootkit': 'u2r', 'satan': 'probe',
            'smurf': 'dos', 'spy': 'r2l', 'teardrop': 'dos', 'warezclient': 'r2l',
            'warezmaster': 'r2l'
        }
        df['Attack Type'] = df.target.apply(lambda r: attack_types[r.strip('.')])
        df.drop(['target'], axis=1, inplace=True)  # Remove original target column
        return df

    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def preprocess_kdd(df):
    """
    Preprocesses the KDD dataset: encodes categorical features, removes correlated features, and scales numerical features.
    """
    if df is None:
        print("Error: Dataframe is empty. Check dataset path and format.")
        return None, None, None, None
    
    # Encode categorical features
    categorical_features = ['protocol_type', 'flag']
    label_encoders = {}
    for col in categorical_features:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    
    # Drop highly correlated features
    df.drop(['num_root', 'srv_serror_rate', 'srv_rerror_rate', 'dst_host_srv_serror_rate',
             'dst_host_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate',
             'dst_host_same_srv_rate', 'service'], axis=1, inplace=True)
    
    # Scale numerical features
    scaler = MinMaxScaler()
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    # Separate features and labels
    X = df.drop(columns=['Attack Type'])
    y = df['Attack Type']
    
    return X, y, label_encoders, scaler