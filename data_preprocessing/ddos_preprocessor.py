import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
'''
def preprocess_ddos_dataset(df):
    """
    Preprocesses the dataset to handle missing values, encode labels, and standardize column names.
    """
    # Normalize column names: Strip spaces, replace spaces with underscores, and convert to lowercase
    df.columns = df.columns.str.strip().str.replace(" ", "_").str.lower()

    # Convert all numeric columns from string to float
    for col in df.columns:
        if col != "label":  # Exclude label from conversion
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Handle missing and infinite values
    df.fillna(0, inplace=True)
    df.replace([float('inf'), float('-inf')], 0, inplace=True)

    # Encode categorical labels
    label_encoder = LabelEncoder()
    if 'label' in df.columns:
        df['label'] = label_encoder.fit_transform(df['label'].astype(str))  # Convert labels to strings before encoding

    # Standardize numerical features excluding the label
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    numerical_cols = [col for col in numerical_cols if col != 'label']  # Exclude 'label' from standardization

    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    return df, label_encoder
    '''

# ======================
# 📁 data_preprocessing/ddos_preprocessor.py (Updated)
# ======================

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess_ddos_dataset(df):
    """
    Preprocesses the dataset to handle missing values, encode labels, and standardize column names.
    """
    print("\n📌 Step 1: Standardizing Column Names")
    df.columns = df.columns.str.strip().str.replace(" ", "_").str.lower()
    print("✅ Standardized Columns:", df.columns.tolist())

    # Check if 'label' column exists
    if 'label' not in df.columns:
        print("❌ ERROR: 'label' column is missing before preprocessing! Check dataset format.")
        return None, None

    print("\n📌 Step 2: Converting Numeric Columns")
    for col in df.columns:
        if col != "label":  # Exclude label from conversion
            df[col] = pd.to_numeric(df[col], errors='coerce')

    print("\n📌 Step 3: Handling Missing & Infinite Values")
    df.fillna(0, inplace=True)
    df.replace([float('inf'), float('-inf')], 0, inplace=True)

    print("\n📌 Step 4: Encoding Labels")
    label_encoder = LabelEncoder()
    df['label'] = label_encoder.fit_transform(df['label'].astype(str))  # Convert labels to strings before encoding

    print("\n📌 Step 5: Standardizing Numerical Features")
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    numerical_cols = [col for col in numerical_cols if col != 'label']  # Exclude 'label' from standardization

    if not numerical_cols:
        print("❌ ERROR: No numerical columns found for scaling! Check dataset format.")
        return df, label_encoder

    print(f"✅ Found {len(numerical_cols)} numerical features for scaling.")
    
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    print("\n✅ Preprocessing completed successfully!")
    print("📌 Available columns after preprocessing:", df.columns.tolist())

    return df, label_encoder