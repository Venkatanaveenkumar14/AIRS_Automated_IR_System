# Automated Incident Response System

## Overview
The **Automated Incident Response System (AIRS)** is an AI/ML-driven security solution designed to detect, classify, and mitigate real-time cyber threats. The system integrates **Splunk logs**, **SIEM tools**, and **firewall automation** to enhance network security by automating incident response workflows.

## Datasets Used
This project leverages the following datasets:
1. **CICIDDoS2019** - A benchmark dataset for **DDoS attack detection**, provided by the Canadian Institute for Cybersecurity (CIC), which includes modern attack scenarios.
2. **CICIDDoS2017** - A widely used dataset containing **various DDoS attack types**, generated in a real-time network environment.

## Features
- **Real-Time Threat Detection**: Uses AI/ML models to classify security logs and identify malicious patterns.
- **Automated Incident Response**: Executes predefined security actions like **blocking IPs, terminating malicious processes, and alerting security teams**.
- **SIEM Integration**: Works with **Splunk and Elastic Stack** for log analysis and visualization.
- **Firewall Automation**: Interacts with **IPTables, Palo Alto APIs**, and other firewall solutions for automated mitigation.
- **Security Dashboard**: A **Flask/Streamlit-based dashboard** to visualize security events and ML predictions.

## Folder Structure
```
Automated_Incident_Response/
├── data_preprocessing/       # Handles log ingestion & feature extraction
│   ├── splunk_log_parser.py    # Parses Splunk logs
│   ├── nsl_kdd_preprocessor.py # Future work: Processes NSL-KDD dataset
│   ├── ddos_preprocessor.py   #Process DDoS dataset from CICIDDoS'17'19
│   ├── feature_engineering.py  # Extracts & normalizes security event data
│
├── ml_model/                # Machine Learning for threat classification
│   ├── train_model.py         # Trains ML model
│   ├── model_predictor.py     # Classifies real-time logs
│   ├── anomaly_detection.py   # Detects unknown attack patterns
│
├── models/                   # Trained Models
│   ├── trainedmodel1.joblib
│   ├── trainedmodel2.joblib
│
├── incident_response/       # Automates security actions & alerting
│   ├── response_engine.py     # Executes mitigation actions
│   ├── firewall_automation.py # Interacts with firewall APIs
│   ├── alerting_system.py     # Sends alerts via Slack/Teams/Email
│
├── dashboard/               # Visualizing Security Events
│   ├── app.py                 # Streamlit-based dashboard
│   ├── logs_visualizer.py     # Plots real-time analytics
│
└── main.py                  # Main execution file integrating all components
```

## Installation & Setup
### Prerequisites
- **Python 3.9+**
- **Splunk (for log analysis)**
- **Elastic Stack (optional)**
- **Palo Alto API (for firewall automation)**
- **Flask or Streamlit (for dashboard visualization)**

### Installation Steps
1. Clone the repository:
   ```sh
   git clone https://github.com/Venkatanaveenkumar14/AIRS_Automated_IR_System.git
   cd Automated_Incident_Response
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Run the incident response system:
   ```sh
   python main.py
   ```
4. DDoS dataset preprocessing:
   ```sh
   python data_preprocessing/ddos_preprocessor.py
   ```
5. Train the ML model:
   ```sh
   python ml_model/train_model.py
   ```
6. Evaluate the trained model:
   ```sh
   python -m ml_model.evaluate_model
   ```
7. View the Real-Time Dashboard:
   ```sh
   streamlit run dashboard/app.py
   ```

## Usage
1. **Monitor security logs in real-time** through Splunk.
2. **Detect and classify attacks** using the trained ML model.
3. **Automate mitigation actions** (IP blocking, process termination, etc.).
4. **Visualize security insights** via the Flask/Streamlit dashboard.

## Future Enhancements
- Integrate **more datasets (e.g., CICIDS2018, UNSW-NB15)** for broader attack coverage.
- Improve **anomaly detection** with reinforcement learning.
- Deploy on **AWS/GCP/Azure** for cloud-based security automation.
- Extend **SOAR (Security Orchestration, Automation, and Response)** capabilities.

## License
This project is open-source and available under the **MIT License**.

--- 
🔗 **GitHub:** [Venkatanaveenkumar14](https://github.com/Venkatanaveenkumar14)

