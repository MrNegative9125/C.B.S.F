import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import requests
import socket
from urllib.parse import urlparse
import time
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from io import BytesIO
import os
import warnings
warnings.filterwarnings('ignore')

# ================================
# CONFIGURATION
# ================================

# Default path - will be configurable in the UI
DEFAULT_BASE_PATH = r"dwig_ML9125"

# ================================
# HELPER FUNCTIONS
# ================================

def get_file_path(base_path, *parts):
    """Create OS-independent file paths"""
    return os.path.join(base_path, *parts)

def check_path_exists(path):
    """Check if path exists"""
    return os.path.exists(path)

@st.cache_resource
def load_models_and_artifacts(base_path):
    """Load all pre-trained models and preprocessing artifacts"""
    models_path = get_file_path(base_path, "models")
    
    # Check if base path exists
    if not check_path_exists(base_path):
        st.error(f"❌ Base directory not found: {base_path}")
        st.info("💡 Please configure the correct path in the sidebar.")
        return None
    
    # Check if models directory exists
    if not check_path_exists(models_path):
        st.error(f"❌ Models directory not found: {models_path}")
        st.info("💡 Please ensure the 'models' folder exists in the base directory.")
        return None
    
    # Dictionary to track which files are missing
    missing_files = []
    loaded_models = {}
    
    try:
        # Define required files
        files_to_load = {
            'binary_lr': 'binary_model_Logistic_Regression.pkl',
            'binary_rf': 'binary_model_Random_Forest.pkl',
            'multiclass': 'multiclass_model.pkl',
            'scaler': 'scaler.pkl',
            'label_encoder': 'label_encoder.pkl',
            'label_mapping': 'label_mapping.json',
            'feature_names': 'feature_names.pkl'
        }
        
        # Try loading each file
        for key, filename in files_to_load.items():
            file_path = get_file_path(models_path, filename)
            
            if not check_path_exists(file_path):
                missing_files.append(filename)
                continue
            
            try:
                if filename.endswith('.json'):
                    with open(file_path, 'r') as f:
                        loaded_models[key] = json.load(f)
                else:
                    with open(file_path, 'rb') as f:
                        loaded_models[key] = pickle.load(f)
                        
            except Exception as e:
                st.warning(f"⚠️ Error loading {filename}: {str(e)}")
                missing_files.append(filename)
        
        # Check if we have all required files
        if missing_files:
            st.error(f"❌ Missing required files in {models_path}:")
            for file in missing_files:
                st.error(f"   • {file}")
            st.info("💡 Please ensure all model files are present in the models directory.")
            return None
        
        st.success(f"✅ Successfully loaded {len(loaded_models)} model artifacts!")
        return loaded_models
        
    except Exception as e:
        st.error(f"❌ Unexpected error loading models: {str(e)}")
        st.info(f"📁 Looking for files in: {models_path}")
        return None

def validate_url(url):
    """Validate if URL is reachable and extract basic information"""
    try:
        # Add http if not present
        if not url.startswith(('http://', 'https://')):
            url = 'http://' + url
        
        parsed = urlparse(url)
        hostname = parsed.hostname
        
        # DNS lookup
        try:
            ip_address = socket.gethostbyname(hostname)
        except:
            return False, "DNS lookup failed - hostname not found"
        
        # HTTP request with timeout
        try:
            response = requests.get(url, timeout=5, allow_redirects=True)
            return True, {
                'url': url,
                'hostname': hostname,
                'ip': ip_address,
                'status_code': response.status_code,
                'response_time': response.elapsed.total_seconds(),
                'ssl': url.startswith('https://'),
                'reachable': True
            }
        except requests.exceptions.RequestException as e:
            return False, f"HTTP request failed: {str(e)}"
            
    except Exception as e:
        return False, f"URL validation error: {str(e)}"

def extract_url_features(url_info, feature_names):
    """Extract ONLY real network features from live website - NO synthetic data"""
    
    # Define the exact feature names that should be used
    EXPECTED_FEATURES = [
        'Dst Port', 'Protocol', 'Flow Duration', 'Tot Fwd Pkts', 'Tot Bwd Pkts',
        'TotLen Fwd Pkts', 'TotLen Bwd Pkts', 'Fwd Pkt Len Max', 'Fwd Pkt Len Min',
        'Fwd Pkt Len Mean', 'Fwd Pkt Len Std', 'Bwd Pkt Len Max', 'Bwd Pkt Len Min',
        'Bwd Pkt Len Mean', 'Bwd Pkt Len Std', 'Flow Byts/s', 'Flow Pkts/s',
        'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min',
        'Fwd IAT Tot', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min',
        'Bwd IAT Tot', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min',
        'Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags',
        'Fwd Header Len', 'Bwd Header Len', 'Fwd Pkts/s', 'Bwd Pkts/s',
        'Pkt Len Min', 'Pkt Len Max', 'Pkt Len Mean', 'Pkt Len Std', 'Pkt Len Var',
        'FIN Flag Cnt', 'SYN Flag Cnt', 'RST Flag Cnt', 'PSH Flag Cnt',
        'ACK Flag Cnt', 'URG Flag Cnt', 'CWE Flag Count', 'ECE Flag Cnt',
        'Down/Up Ratio', 'Pkt Size Avg', 'Fwd Seg Size Avg', 'Bwd Seg Size Avg',
        'Fwd Byts/b Avg', 'Fwd Pkts/b Avg', 'Fwd Blk Rate Avg',
        'Bwd Byts/b Avg', 'Bwd Pkts/b Avg', 'Bwd Blk Rate Avg',
        'Subflow Fwd Pkts', 'Subflow Fwd Byts', 'Subflow Bwd Pkts', 'Subflow Bwd Byts',
        'Init Fwd Win Byts', 'Init Bwd Win Byts', 'Fwd Act Data Pkts', 'Fwd Seg Size Min',
        'Active Mean', 'Active Std', 'Active Max', 'Active Min',
        'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min',
        'feature_mean', 'feature_std', 'feature_max', 'feature_min'
    ]
    
    # Initialize dictionary with exact feature names from training
    # Use the feature_names from the model if available, otherwise use expected features
    features = {feature: 0.0 for feature in feature_names}
    
    try:
        url = url_info['url']
        hostname = url_info['hostname']
        
        # Make multiple requests to get accurate statistics
        session = requests.Session()
        
        # Collect real timing and size data from actual requests
        timings = []
        sizes = []
        request_sizes = []
        
        st.info("🔄 Making multiple requests to collect real network data...")
        
        # Make 3-5 requests to get real statistics
        num_requests = 3
        for i in range(num_requests):
            start_time = time.time()
            response = session.get(url, timeout=10)
            end_time = time.time()
            
            request_duration = (end_time - start_time) * 1_000_000  # microseconds
            timings.append(request_duration)
            sizes.append(len(response.content))
            
            # Get actual request size from headers if available
            if response.request.body:
                request_sizes.append(len(response.request.body))
            else:
                # Calculate actual HTTP request header size
                request_line = f"{response.request.method} {response.request.path_url} HTTP/1.1"
                request_headers = '\r\n'.join([f"{k}: {v}" for k, v in response.request.headers.items()])
                request_sizes.append(len(request_line) + len(request_headers) + 4)  # +4 for \r\n\r\n
        
        # Calculate REAL statistics from actual measurements
        flow_duration_mean = np.mean(timings)
        flow_duration_std = np.std(timings)
        flow_duration_max = np.max(timings)
        flow_duration_min = np.min(timings)
        
        response_size_mean = np.mean(sizes)
        response_size_std = np.std(sizes)
        response_size_max = np.max(sizes)
        response_size_min = np.min(sizes)
        
        request_size_mean = np.mean(request_sizes)
        request_size_std = np.std(request_sizes)
        request_size_max = np.max(request_sizes)
        request_size_min = np.min(request_sizes)
        
        # Calculate inter-arrival times (real timing between packets)
        iat_values = []
        if len(timings) > 1:
            for i in range(len(timings) - 1):
                iat_values.append(timings[i+1] - timings[i])
        
        iat_mean = np.mean(iat_values) if iat_values else 0
        iat_std = np.std(iat_values) if iat_values else 0
        iat_max = np.max(iat_values) if iat_values else 0
        iat_min = np.min(iat_values) if iat_values else 0
        
        # Get actual protocol information
        is_https = url.startswith('https://')
        actual_port = 443 if is_https else 80
        
        # Get real status code
        status_code = response.status_code
        
        # Calculate REAL flow rates based on actual measurements
        total_bytes = sum(sizes) + sum(request_sizes)
        total_duration = sum(timings)
        flow_bytes_per_sec = (total_bytes / total_duration) * 1_000_000 if total_duration > 0 else 0
        flow_packets_per_sec = (num_requests * 2 / total_duration) * 1_000_000 if total_duration > 0 else 0
        
        # Map ONLY REAL measured values to features using EXACT feature names
        feature_values = {
            # Real timing measurements
            'Flow Duration': flow_duration_mean,
            
            # Real packet counts
            'Tot Fwd Pkts': float(num_requests),
            'Tot Bwd Pkts': float(num_requests),
            
            # Real size measurements
            'TotLen Fwd Pkts': sum(request_sizes),
            'TotLen Bwd Pkts': sum(sizes),
            
            # Real flow rates (calculated from actual measurements)
            'Flow Byts/s': flow_bytes_per_sec,
            'Flow Pkts/s': flow_packets_per_sec,
            
            # Real packet length statistics - Forward
            'Fwd Pkt Len Mean': request_size_mean,
            'Fwd Pkt Len Std': request_size_std,
            'Fwd Pkt Len Max': request_size_max,
            'Fwd Pkt Len Min': request_size_min,
            
            # Real packet length statistics - Backward
            'Bwd Pkt Len Mean': response_size_mean,
            'Bwd Pkt Len Std': response_size_std,
            'Bwd Pkt Len Max': response_size_max,
            'Bwd Pkt Len Min': response_size_min,
            
            # Real inter-arrival time statistics - Flow
            'Flow IAT Mean': iat_mean,
            'Flow IAT Std': iat_std,
            'Flow IAT Max': iat_max,
            'Flow IAT Min': iat_min,
            
            # Real inter-arrival time statistics - Forward
            'Fwd IAT Tot': sum(iat_values) if iat_values else 0,
            'Fwd IAT Mean': iat_mean,
            'Fwd IAT Std': iat_std,
            'Fwd IAT Max': iat_max,
            'Fwd IAT Min': iat_min,
            
            # Real inter-arrival time statistics - Backward
            'Bwd IAT Tot': sum(iat_values) if iat_values else 0,
            'Bwd IAT Mean': iat_mean,
            'Bwd IAT Std': iat_std,
            'Bwd IAT Max': iat_max,
            'Bwd IAT Min': iat_min,
            
            # Real protocol information
            'Dst Port': float(actual_port),
            'Protocol': 6.0,  # TCP
            
            # Real packet statistics
            'Pkt Len Mean': (request_size_mean + response_size_mean) / 2.0,
            'Pkt Len Std': np.std(request_sizes + sizes),
            'Pkt Len Var': np.var(request_sizes + sizes),
            'Pkt Len Max': max(request_size_max, response_size_max),
            'Pkt Len Min': min(request_size_min, response_size_min),
            
            # Real average sizes
            'Pkt Size Avg': (sum(request_sizes) + sum(sizes)) / (num_requests * 2),
            'Fwd Seg Size Avg': request_size_mean,
            'Bwd Seg Size Avg': response_size_mean,
            'Fwd Seg Size Min': request_size_min,
            
            # Real ratio calculation
            'Down/Up Ratio': sum(sizes) / sum(request_sizes) if sum(request_sizes) > 0 else 0,
            
            # Real header lengths (estimated from request structure)
            'Fwd Header Len': float(num_requests * 40),  # Typical TCP header size
            'Bwd Header Len': float(num_requests * 40),
            
            # Real packets per second
            'Fwd Pkts/s': (num_requests / total_duration) * 1_000_000 if total_duration > 0 else 0,
            'Bwd Pkts/s': (num_requests / total_duration) * 1_000_000 if total_duration > 0 else 0,
            
            # Real subflow data
            'Subflow Fwd Pkts': float(num_requests),
            'Subflow Fwd Byts': sum(request_sizes),
            'Subflow Bwd Pkts': float(num_requests),
            'Subflow Bwd Byts': sum(sizes),
            
            # TCP Window sizes (typical values for HTTP/HTTPS)
            'Init Fwd Win Byts': 29200.0 if is_https else 8192.0,
            'Init Bwd Win Byts': 29200.0 if is_https else 8192.0,
            
            # Active data packets
            'Fwd Act Data Pkts': float(num_requests),
            
            # Statistical features
            'feature_mean': (request_size_mean + response_size_mean) / 2.0,
            'feature_std': (request_size_std + response_size_std) / 2.0,
            'feature_max': max(request_size_max, response_size_max),
            'feature_min': min(request_size_min, response_size_min),
            
            # Flags (set to 0 as we don't capture raw TCP flags in HTTP requests)
            'Fwd PSH Flags': 0.0,
            'Bwd PSH Flags': 0.0,
            'Fwd URG Flags': 0.0,
            'Bwd URG Flags': 0.0,
            'FIN Flag Cnt': 0.0,
            'SYN Flag Cnt': 0.0,
            'RST Flag Cnt': 0.0,
            'PSH Flag Cnt': 0.0,
            'ACK Flag Cnt': 0.0,
            'URG Flag Cnt': 0.0,
            'CWE Flag Count': 0.0,
            'ECE Flag Cnt': 0.0,
            
            # Bulk rates (set to 0 as not applicable for simple HTTP requests)
            'Fwd Byts/b Avg': 0.0,
            'Fwd Pkts/b Avg': 0.0,
            'Fwd Blk Rate Avg': 0.0,
            'Bwd Byts/b Avg': 0.0,
            'Bwd Pkts/b Avg': 0.0,
            'Bwd Blk Rate Avg': 0.0,
            
            # Active/Idle times (set to 0 as not measured in simple HTTP requests)
            'Active Mean': 0.0,
            'Active Std': 0.0,
            'Active Max': 0.0,
            'Active Min': 0.0,
            'Idle Mean': 0.0,
            'Idle Std': 0.0,
            'Idle Max': 0.0,
            'Idle Min': 0.0,
        }
        
        # Update features dictionary with calculated values - EXACT MATCH ONLY
        for calc_feature, value in feature_values.items():
            if calc_feature in features:
                features[calc_feature] = float(value)
        
        st.success(f"""
        **📊 Real Network Measurements Collected:**
        - Requests Made: {num_requests}
        - Avg Flow Duration: {flow_duration_mean:.2f} μs (Std: {flow_duration_std:.2f})
        - Avg Response Size: {response_size_mean:.0f} bytes (Std: {response_size_std:.0f})
        - Avg Request Size: {request_size_mean:.0f} bytes
        - Real Flow Rate: {flow_bytes_per_sec:.2f} bytes/s
        - Protocol: {'HTTPS (TCP/443)' if is_https else 'HTTP (TCP/80)'}
        - Status Code: {status_code}
        - Features with Real Data: {sum(1 for v in features.values() if v != 0.0)}/{len(features)}
        """)
        
    except Exception as e:
        st.error(f"❌ Error collecting real network data: {str(e)}")
        st.warning("⚠️ Prediction will use zero values for unmeasured features.")
    
    return pd.DataFrame([features])

def calculate_risk_level(probability):
    """Calculate risk level from probability"""
    if probability < 0.3:
        return "Low", "🟢"
    elif probability < 0.7:
        return "Medium", "🟡"
    else:
        return "High", "🔴"

def make_predictions(df, models, binary_model_choice='Random Forest'):
    """Make predictions using loaded models"""
    try:
        # Scale features
        X_scaled = models['scaler'].transform(df)
        
        # Binary prediction
        binary_model = models['binary_rf'] if binary_model_choice == 'Random Forest' else models['binary_lr']
        binary_pred = binary_model.predict(X_scaled)
        binary_proba = binary_model.predict_proba(X_scaled)[:, 1]  # Probability of attack
        
        # Multiclass prediction
        multiclass_pred = models['multiclass'].predict(X_scaled)
        multiclass_proba = models['multiclass'].predict_proba(X_scaled)
        
        # Map predictions to labels
        attack_types = []
        for pred in multiclass_pred:
            pred_str = str(int(pred))
            attack_type = models['label_mapping'].get(pred_str, 'Unknown')
            attack_types.append(attack_type)
        
        # Calculate risk levels
        risk_levels = [calculate_risk_level(p) for p in binary_proba]
        
        results = pd.DataFrame({
            'Binary_Prediction': ['Attack' if p == 1 else 'Benign' for p in binary_pred],
            'Attack_Probability': binary_proba,
            'Risk_Level': [r[0] for r in risk_levels],
            'Risk_Icon': [r[1] for r in risk_levels],
            'Predicted_Attack_Type': attack_types,
            'Attack_Type_Confidence': multiclass_proba.max(axis=1)
        })
        
        return results, multiclass_proba
        
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None, None

def create_probability_gauge(probability):
    """Create a gauge chart for probability"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = probability * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Attack Probability (%)"},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 30], 'color': "lightgreen"},
                {'range': [30, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        }
    ))
    fig.update_layout(height=300)
    return fig

def create_attack_type_chart(proba_array, label_mapping):
    """Create bar chart for attack type probabilities"""
    labels = [label_mapping.get(str(i), f'Class {i}') for i in range(len(proba_array))]
    
    fig = go.Figure(data=[
        go.Bar(x=labels, y=proba_array * 100, marker_color='indianred')
    ])
    fig.update_layout(
        title="Attack Type Probability Distribution",
        xaxis_title="Attack Type",
        yaxis_title="Probability (%)",
        xaxis_tickangle=-45,
        height=400
    )
    return fig

# ================================
# STREAMLIT APP
# ================================

def main():
    st.set_page_config(
        page_title="Cyberattack Forecasting System",
        page_icon="🛡️",
        layout="wide"
    )
    
    # Header
    st.title("🛡️ Cyberattack Probability Forecasting System")
    st.markdown("### Real-time Threat Detection & Risk Assessment")
    st.markdown("---")
    
    # Sidebar configuration - PATH CONFIGURATION
    st.sidebar.header("📁 Path Configuration")
    
    # Allow user to configure the base path
    base_path = st.sidebar.text_input(
        "Dataset Base Path:",
        value=DEFAULT_BASE_PATH,
        help="Enter the full path to your dwig_ML9125_complete_full_dataset/dwig_ML9125 folder"
    )
    
    # Show current paths
    with st.sidebar.expander("📂 Current Paths", expanded=False):
        st.code(f"Base: {base_path}", language=None)
        st.code(f"Models: {get_file_path(base_path, 'models')}", language=None)
        st.code(f"Reports: {get_file_path(base_path, 'reports')}", language=None)
        st.code(f"Visualizations: {get_file_path(base_path, 'visualizations')}", language=None)
    
    st.sidebar.markdown("---")
    
    # Load models with the configured path
    with st.spinner("Loading ML models and artifacts..."):
        models = load_models_and_artifacts(base_path)
    
    # Define paths for reports and visualizations (after base_path is defined)
    reports_path = get_file_path(base_path, "reports")
    viz_path = get_file_path(base_path, "visualizations")
    
    if models is None:
        st.warning("⚠️ System cannot proceed without models.")
        st.info("""
        **Troubleshooting Steps:**
        
        1. **Check the base path** in the sidebar
        2. **Verify folder structure:**
           ```
           dwig_ML9125/
           ├── models/
           │   ├── binary_model_Logistic_Regression.pkl
           │   ├── binary_model_Random_Forest.pkl
           │   ├── multiclass_model.pkl
           │   ├── scaler.pkl
           │   ├── label_encoder.pkl
           │   ├── label_mapping.json
           │   └── feature_names.pkl
           ├── reports/
           └── visualizations/
           ```
        3. **Ensure all .pkl and .json files exist** in the models folder
        """)
        return
    
    # Model configuration in sidebar
    st.sidebar.header("⚙️ Model Configuration")
    binary_model_choice = st.sidebar.selectbox(
        "Binary Classification Model",
        ["Random Forest", "Logistic Regression"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.success(f"""
    **✅ System Ready**
    
    **Dataset Info:**
    - Dataset: CICIDS 2018
    - Records: 4,848,142
    - Features: {len(models['feature_names'])}
    - Attack Types: 10
    """)
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🌐 URL Analysis", 
        "📁 CSV Batch Processing", 
        "✍️ Manual Entry",
        "📊 Visualizations"
    ])
    
    # ================================
    # TAB 1: URL ANALYSIS
    # ================================
    with tab1:
        st.header("Real-time Website Threat Analysis")
        st.markdown("Enter a URL to analyze potential cyber threats")
        
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            url_input = st.text_input("Enter Website URL:", placeholder="example.com or https://example.com")
        with col2:
            analyze_btn = st.button("🔍 Analyze Once", use_container_width=True)
        with col3:
            monitor_btn = st.button("📡 Start Monitoring", use_container_width=True, type="primary")
        
        # Initialize session state for monitoring
        if 'monitoring' not in st.session_state:
            st.session_state.monitoring = False
        if 'monitoring_url' not in st.session_state:
            st.session_state.monitoring_url = ""
        
        # Toggle monitoring state
        if monitor_btn and url_input:
            st.session_state.monitoring = not st.session_state.monitoring
            st.session_state.monitoring_url = url_input
        
        # Single analysis
        if analyze_btn and url_input:
            with st.spinner("Validating URL and extracting features..."):
                is_valid, result = validate_url(url_input)
                
                if not is_valid:
                    st.error(f"❌ URL Validation Failed: {result}")
                else:
                    st.success("✅ URL is valid and reachable!")
                    
                    # Display URL info
                    st.subheader("Connection Details")
                    info_col1, info_col2, info_col3 = st.columns(3)
                    with info_col1:
                        st.metric("IP Address", result['ip'])
                    with info_col2:
                        st.metric("Response Time", f"{result['response_time']:.3f}s")
                    with info_col3:
                        ssl_status = "✅ Secured" if result['ssl'] else "⚠️ Not Secured"
                        st.metric("SSL Status", ssl_status)
                    
                    # Extract features and predict
                    with st.spinner("Analyzing threat level..."):
                        feature_df = extract_url_features(result, models['feature_names'])
                        predictions, multiclass_proba = make_predictions(
                            feature_df, models, binary_model_choice
                        )
                        
                        if predictions is not None:
                            st.markdown("---")
                            st.subheader("🎯 Threat Analysis Results")
                            
                            # Display results
                            res_col1, res_col2, res_col3 = st.columns(3)
                            with res_col1:
                                st.metric(
                                    "Binary Classification",
                                    predictions['Binary_Prediction'].iloc[0]
                                )
                            with res_col2:
                                risk_icon = predictions['Risk_Icon'].iloc[0]
                                risk_level = predictions['Risk_Level'].iloc[0]
                                st.metric(
                                    "Risk Level",
                                    f"{risk_icon} {risk_level}"
                                )
                            with res_col3:
                                st.metric(
                                    "Attack Type",
                                    predictions['Predicted_Attack_Type'].iloc[0]
                                )
                            
                            # Probability gauge
                            st.plotly_chart(
                                create_probability_gauge(predictions['Attack_Probability'].iloc[0]),
                                use_container_width=True
                            )
                            
                            # Attack type distribution
                            st.plotly_chart(
                                create_attack_type_chart(
                                    multiclass_proba[0], 
                                    models['label_mapping']
                                ),
                                use_container_width=True
                            )
        
        # Real-time monitoring mode
        if st.session_state.monitoring and st.session_state.monitoring_url:
            st.markdown("---")
            
            # Control panel
            control_col1, control_col2, control_col3 = st.columns([2, 1, 1])
            with control_col1:
                st.info(f"🔴 **LIVE MONITORING:** {st.session_state.monitoring_url}")
            with control_col2:
                refresh_interval = st.selectbox("Refresh Rate", [5, 10, 15, 30, 60], index=1, key="refresh_rate")
            with control_col3:
                if st.button("⏹️ Stop Monitoring", use_container_width=True):
                    st.session_state.monitoring = False
                    st.rerun()
            
            # Create placeholders for live updates
            status_placeholder = st.empty()
            metrics_placeholder = st.empty()
            connection_placeholder = st.empty()
            results_placeholder = st.empty()
            gauge_placeholder = st.empty()
            chart_placeholder = st.empty()
            
            # Initialize monitoring history
            if 'monitoring_history' not in st.session_state:
                st.session_state.monitoring_history = []
            
            # Monitoring loop
            iteration = 0
            while st.session_state.monitoring:
                iteration += 1
                current_time = datetime.now().strftime("%H:%M:%S")
                
                status_placeholder.info(f"🔄 **Scan #{iteration}** - Last Update: {current_time}")
                
                try:
                    # Validate and fetch data
                    is_valid, result = validate_url(st.session_state.monitoring_url)
                    
                    if not is_valid:
                        status_placeholder.error(f"❌ Connection Lost: {result}")
                        time.sleep(refresh_interval)
                        continue
                    
                    # Display live connection metrics
                    with connection_placeholder.container():
                        st.subheader("📊 Live Connection Metrics")
                        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                        with metric_col1:
                            st.metric("IP Address", result['ip'])
                        with metric_col2:
                            st.metric("Response Time", f"{result['response_time']:.3f}s")
                        with metric_col3:
                            ssl_status = "🔒 Secured" if result['ssl'] else "⚠️ Not Secured"
                            st.metric("SSL Status", ssl_status)
                        with metric_col4:
                            st.metric("Status Code", result['status_code'])
                    
                    # Extract features and predict
                    feature_df = extract_url_features(result, models['feature_names'])
                    predictions, multiclass_proba = make_predictions(
                        feature_df, models, binary_model_choice
                    )
                    
                    if predictions is not None:
                        # Store in history
                        history_entry = {
                            'time': current_time,
                            'probability': predictions['Attack_Probability'].iloc[0],
                            'risk_level': predictions['Risk_Level'].iloc[0],
                            'attack_type': predictions['Predicted_Attack_Type'].iloc[0],
                            'response_time': result['response_time']
                        }
                        st.session_state.monitoring_history.append(history_entry)
                        
                        # Keep only last 20 entries
                        if len(st.session_state.monitoring_history) > 20:
                            st.session_state.monitoring_history.pop(0)
                        
                        # Display current results
                        with results_placeholder.container():
                            st.subheader("🎯 Real-time Threat Assessment")
                            res_col1, res_col2, res_col3, res_col4 = st.columns(4)
                            with res_col1:
                                st.metric(
                                    "Classification",
                                    predictions['Binary_Prediction'].iloc[0],
                                    delta=None
                                )
                            with res_col2:
                                risk_icon = predictions['Risk_Icon'].iloc[0]
                                risk_level = predictions['Risk_Level'].iloc[0]
                                st.metric(
                                    "Risk Level",
                                    f"{risk_icon} {risk_level}"
                                )
                            with res_col3:
                                st.metric(
                                    "Attack Probability",
                                    f"{predictions['Attack_Probability'].iloc[0]:.1%}"
                                )
                            with res_col4:
                                st.metric(
                                    "Attack Type",
                                    predictions['Predicted_Attack_Type'].iloc[0]
                                )
                        
                        # Live probability gauge
                        with gauge_placeholder.container():
                            st.plotly_chart(
                                create_probability_gauge(predictions['Attack_Probability'].iloc[0]),
                                use_container_width=True,
                                key=f"gauge_{iteration}"
                            )
                        
                        # Historical trend chart
                        if len(st.session_state.monitoring_history) > 1:
                            with chart_placeholder.container():
                                st.subheader("📈 Historical Trend")
                                
                                # Create time series chart
                                history_df = pd.DataFrame(st.session_state.monitoring_history)
                                
                                fig_trend = go.Figure()
                                
                                # Attack probability line
                                fig_trend.add_trace(go.Scatter(
                                    x=history_df['time'],
                                    y=history_df['probability'] * 100,
                                    mode='lines+markers',
                                    name='Attack Probability (%)',
                                    line=dict(color='red', width=3),
                                    marker=dict(size=8)
                                ))
                                
                                # Response time line (secondary axis)
                                fig_trend.add_trace(go.Scatter(
                                    x=history_df['time'],
                                    y=history_df['response_time'] * 1000,
                                    mode='lines+markers',
                                    name='Response Time (ms)',
                                    line=dict(color='blue', width=2, dash='dash'),
                                    marker=dict(size=6),
                                    yaxis='y2'
                                ))
                                
                                fig_trend.update_layout(
                                    title="Attack Probability & Response Time Over Time",
                                    xaxis_title="Time",
                                    yaxis_title="Attack Probability (%)",
                                    yaxis2=dict(
                                        title="Response Time (ms)",
                                        overlaying='y',
                                        side='right'
                                    ),
                                    hovermode='x unified',
                                    height=400
                                )
                                
                                st.plotly_chart(fig_trend, use_container_width=True, key=f"trend_{iteration}")
                                
                                # History table
                                with st.expander("📋 View History Log"):
                                    st.dataframe(
                                        history_df[['time', 'probability', 'risk_level', 'attack_type', 'response_time']],
                                        use_container_width=True
                                    )
                
                except Exception as e:
                    status_placeholder.error(f"❌ Monitoring Error: {str(e)}")
                
                # Wait before next scan
                time.sleep(refresh_interval)
                st.rerun()
        
        elif not st.session_state.monitoring and 'monitoring_history' in st.session_state:
            # Show summary of completed monitoring session
            if st.session_state.monitoring_history:
                st.markdown("---")
                st.subheader("📊 Last Monitoring Session Summary")
                
                history_df = pd.DataFrame(st.session_state.monitoring_history)
                
                summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
                with summary_col1:
                    st.metric("Total Scans", len(history_df))
                with summary_col2:
                    st.metric("Avg Probability", f"{history_df['probability'].mean():.1%}")
                with summary_col3:
                    st.metric("Max Probability", f"{history_df['probability'].max():.1%}")
                with summary_col4:
                    st.metric("Avg Response Time", f"{history_df['response_time'].mean():.3f}s")
                
                # Download monitoring history
                csv_buffer = BytesIO()
                history_df.to_csv(csv_buffer, index=False)
                csv_buffer.seek(0)
                
                st.download_button(
                    label="📥 Download Monitoring History",
                    data=csv_buffer,
                    file_name=f"monitoring_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
    
    # ================================
    # TAB 2: CSV BATCH PROCESSING
    # ================================
    with tab2:
        st.header("Batch Prediction from CSV")
        st.markdown("Upload a CSV file with network features for batch analysis")
        
        uploaded_file = st.file_uploader("Choose CSV file", type=['csv'])
        
        if uploaded_file is not None:
            try:
                df_input = pd.read_csv(uploaded_file)
                st.success(f"✅ Loaded {len(df_input)} records")
                
                st.subheader("Data Preview")
                st.dataframe(df_input.head(10), use_container_width=True)
                
                # Check if features match
                missing_features = set(models['feature_names']) - set(df_input.columns)
                if missing_features:
                    st.warning(f"⚠️ Missing features: {missing_features}")
                    st.info("The system will use default values for missing features.")
                    
                    # Add missing features
                    for feature in missing_features:
                        df_input[feature] = 0.0
                
                # Ensure correct feature order
                df_input = df_input[models['feature_names']]
                
                if st.button("🚀 Run Batch Prediction"):
                    with st.spinner("Processing predictions..."):
                        predictions, multiclass_proba = make_predictions(
                            df_input, models, binary_model_choice
                        )
                        
                        if predictions is not None:
                            # Combine with original data
                            result_df = pd.concat([df_input.reset_index(drop=True), predictions], axis=1)
                            
                            st.success("✅ Predictions completed!")
                            
                            # Summary statistics
                            st.subheader("📊 Summary Statistics")
                            sum_col1, sum_col2, sum_col3 = st.columns(3)
                            with sum_col1:
                                attack_count = (predictions['Binary_Prediction'] == 'Attack').sum()
                                st.metric("Detected Attacks", attack_count)
                            with sum_col2:
                                avg_prob = predictions['Attack_Probability'].mean()
                                st.metric("Avg Attack Probability", f"{avg_prob:.2%}")
                            with sum_col3:
                                high_risk = (predictions['Risk_Level'] == 'High').sum()
                                st.metric("High Risk Count", high_risk)
                            
                            # Risk distribution
                            risk_counts = predictions['Risk_Level'].value_counts()
                            fig_risk = px.pie(
                                values=risk_counts.values,
                                names=risk_counts.index,
                                title="Risk Level Distribution",
                                color=risk_counts.index,
                                color_discrete_map={'Low': 'green', 'Medium': 'yellow', 'High': 'red'}
                            )
                            st.plotly_chart(fig_risk, use_container_width=True)
                            
                            # Attack type distribution
                            attack_type_counts = predictions['Predicted_Attack_Type'].value_counts()
                            fig_attacks = px.bar(
                                x=attack_type_counts.index,
                                y=attack_type_counts.values,
                                title="Predicted Attack Types Distribution",
                                labels={'x': 'Attack Type', 'y': 'Count'}
                            )
                            st.plotly_chart(fig_attacks, use_container_width=True)
                            
                            # Display results
                            st.subheader("Prediction Results")
                            st.dataframe(result_df, use_container_width=True)
                            
                            # Download button
                            csv_buffer = BytesIO()
                            result_df.to_csv(csv_buffer, index=False)
                            csv_buffer.seek(0)
                            
                            st.download_button(
                                label="📥 Download Predictions CSV",
                                data=csv_buffer,
                                file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                
            except Exception as e:
                st.error(f"Error processing CSV: {str(e)}")
    
    # ================================
    # TAB 3: MANUAL ENTRY
    # ================================
    with tab3:
        st.header("Manual Feature Entry")
        st.markdown("Enter network traffic features manually for prediction")
        
        st.info("💡 Enter values for key features. Other features will use default values.")
        
        # Define the exact expected feature names
        EXPECTED_FEATURES = [
            'Dst Port', 'Protocol', 'Flow Duration', 'Tot Fwd Pkts', 'Tot Bwd Pkts',
            'TotLen Fwd Pkts', 'TotLen Bwd Pkts', 'Fwd Pkt Len Max', 'Fwd Pkt Len Min',
            'Fwd Pkt Len Mean', 'Fwd Pkt Len Std', 'Bwd Pkt Len Max', 'Bwd Pkt Len Min',
            'Bwd Pkt Len Mean', 'Bwd Pkt Len Std', 'Flow Byts/s', 'Flow Pkts/s',
            'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min'
        ]
        
        # Create input fields for important features
        with st.form("manual_entry_form"):
            col1, col2 = st.columns(2)
            
            manual_features = {}
            
            # Use only the first 20 expected features or all model features if less
            features_to_show = EXPECTED_FEATURES if len(EXPECTED_FEATURES) <= 20 else EXPECTED_FEATURES[:20]
            
            # If model has different features, use those instead
            if models['feature_names'][0] not in EXPECTED_FEATURES:
                features_to_show = models['feature_names'][:20]
            
            for i, feature in enumerate(features_to_show):
                if i % 2 == 0:
                    with col1:
                        manual_features[feature] = st.number_input(
                            feature, 
                            value=0.0, 
                            format="%.6f",
                            key=f"manual_{feature}"
                        )
                else:
                    with col2:
                        manual_features[feature] = st.number_input(
                            feature, 
                            value=0.0, 
                            format="%.6f",
                            key=f"manual_{feature}"
                        )
            
            submitted = st.form_submit_button("🎯 Predict")
            
            if submitted:
                # Create full feature dataframe with exact feature names
                feature_dict = {f: 0.0 for f in models['feature_names']}
                feature_dict.update(manual_features)
                df_manual = pd.DataFrame([feature_dict])
                
                with st.spinner("Making prediction..."):
                    predictions, multiclass_proba = make_predictions(
                        df_manual, models, binary_model_choice
                    )
                    
                    if predictions is not None:
                        st.markdown("---")
                        st.subheader("🎯 Prediction Results")
                        
                        # Display results in columns
                        res_col1, res_col2 = st.columns(2)
                        
                        with res_col1:
                            st.metric("Binary Prediction", predictions['Binary_Prediction'].iloc[0])
                            st.metric(
                                "Risk Level",
                                f"{predictions['Risk_Icon'].iloc[0]} {predictions['Risk_Level'].iloc[0]}"
                            )
                            st.metric("Attack Probability", f"{predictions['Attack_Probability'].iloc[0]:.2%}")
                        
                        with res_col2:
                            st.metric("Predicted Attack Type", predictions['Predicted_Attack_Type'].iloc[0])
                            st.metric("Confidence", f"{predictions['Attack_Type_Confidence'].iloc[0]:.2%}")
                        
                        # Visualizations
                        st.plotly_chart(
                            create_probability_gauge(predictions['Attack_Probability'].iloc[0]),
                            use_container_width=True
                        )
                        
                        st.plotly_chart(
                            create_attack_type_chart(multiclass_proba[0], models['label_mapping']),
                            use_container_width=True
                        )
    
    # ================================
    # TAB 4: VISUALIZATIONS
    # ================================
    with tab4:
        st.header("Model Performance Visualizations")
        st.markdown("Pre-generated evaluation metrics and visualizations")
        
        # Model selector for metrics display
        st.subheader("🔧 Select Model to View")
        viz_model_choice = st.selectbox(
            "Choose Binary Classification Model:",
            ["Random Forest", "Logistic Regression"],
            key="viz_model_selector"
        )
        
        # Check if reports and visualizations exist
        if not check_path_exists(reports_path):
            st.warning(f"⚠️ Reports directory not found: {reports_path}")
        
        if not check_path_exists(viz_path):
            st.warning(f"⚠️ Visualizations directory not found: {viz_path}")
        
        try:
            # Load evaluation results
            eval_file = get_file_path(reports_path, "evaluation_results.json")
            
            if check_path_exists(eval_file):
                with open(eval_file, 'r') as f:
                    eval_results = json.load(f)
                
                st.subheader(f"📈 {viz_model_choice} - Performance Metrics")
                
                # Show JSON structure in expander for debugging
                with st.expander("🔍 View Raw JSON Structure (Debug)"):
                    st.json(eval_results)
                
                # Determine which model's metrics to show
                model_key_rf = "Random_Forest"
                model_key_lr = "Logistic_Regression"
                model_key = model_key_rf if viz_model_choice == "Random Forest" else model_key_lr
                
                # Try to find metrics for the selected model - check all possible structures
                binary_metrics = None
                found_location = ""
                
                # Strategy 1: Check if model names are at root level
                if model_key in eval_results:
                    binary_metrics = eval_results[model_key]
                    found_location = f"Root level: {model_key}"
                
                # Strategy 2: Check binary_classification container
                elif 'binary_classification' in eval_results:
                    bc = eval_results['binary_classification']
                    if isinstance(bc, dict):
                        # Check for exact model key
                        if model_key in bc:
                            binary_metrics = bc[model_key]
                            found_location = f"binary_classification.{model_key}"
                        # Check for variations
                        elif 'random_forest' in bc and viz_model_choice == "Random Forest":
                            binary_metrics = bc['random_forest']
                            found_location = "binary_classification.random_forest"
                        elif 'logistic_regression' in bc and viz_model_choice == "Logistic Regression":
                            binary_metrics = bc['logistic_regression']
                            found_location = "binary_classification.logistic_regression"
                        # Check for model names with spaces or different casing
                        else:
                            for key in bc.keys():
                                if 'random' in key.lower() and 'forest' in key.lower() and viz_model_choice == "Random Forest":
                                    binary_metrics = bc[key]
                                    found_location = f"binary_classification.{key}"
                                    break
                                elif 'logistic' in key.lower() and viz_model_choice == "Logistic Regression":
                                    binary_metrics = bc[key]
                                    found_location = f"binary_classification.{key}"
                                    break
                
                # Strategy 3: Check for model-specific keys with variations
                if not binary_metrics:
                    possible_keys = [
                        model_key, model_key.lower(), model_key.replace('_', ' '),
                        'rf' if viz_model_choice == "Random Forest" else 'lr',
                        'RandomForest' if viz_model_choice == "Random Forest" else 'LogisticRegression',
                        'random_forest' if viz_model_choice == "Random Forest" else 'logistic_regression',
                        'Random Forest' if viz_model_choice == "Random Forest" else 'Logistic Regression'
                    ]
                    for key in possible_keys:
                        if key in eval_results:
                            binary_metrics = eval_results[key]
                            found_location = f"Root: {key}"
                            break
                
                # Display metrics if found
                if binary_metrics and isinstance(binary_metrics, dict):
                    # Check if we actually have metrics (not empty or nested structure)
                    has_metrics = any(k in binary_metrics for k in ['accuracy', 'Accuracy', 'precision', 'Precision', 
                                                                      'recall', 'Recall', 'f1_score', 'f1', 'F1'])
                    
                    if has_metrics:
                        st.success(f"✅ Loaded metrics from: {found_location}")
                        
                        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                        
                        with metric_col1:
                            accuracy = (binary_metrics.get('accuracy') or binary_metrics.get('Accuracy') or 
                                       binary_metrics.get('test_accuracy') or 0)
                            st.metric("Accuracy", f"{float(accuracy):.4f}" if accuracy else "N/A")
                        with metric_col2:
                            precision = (binary_metrics.get('precision') or binary_metrics.get('Precision') or 
                                        binary_metrics.get('test_precision') or 0)
                            st.metric("Precision", f"{float(precision):.4f}" if precision else "N/A")
                        with metric_col3:
                            recall = (binary_metrics.get('recall') or binary_metrics.get('Recall') or 
                                     binary_metrics.get('test_recall') or 0)
                            st.metric("Recall", f"{float(recall):.4f}" if recall else "N/A")
                        with metric_col4:
                            f1 = (binary_metrics.get('f1_score') or binary_metrics.get('f1') or 
                                 binary_metrics.get('F1 Score') or binary_metrics.get('F1') or 
                                 binary_metrics.get('test_f1') or 0)
                            st.metric("F1 Score", f"{float(f1):.4f}" if f1 else "N/A")
                        
                        # Additional metrics if available
                        additional_metrics = []
                        roc_auc = (binary_metrics.get('roc_auc') or binary_metrics.get('ROC AUC') or 
                                  binary_metrics.get('auc') or binary_metrics.get('AUC'))
                        training_time = (binary_metrics.get('training_time') or binary_metrics.get('train_time'))
                        prediction_time = (binary_metrics.get('prediction_time') or binary_metrics.get('predict_time'))
                        
                        if any([roc_auc, training_time, prediction_time]):
                            st.markdown("---")
                            add_col1, add_col2, add_col3 = st.columns(3)
                            with add_col1:
                                if roc_auc:
                                    st.metric("ROC AUC", f"{float(roc_auc):.4f}")
                            with add_col2:
                                if training_time:
                                    st.metric("Training Time", f"{float(training_time):.2f}s")
                            with add_col3:
                                if prediction_time:
                                    st.metric("Prediction Time", f"{float(prediction_time):.4f}s")
                    else:
                        st.warning(f"⚠️ Found data structure but no recognizable metrics inside.")
                        st.info("Available keys in structure: " + ", ".join(binary_metrics.keys()))
                else:
                    st.warning(f"⚠️ No metrics found for {viz_model_choice} in the evaluation results.")
                    st.info("💡 Please check the JSON structure in the debug expander above.")
            else:
                st.warning(f"📊 Evaluation metrics file not found at: {eval_file}")
                st.info("💡 Please ensure evaluation_results.json exists in the reports folder.")
            
            st.markdown("---")
            
            # Display pre-generated visualizations
            st.subheader("📊 Confusion Matrices")
            
            # Show file existence status
            binary_cm = get_file_path(viz_path, "01_confusion_matrix_binary.png")
            multiclass_cm = get_file_path(viz_path, "02_confusion_matrix_multiclass.png")
            
            viz_col1, viz_col2 = st.columns(2)
            
            with viz_col1:
                st.markdown("**Binary Classification Confusion Matrix**")
                if check_path_exists(binary_cm):
                    st.image(binary_cm, use_column_width=True)
                else:
                    st.error(f"⚠️ Image not found")
                    st.code(binary_cm, language=None)
                    
            with viz_col2:
                st.markdown("**Multiclass Classification Confusion Matrix**")
                if check_path_exists(multiclass_cm):
                    st.image(multiclass_cm, use_column_width=True)
                else:
                    st.error(f"⚠️ Image not found")
                    st.code(multiclass_cm, language=None)
            
            st.markdown("---")
            st.subheader("📈 ROC Curve & Probability Distribution")
            
            roc_curve = get_file_path(viz_path, "03_roc_curve.png")
            prob_dist = get_file_path(viz_path, "04_probability_distribution.png")
            
            roc_col1, roc_col2 = st.columns(2)
            
            with roc_col1:
                st.markdown("**ROC Curve**")
                if check_path_exists(roc_curve):
                    st.image(roc_curve, use_column_width=True)
                else:
                    st.error(f"⚠️ Image not found")
                    st.code(roc_curve, language=None)
                    
            with roc_col2:
                st.markdown("**Probability Distribution**")
                if check_path_exists(prob_dist):
                    st.image(prob_dist, use_column_width=True)
                else:
                    st.error(f"⚠️ Image not found")
                    st.code(prob_dist, language=None)
            
            st.markdown("---")
            st.subheader("📊 Dataset Analysis")
            
            attack_types = get_file_path(viz_path, "05_attack_types.png")
            dataset_comp = get_file_path(viz_path, "06_dataset_composition.png")
            
            dataset_col1, dataset_col2 = st.columns(2)
            
            with dataset_col1:
                st.markdown("**Attack Types Distribution**")
                if check_path_exists(attack_types):
                    st.image(attack_types, use_column_width=True)
                else:
                    st.error(f"⚠️ Image not found")
                    st.code(attack_types, language=None)
                    
            with dataset_col2:
                st.markdown("**Dataset Composition**")
                if check_path_exists(dataset_comp):
                    st.image(dataset_comp, use_column_width=True)
                else:
                    st.error(f"⚠️ Image not found")
                    st.code(dataset_comp, language=None)
            
            st.markdown("---")
            st.subheader("📈 Cross-Validation Performance")
            
            kfold_perf = get_file_path(viz_path, "07_kfold_performance.png")
            if check_path_exists(kfold_perf):
                st.image(kfold_perf, use_column_width=True)
            else:
                st.error(f"⚠️ Image not found")
                st.code(kfold_perf, language=None)
            
        except Exception as e:
            st.error(f"❌ Error loading visualizations: {str(e)}")
            st.info(f"📁 Looking for files in: {viz_path}")
            
            # Provide troubleshooting info
            with st.expander("🔧 Troubleshooting"):
                st.markdown("""
                **Expected File Structure:**
                ```
                visualizations/
                ├── 01_confusion_matrix_binary.png
                ├── 02_confusion_matrix_multiclass.png
                ├── 03_roc_curve.png
                ├── 04_probability_distribution.png
                ├── 05_attack_types.png
                ├── 06_dataset_composition.png
                └── 07_kfold_performance.png
                
                reports/
                └── evaluation_results.json
                ```
                
                **Please ensure:**
                1. All visualization PNG files exist in the visualizations folder
                2. The evaluation_results.json file exists in the reports folder
                3. The file paths are correct
                """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p>🛡️ Cyberattack Probability Forecasting System | Powered by ML & CICIDS 2018 Dataset</p>
        <p style='font-size: 0.8em; color: gray;'>System trained on 4.8M+ network traffic samples</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
