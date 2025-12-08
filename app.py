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
from io import BytesIO
import os
import warnings

warnings.filterwarnings('ignore')

# Plotly imports
import plotly.graph_objects as go
import plotly.express as px

# ================================
# CONFIGURATION
# ================================
DEFAULT_BASE_PATH = "dwig_ML9125"

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
    
    if not check_path_exists(base_path):
        st.error(f"❌ Base directory not found: {base_path}")
        return None
    
    if not check_path_exists(models_path):
        st.error(f"❌ Models directory not found: {models_path}")
        return None
    
    missing_files = []
    loaded_models = {}
    
    try:
        files_to_load = {
            'binary_lr': 'binary_model_Logistic_Regression.pkl',
            'binary_rf': 'binary_model_Random_Forest.pkl',
            'multiclass': 'multiclass_model.pkl',
            'scaler': 'scaler.pkl',
            'label_encoder': 'label_encoder.pkl',
            'label_mapping': 'label_mapping.json',
            'feature_names': 'feature_names.pkl'
        }
        
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
        
        if missing_files:
            st.error(f"❌ Missing required files:")
            for file in missing_files:
                st.error(f"   • {file}")
            return None
        
        st.success(f"✅ Successfully loaded {len(loaded_models)} model artifacts!")
        return loaded_models
        
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        return None

def validate_url(url):
    """Validate if URL is reachable"""
    try:
        if not url.startswith(('http://', 'https://')):
            url = 'http://' + url
        
        parsed = urlparse(url)
        hostname = parsed.hostname
        
        if not hostname:
            return False, "Invalid URL format"
        
        try:
            ip_address = socket.gethostbyname(hostname)
        except:
            return False, "DNS lookup failed"
        
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
            return False, f"Connection failed: {str(e)}"
            
    except Exception as e:
        return False, f"Error: {str(e)}"

def extract_url_features(url_info, feature_names):
    """Extract network features from URL"""
    features = {feature: 0.0 for feature in feature_names}
    
    try:
        url = url_info['url']
        session = requests.Session()
        
        timings = []
        sizes = []
        request_sizes = []
        
        st.info("🔄 Collecting network data...")
        
        num_requests = 3
        for i in range(num_requests):
            try:
                start_time = time.time()
                response = session.get(url, timeout=10)
                end_time = time.time()
                
                request_duration = (end_time - start_time) * 1_000_000
                timings.append(request_duration)
                sizes.append(len(response.content))
                
                if response.request.body:
                    request_sizes.append(len(response.request.body))
                else:
                    request_sizes.append(200)  # Default header size
            except:
                continue
        
        if not timings:
            st.warning("⚠️ Could not collect network data")
            return pd.DataFrame([features])
        
        flow_duration_mean = np.mean(timings)
        response_size_mean = np.mean(sizes)
        request_size_mean = np.mean(request_sizes)
        
        is_https = url.startswith('https://')
        actual_port = 443 if is_https else 80
        
        total_bytes = sum(sizes) + sum(request_sizes)
        total_duration = sum(timings)
        flow_bytes_per_sec = (total_bytes / total_duration) * 1_000_000 if total_duration > 0 else 0
        
        feature_values = {
            'Flow Duration': flow_duration_mean,
            'Tot Fwd Pkts': float(num_requests),
            'Tot Bwd Pkts': float(num_requests),
            'TotLen Fwd Pkts': sum(request_sizes),
            'TotLen Bwd Pkts': sum(sizes),
            'Flow Byts/s': flow_bytes_per_sec,
            'Dst Port': float(actual_port),
            'Protocol': 6.0,
        }
        
        for calc_feature, value in feature_values.items():
            if calc_feature in features:
                features[calc_feature] = float(value)
        
        st.success("✅ Network data collected!")
        
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
    
    return pd.DataFrame([features])

def calculate_risk_level(probability):
    """Calculate risk level"""
    if probability < 0.3:
        return "Low", "🟢"
    elif probability < 0.7:
        return "Medium", "🟡"
    else:
        return "High", "🔴"

def make_predictions(df, models, binary_model_choice='Random Forest'):
    """Make predictions"""
    try:
        X_scaled = models['scaler'].transform(df)
        
        binary_model = models['binary_rf'] if binary_model_choice == 'Random Forest' else models['binary_lr']
        binary_pred = binary_model.predict(X_scaled)
        binary_proba = binary_model.predict_proba(X_scaled)[:, 1]
        
        multiclass_pred = models['multiclass'].predict(X_scaled)
        multiclass_proba = models['multiclass'].predict_proba(X_scaled)
        
        attack_types = []
        for pred in multiclass_pred:
            pred_str = str(int(pred))
            attack_type = models['label_mapping'].get(pred_str, 'Unknown')
            attack_types.append(attack_type)
        
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
    """Create gauge chart"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = probability * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Attack Probability (%)"},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 30], 'color': "lightgreen"},
                {'range': [30, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "red"}
            ]
        }
    ))
    fig.update_layout(height=250)
    return fig

def create_attack_type_chart(proba_array, label_mapping):
    """Create bar chart"""
    labels = [label_mapping.get(str(i), f'Class {i}') for i in range(len(proba_array))]
    
    fig = go.Figure(data=[
        go.Bar(x=labels, y=proba_array * 100, marker_color='indianred')
    ])
    fig.update_layout(
        title="Attack Type Probabilities",
        xaxis_title="Attack Type",
        yaxis_title="Probability (%)",
        xaxis_tickangle=-45,
        height=350
    )
    return fig

# ================================
# MAIN APP
# ================================

def main():
    st.set_page_config(
        page_title="Cyberattack Forecasting",
        page_icon="🛡️",
        layout="wide"
    )
    
    st.title("🛡️ Cyberattack Probability Forecasting System")
    st.markdown("### Real-time Threat Detection & Risk Assessment")
    st.markdown("---")
    
    # Load models
    with st.spinner("Loading models..."):
        models = load_models_and_artifacts(DEFAULT_BASE_PATH)
    
    if models is None:
        st.stop()
    
    # Sidebar
    st.sidebar.header("⚙️ Configuration")
    binary_model_choice = st.sidebar.selectbox(
        "Binary Model",
        ["Random Forest", "Logistic Regression"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.success(f"""
    **✅ System Ready**
    
    Dataset: CICIDS 2018  
    Features: {len(models['feature_names'])}  
    Attack Types: 10
    """)
    
    # Tabs
    tab1, tab2, tab3 = st.tabs([
        "🌐 URL Analysis", 
        "📁 CSV Analysis", 
        "✍️ Manual Entry"
    ])
    
    # TAB 1: URL Analysis
    with tab1:
        st.header("Website Threat Analysis")
        
        url_input = st.text_input("Enter URL:", placeholder="example.com")
        
        if st.button("🔍 Analyze", type="primary"):
            if not url_input:
                st.warning("Please enter a URL")
            else:
                with st.spinner("Analyzing..."):
                    is_valid, result = validate_url(url_input)
                    
                    if not is_valid:
                        st.error(f"❌ {result}")
                    else:
                        st.success("✅ URL reachable!")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("IP", result['ip'])
                        with col2:
                            st.metric("Response", f"{result['response_time']:.3f}s")
                        with col3:
                            st.metric("SSL", "✅" if result['ssl'] else "⚠️")
                        
                        feature_df = extract_url_features(result, models['feature_names'])
                        predictions, multiclass_proba = make_predictions(
                            feature_df, models, binary_model_choice
                        )
                        
                        if predictions is not None:
                            st.markdown("---")
                            st.subheader("🎯 Results")
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Classification", predictions['Binary_Prediction'].iloc[0])
                            with col2:
                                st.metric("Risk", 
                                    f"{predictions['Risk_Icon'].iloc[0]} {predictions['Risk_Level'].iloc[0]}")
                            with col3:
                                st.metric("Attack Type", predictions['Predicted_Attack_Type'].iloc[0])
                            
                            st.plotly_chart(
                                create_probability_gauge(predictions['Attack_Probability'].iloc[0]),
                                use_container_width=True
                            )
    
    # TAB 2: CSV Analysis
    with tab2:
        st.header("Batch CSV Analysis")
        
        uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
        
        if uploaded_file is not None:
            try:
                df_input = pd.read_csv(uploaded_file)
                st.success(f"✅ Loaded {len(df_input)} records")
                
                st.dataframe(df_input.head(5))
                
                missing = set(models['feature_names']) - set(df_input.columns)
                if missing:
                    st.warning(f"⚠️ Missing {len(missing)} features")
                    for f in missing:
                        df_input[f] = 0.0
                
                df_input = df_input[models['feature_names']]
                
                if st.button("🚀 Predict"):
                    with st.spinner("Processing..."):
                        predictions, _ = make_predictions(df_input, models, binary_model_choice)
                        
                        if predictions is not None:
                            result_df = pd.concat([df_input.reset_index(drop=True), predictions], axis=1)
                            st.success("✅ Done!")
                            st.dataframe(result_df)
                            
                            csv = result_df.to_csv(index=False)
                            st.download_button(
                                "📥 Download",
                                csv,
                                "predictions.csv",
                                "text/csv"
                            )
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    # TAB 3: Manual Entry
    with tab3:
        st.header("Manual Entry")
        st.info("Enter feature values manually")
        
        with st.form("manual_form"):
            col1, col2 = st.columns(2)
            
            manual_features = {}
            sample_features = [
                'Dst Port', 'Protocol', 'Flow Duration', 'Tot Fwd Pkts',
                'Tot Bwd Pkts', 'TotLen Fwd Pkts', 'TotLen Bwd Pkts', 'Flow Byts/s'
            ]
            
            features_to_show = [f for f in sample_features if f in models['feature_names']]
            
            for i, feature in enumerate(features_to_show):
                with col1 if i % 2 == 0 else col2:
                    manual_features[feature] = st.number_input(
                        feature, value=0.0, format="%.2f"
                    )
            
            if st.form_submit_button("🎯 Predict"):
                feature_dict = {f: 0.0 for f in models['feature_names']}
                feature_dict.update(manual_features)
                df_manual = pd.DataFrame([feature_dict])
                
                predictions, _ = make_predictions(df_manual, models, binary_model_choice)
                
                if predictions is not None:
                    st.success("✅ Prediction complete!")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Prediction", predictions['Binary_Prediction'].iloc[0])
                    with col2:
                        st.metric("Risk", 
                            f"{predictions['Risk_Icon'].iloc[0]} {predictions['Risk_Level'].iloc[0]}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p>🛡️ Cyberattack Forecasting | CICIDS 2018</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
