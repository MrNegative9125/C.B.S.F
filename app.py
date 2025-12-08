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

# Import plotly with error handling
try:
    import plotly.graph_objects as go
    import plotly.express as px
except ImportError:
    st.error("Plotly not installed. Installing...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'plotly'])
    import plotly.graph_objects as go
    import plotly.express as px

# ================================
# CONFIGURATION FOR GITHUB DEPLOYMENT
# ================================
# This will automatically work when deployed on Streamlit Cloud
# The path "dwig_ML9125" matches your GitHub repo structure
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
    """Validate if URL is reachable and extract basic information"""
    try:
        if not url.startswith(('http://', 'https://')):
            url = 'http://' + url
        
        parsed = urlparse(url)
        hostname = parsed.hostname
        
        try:
            ip_address = socket.gethostbyname(hostname)
        except:
            return False, "DNS lookup failed - hostname not found"
        
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
    """Extract real network features from live website"""
    features = {feature: 0.0 for feature in feature_names}
    
    try:
        url = url_info['url']
        session = requests.Session()
        
        timings = []
        sizes = []
        request_sizes = []
        
        st.info("🔄 Collecting real network data from website...")
        
        num_requests = 3
        for i in range(num_requests):
            start_time = time.time()
            response = session.get(url, timeout=10)
            end_time = time.time()
            
            request_duration = (end_time - start_time) * 1_000_000
            timings.append(request_duration)
            sizes.append(len(response.content))
            
            if response.request.body:
                request_sizes.append(len(response.request.body))
            else:
                request_line = f"{response.request.method} {response.request.path_url} HTTP/1.1"
                request_headers = '\r\n'.join([f"{k}: {v}" for k, v in response.request.headers.items()])
                request_sizes.append(len(request_line) + len(request_headers) + 4)
        
        flow_duration_mean = np.mean(timings)
        flow_duration_std = np.std(timings)
        response_size_mean = np.mean(sizes)
        response_size_std = np.std(sizes)
        request_size_mean = np.mean(request_sizes)
        
        iat_values = []
        if len(timings) > 1:
            for i in range(len(timings) - 1):
                iat_values.append(timings[i+1] - timings[i])
        
        iat_mean = np.mean(iat_values) if iat_values else 0
        iat_std = np.std(iat_values) if iat_values else 0
        
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
            'Flow Pkts/s': (num_requests * 2 / total_duration) * 1_000_000 if total_duration > 0 else 0,
            'Fwd Pkt Len Mean': request_size_mean,
            'Fwd Pkt Len Std': response_size_std,
            'Bwd Pkt Len Mean': response_size_mean,
            'Bwd Pkt Len Std': response_size_std,
            'Flow IAT Mean': iat_mean,
            'Flow IAT Std': iat_std,
            'Fwd IAT Mean': iat_mean,
            'Bwd IAT Mean': iat_mean,
            'Dst Port': float(actual_port),
            'Protocol': 6.0,
            'Pkt Len Mean': (request_size_mean + response_size_mean) / 2.0,
            'Pkt Size Avg': (sum(request_sizes) + sum(sizes)) / (num_requests * 2),
            'Init Fwd Win Byts': 29200.0 if is_https else 8192.0,
            'Init Bwd Win Byts': 29200.0 if is_https else 8192.0,
        }
        
        for calc_feature, value in feature_values.items():
            if calc_feature in features:
                features[calc_feature] = float(value)
        
        st.success(f"""
        **📊 Network Data Collected:**
        - Requests: {num_requests}
        - Avg Response Size: {response_size_mean:.0f} bytes
        - Flow Rate: {flow_bytes_per_sec:.2f} bytes/s
        - Protocol: {'HTTPS' if is_https else 'HTTP'}
        """)
        
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
    
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
    
    st.title("🛡️ Cyberattack Probability Forecasting System")
    st.markdown("### Real-time Threat Detection & Risk Assessment")
    st.markdown("---")
    
    # Load models
    with st.spinner("Loading ML models..."):
        models = load_models_and_artifacts(DEFAULT_BASE_PATH)
    
    if models is None:
        st.error("⚠️ Failed to load models. Please check the repository structure.")
        st.info("""
        **Expected structure:**
        ```
        C.B.S.F/
        ├── app.py (this file)
        └── dwig_ML9125/
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
        """)
        return
    
    # Sidebar
    st.sidebar.header("⚙️ Model Configuration")
    binary_model_choice = st.sidebar.selectbox(
        "Binary Classification Model",
        ["Random Forest", "Logistic Regression"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.success(f"""
    **✅ System Ready**
    
    **Dataset:** CICIDS 2018
    **Features:** {len(models['feature_names'])}
    **Attack Types:** 10
    """)
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🌐 URL Analysis", 
        "📁 CSV Batch Processing", 
        "✍️ Manual Entry",
        "📊 Visualizations"
    ])
    
    # TAB 1: URL ANALYSIS
    with tab1:
        st.header("Real-time Website Threat Analysis")
        
        col1, col2 = st.columns([4, 1])
        with col1:
            url_input = st.text_input("Enter Website URL:", placeholder="example.com")
        with col2:
            analyze_btn = st.button("🔍 Analyze", use_container_width=True, type="primary")
        
        if analyze_btn and url_input:
            with st.spinner("Validating URL..."):
                is_valid, result = validate_url(url_input)
                
                if not is_valid:
                    st.error(f"❌ {result}")
                else:
                    st.success("✅ URL is reachable!")
                    
                    info_col1, info_col2, info_col3 = st.columns(3)
                    with info_col1:
                        st.metric("IP Address", result['ip'])
                    with info_col2:
                        st.metric("Response Time", f"{result['response_time']:.3f}s")
                    with info_col3:
                        st.metric("SSL", "✅ Yes" if result['ssl'] else "⚠️ No")
                    
                    with st.spinner("Analyzing threat..."):
                        feature_df = extract_url_features(result, models['feature_names'])
                        predictions, multiclass_proba = make_predictions(
                            feature_df, models, binary_model_choice
                        )
                        
                        if predictions is not None:
                            st.markdown("---")
                            st.subheader("🎯 Threat Analysis")
                            
                            res_col1, res_col2, res_col3 = st.columns(3)
                            with res_col1:
                                st.metric("Classification", predictions['Binary_Prediction'].iloc[0])
                            with res_col2:
                                st.metric("Risk Level", 
                                    f"{predictions['Risk_Icon'].iloc[0]} {predictions['Risk_Level'].iloc[0]}")
                            with res_col3:
                                st.metric("Attack Type", predictions['Predicted_Attack_Type'].iloc[0])
                            
                            st.plotly_chart(
                                create_probability_gauge(predictions['Attack_Probability'].iloc[0]),
                                use_container_width=True
                            )
                            
                            st.plotly_chart(
                                create_attack_type_chart(multiclass_proba[0], models['label_mapping']),
                                use_container_width=True
                            )
    
    # TAB 2: CSV BATCH PROCESSING
    with tab2:
        st.header("Batch Prediction from CSV")
        
        uploaded_file = st.file_uploader("Choose CSV file", type=['csv'])
        
        if uploaded_file is not None:
            try:
                df_input = pd.read_csv(uploaded_file)
                st.success(f"✅ Loaded {len(df_input)} records")
                
                st.dataframe(df_input.head(10), use_container_width=True)
                
                missing_features = set(models['feature_names']) - set(df_input.columns)
                if missing_features:
                    st.warning(f"⚠️ Missing {len(missing_features)} features - using defaults")
                    for feature in missing_features:
                        df_input[feature] = 0.0
                
                df_input = df_input[models['feature_names']]
                
                if st.button("🚀 Run Prediction"):
                    with st.spinner("Processing..."):
                        predictions, multiclass_proba = make_predictions(
                            df_input, models, binary_model_choice
                        )
                        
                        if predictions is not None:
                            result_df = pd.concat([df_input.reset_index(drop=True), predictions], axis=1)
                            
                            st.success("✅ Complete!")
                            
                            sum_col1, sum_col2, sum_col3 = st.columns(3)
                            with sum_col1:
                                st.metric("Attacks", (predictions['Binary_Prediction'] == 'Attack').sum())
                            with sum_col2:
                                st.metric("Avg Probability", f"{predictions['Attack_Probability'].mean():.2%}")
                            with sum_col3:
                                st.metric("High Risk", (predictions['Risk_Level'] == 'High').sum())
                            
                            risk_counts = predictions['Risk_Level'].value_counts()
                            fig_risk = px.pie(
                                values=risk_counts.values,
                                names=risk_counts.index,
                                title="Risk Distribution",
                                color=risk_counts.index,
                                color_discrete_map={'Low': 'green', 'Medium': 'yellow', 'High': 'red'}
                            )
                            st.plotly_chart(fig_risk, use_container_width=True)
                            
                            st.dataframe(result_df, use_container_width=True)
                            
                            csv_buffer = BytesIO()
                            result_df.to_csv(csv_buffer, index=False)
                            csv_buffer.seek(0)
                            
                            st.download_button(
                                "📥 Download Results",
                                data=csv_buffer,
                                file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    # TAB 3: MANUAL ENTRY
    with tab3:
        st.header("Manual Feature Entry")
        
        st.info("💡 Enter values for key features")
        
        EXPECTED_FEATURES = [
            'Dst Port', 'Protocol', 'Flow Duration', 'Tot Fwd Pkts', 'Tot Bwd Pkts',
            'TotLen Fwd Pkts', 'TotLen Bwd Pkts', 'Fwd Pkt Len Mean', 'Bwd Pkt Len Mean',
            'Flow Byts/s', 'Flow Pkts/s', 'Flow IAT Mean', 'Fwd IAT Mean', 'Bwd IAT Mean',
            'Pkt Len Mean', 'Pkt Size Avg', 'Init Fwd Win Byts', 'Init Bwd Win Byts'
        ]
        
        with st.form("manual_form"):
            col1, col2 = st.columns(2)
            
            manual_features = {}
            features_to_show = [f for f in EXPECTED_FEATURES if f in models['feature_names']][:18]
            
            for i, feature in enumerate(features_to_show):
                with col1 if i % 2 == 0 else col2:
                    manual_features[feature] = st.number_input(
                        feature, value=0.0, format="%.6f", key=f"m_{feature}"
                    )
            
            submitted = st.form_submit_button("🎯 Predict", use_container_width=True)
            
            if submitted:
                feature_dict = {f: 0.0 for f in models['feature_names']}
                feature_dict.update(manual_features)
                df_manual = pd.DataFrame([feature_dict])
                
                with st.spinner("Predicting..."):
                    predictions, multiclass_proba = make_predictions(
                        df_manual, models, binary_model_choice
                    )
                    
                    if predictions is not None:
                        st.markdown("---")
                        st.subheader("🎯 Results")
                        
                        res_col1, res_col2 = st.columns(2)
                        with res_col1:
                            st.metric("Prediction", predictions['Binary_Prediction'].iloc[0])
                            st.metric("Risk", 
                                f"{predictions['Risk_Icon'].iloc[0]} {predictions['Risk_Level'].iloc[0]}")
                        with res_col2:
                            st.metric("Attack Type", predictions['Predicted_Attack_Type'].iloc[0])
                            st.metric("Confidence", f"{predictions['Attack_Type_Confidence'].iloc[0]:.2%}")
                        
                        st.plotly_chart(
                            create_probability_gauge(predictions['Attack_Probability'].iloc[0]),
                            use_container_width=True
                        )
    
    # TAB 4: VISUALIZATIONS
    with tab4:
        st.header("Model Performance Visualizations")
        
        reports_path = get_file_path(DEFAULT_BASE_PATH, "reports")
        viz_path = get_file_path(DEFAULT_BASE_PATH, "visualizations")
        
        viz_model = st.selectbox("Select Model:", ["Random Forest", "Logistic Regression"])
        
        try:
            eval_file = get_file_path(reports_path, "evaluation_results.json")
            
            if check_path_exists(eval_file):
                with open(eval_file, 'r') as f:
                    eval_results = json.load(f)
                
                st.subheader(f"📈 {viz_model} Metrics")
                
                # Try different key patterns
                model_key = "Random_Forest" if viz_model == "Random Forest" else "Logistic_Regression"
                metrics = None
                
                if model_key in eval_results:
                    metrics = eval_results[model_key]
                elif 'binary_classification' in eval_results:
                    bc = eval_results['binary_classification']
                    if isinstance(bc, dict) and model_key in bc:
                        metrics = bc[model_key]
                
                if metrics and isinstance(metrics, dict):
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        acc = metrics.get('accuracy', metrics.get('Accuracy', 0))
                        st.metric("Accuracy", f"{float(acc):.4f}" if acc else "N/A")
                    with col2:
                        prec = metrics.get('precision', metrics.get('Precision', 0))
                        st.metric("Precision", f"{float(prec):.4f}" if prec else "N/A")
                    with col3:
                        rec = metrics.get('recall', metrics.get('Recall', 0))
                        st.metric("Recall", f"{float(rec):.4f}" if rec else "N/A")
                    with col4:
                        f1 = metrics.get('f1_score', metrics.get('f1', metrics.get('F1', 0)))
                        st.metric("F1 Score", f"{float(f1):.4f}" if f1 else "N/A")
            
            st.markdown("---")
            st.subheader("📊 Confusion Matrices")
            
            viz_col1, viz_col2 = st.columns(2)
            with viz_col1:
                binary_cm = get_file_path(viz_path, "01_confusion_matrix_binary.png")
                if check_path_exists(binary_cm):
                    st.image(binary_cm, use_column_width=True)
            with viz_col2:
                multi_cm = get_file_path(viz_path, "02_confusion_matrix_multiclass.png")
                if check_path_exists(multi_cm):
                    st.image(multi_cm, use_column_width=True)
            
            st.markdown("---")
            st.subheader("📈 Performance Curves")
            
            roc_col1, roc_col2 = st.columns(2)
            with roc_col1:
                roc = get_file_path(viz_path, "03_roc_curve.png")
                if check_path_exists(roc):
                    st.image(roc, use_column_width=True)
            with roc_col2:
                prob = get_file_path(viz_path, "04_probability_distribution.png")
                if check_path_exists(prob):
                    st.image(prob, use_column_width=True)
            
        except Exception as e:
            st.error(f"❌ Error loading visualizations: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p>🛡️ Cyberattack Forecasting System | CICIDS 2018 Dataset</p>
        <p style='font-size: 0.8em; color: gray;'>4.8M+ samples analyzed</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
