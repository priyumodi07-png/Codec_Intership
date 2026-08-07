import streamlit as st
import os
import json
import plotly.graph_objects as go
import plotly.express as px
from prediction import predict_news
from PIL import Image

# 1. Page Configuration
st.set_page_config(
    page_title="AI Fake News Detection",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Premium Custom CSS
st.markdown("""
    <style>
    /* Main Backgrounds */
    .stApp {
        background-color: #f8f9fa;
    }
    .stApp[data-theme="dark"] {
        background-color: #0d1117;
    }
    
    /* Typography & Headers */
    .main-title {
        text-align: center;
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        font-size: 2.5rem;
        background: -webkit-linear-gradient(45deg, #2196F3, #4CAF50);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 10px;
        padding-top: 20px;
    }
    
    .subtitle {
        text-align: center;
        font-size: 1.1rem;
        color: #6c757d;
        margin-bottom: 40px;
    }

    /* Result Boxes (Glassmorphism inspired) */
    .result-box-real {
        padding: 30px;
        border-radius: 16px;
        background: linear-gradient(135deg, rgba(209, 231, 221, 0.9), rgba(163, 207, 187, 0.9));
        color: #0f5132;
        text-align: center;
        font-size: 32px;
        font-weight: 800;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.1);
        backdrop-filter: blur(4px);
        border: 1px solid rgba(255, 255, 255, 0.18);
        transition: transform 0.3s ease;
    }
    .result-box-real:hover { transform: translateY(-5px); }
    
    .result-box-fake {
        padding: 30px;
        border-radius: 16px;
        background: linear-gradient(135deg, rgba(248, 215, 218, 0.9), rgba(241, 176, 183, 0.9));
        color: #842029;
        text-align: center;
        font-size: 32px;
        font-weight: 800;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.1);
        backdrop-filter: blur(4px);
        border: 1px solid rgba(255, 255, 255, 0.18);
        transition: transform 0.3s ease;
    }
    .result-box-fake:hover { transform: translateY(-5px); }
    
    /* Dark Mode Adjustments for Result Boxes */
    .stApp[data-theme="dark"] .result-box-real {
        background: linear-gradient(135deg, rgba(15, 81, 50, 0.8), rgba(20, 108, 67, 0.8));
        color: #d1e7dd;
    }
    .stApp[data-theme="dark"] .result-box-fake {
        background: linear-gradient(135deg, rgba(132, 32, 41, 0.8), rgba(176, 42, 55, 0.8));
        color: #f8d7da;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 20px;
        margin-top: 50px;
        border-top: 1px solid #e9ecef;
        color: #6c757d;
        font-size: 0.9rem;
    }
    .stApp[data-theme="dark"] .footer {
        border-top: 1px solid #30363d;
        color: #8b949e;
    }
    </style>
""", unsafe_allow_html=True)

# 3. Helper Functions
def load_metrics():
    metrics_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'metrics.json')
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            return json.load(f)
    return None

def create_gauge_chart(confidence, prediction):
    # Dynamic coloring based on prediction
    color = "#28a745" if prediction == "REAL" else "#dc3545"
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = confidence,
        number = {'suffix': "%", 'font': {'size': 40}},
        title = {'text': "Model Confidence", 'font': {'size': 18}},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': color, 'thickness': 0.75},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': "rgba(128, 128, 128, 0.1)"},
                {'range': [50, 80], 'color': "rgba(128, 128, 128, 0.2)"},
                {'range': [80, 100], 'color': "rgba(128, 128, 128, 0.3)"}
            ],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': confidence
            }
        }
    ))
    # Make transparent background for dark mode compatibility
    fig.update_layout(
        height=300, 
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={'color': "#888"}
    )
    return fig

# Pre-loaded sample articles for demo
SAMPLE_REAL = """The World Health Organization has announced a new initiative to combat the spread of infectious diseases in developing nations. The program will provide $50 million in funding over the next five years to improve local healthcare infrastructure and train medical professionals. The announcement was made during the annual global health summit in Geneva."""
SAMPLE_FAKE = """Breaking: Scientists have just discovered that the moon is actually made entirely of cheese! A secret mission by a rogue billionaire brought back a massive chunk of cheddar. Experts are currently arguing over whether it's mild or sharp. The government has been hiding this from us since the 1960s!"""

def main():
    # Sidebar Navigation & Demo News
    st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2965/2965306.png", width=80)
    st.sidebar.title("AI Navigation")
    page = st.sidebar.radio("Menu", ["🔍 Fake News Predictor", "📊 Model Analytics Dashboard", "ℹ️ Project Details"])

    st.sidebar.markdown("---")
    st.sidebar.subheader("🧪 Test Data (Demo)")
    st.sidebar.markdown("Click below to copy a sample article:")
    
    with st.sidebar.expander("Sample Real News"):
        st.code(SAMPLE_REAL, language="text")
        
    with st.sidebar.expander("Sample Fake News"):
        st.code(SAMPLE_FAKE, language="text")

    st.sidebar.markdown("---")
    st.sidebar.caption("© 2026 AI Fake News Detection System")

    # Main Content Area
    if page == "🔍 Fake News Predictor":
        # Load Banner Image
        banner_path = os.path.join(os.path.dirname(__file__), '..', 'static', 'banner.png')
        if os.path.exists(banner_path):
            img = Image.open(banner_path)
            st.image(img, use_container_width=True)

            
        st.markdown("<h1 class='main-title'>AI Fake News Detection System</h1>", unsafe_allow_html=True)
        st.markdown("<p class='subtitle'>Analyze news articles instantly using advanced NLP and Machine Learning.</p>", unsafe_allow_html=True)

        # Input Section
        with st.container():
            news_input = st.text_area(
                "Paste the news article or headline here:", 
                height=200, 
                placeholder="Enter text to verify its authenticity..."
            )
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                analyze_btn = st.button("Analyze Authenticity 🔍", type="primary", use_container_width=True)

        if analyze_btn:
            if not news_input or len(news_input.strip()) < 10:
                st.toast("⚠️ Please enter a longer text block for accurate analysis.", icon="⚠️")
                st.warning("Input is too short! Machine learning models require sufficient context (at least a full sentence or two) to make accurate predictions.")
            else:
                with st.spinner("Analyzing linguistic patterns and cross-referencing features..."):
                    result = predict_news(news_input)

                    if "error" in result:
                        st.error(f"Prediction Error: {result['error']}")
                    else:
                        st.toast("Analysis Complete!", icon="✅")
                        st.markdown("### 📊 Analysis Results")
                        
                        prediction = result["prediction"]
                        confidence = result.get("confidence")

                        res_col1, res_col2 = st.columns([1, 1], gap="large")
                        
                        with res_col1:
                            st.write("") # Spacing
                            st.write("") 
                            if prediction == "REAL":
                                st.markdown(f"<div class='result-box-real'>🛡️ THIS NEWS IS REAL</div>", unsafe_allow_html=True)
                                st.success("The model identified linguistic patterns consistent with legitimate journalism.")
                            else:
                                st.markdown(f"<div class='result-box-fake'>🚨 THIS NEWS IS FAKE</div>", unsafe_allow_html=True)
                                st.error("The model identified linguistic patterns commonly found in fabricated or deceptive content.")
                                
                        with res_col2:
                            if confidence:
                                fig = create_gauge_chart(confidence, prediction)
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.info("Confidence score not available.")
                                
                        with st.expander("Show Internal NLP Processing Details"):
                            st.write("**Cleaned & Tokenized Text fed to the model:**")
                            st.code(result.get("cleaned_text", ""), language="text")

    elif page == "📊 Model Analytics Dashboard":
        st.title("📊 Model Analytics Dashboard")
        st.markdown("Compare the performance of various machine learning algorithms trained on over 6,000 real and fake news articles.")
        
        metrics = load_metrics()
        
        if not metrics:
            st.error("No model metrics found. Please execute the training pipeline first.")
            return
            
        best_model_name = metrics.get('best_model', 'Unknown')
        st.info(f"🏆 **System Selected Best Model:** `{best_model_name}` (Currently Active for Predictions)")
        
        st.markdown("---")
        
        # Prepare data for plotting
        model_names = [m for m in metrics.keys() if m != 'best_model']
        acc_data = [{'Model': m, 'Score': metrics[m]['accuracy'], 'Metric': 'Accuracy'} for m in model_names]
        f1_data = [{'Model': m, 'Score': metrics[m]['f1_score'], 'Metric': 'F1-Score'} for m in model_names]
            
        # Plotly Bar Charts
        fig_acc = px.bar(acc_data, x='Model', y='Score', color='Model', 
                         title="Accuracy Comparison", text_auto='.2%', 
                         color_discrete_sequence=px.colors.qualitative.Pastel)
        fig_acc.update_layout(yaxis_range=[0,1.1], paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        
        fig_f1 = px.bar(f1_data, x='Model', y='Score', color='Model', 
                        title="F1-Score Comparison", text_auto='.2%',
                        color_discrete_sequence=px.colors.qualitative.Set2)
        fig_f1.update_layout(yaxis_range=[0,1.1], paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(fig_acc, use_container_width=True)
        with c2:
            st.plotly_chart(fig_f1, use_container_width=True)
            
        st.markdown("### Deep Dive: Confusion Matrices")
        st.caption("Visualizing True Positives vs False Positives")
        
        cm_cols = st.columns(len(model_names))
        
        for idx, m in enumerate(model_names):
            with cm_cols[idx]:
                st.markdown(f"**{m}**")
                cm = metrics[m].get('confusion_matrix')
                if cm:
                    cm_fig = px.imshow(cm, text_auto=True, color_continuous_scale='RdBu_r',
                                     labels=dict(x="Predicted", y="Actual", color="Count"),
                                     x=['FAKE', 'REAL'], y=['FAKE', 'REAL'])
                    cm_fig.update_layout(margin=dict(l=20, r=20, t=20, b=20), height=250)
                    st.plotly_chart(cm_fig, use_container_width=True)

    elif page == "ℹ️ Project Details":
        st.title("ℹ️ Project Architecture")
        
        st.markdown("""
        ### Fake News Detection System (v3.0 - Premium Edition)
        
        This application represents a comprehensive, end-to-end Machine Learning pipeline designed to combat misinformation.
        
        #### Core Technologies
        - **Frontend**: `Streamlit`, `Plotly`, Custom CSS
        - **Backend Logic**: `Python 3`
        - **Machine Learning**: `Scikit-Learn` (GridSearchCV, Logistic Regression, Naive Bayes, Random Forest)
        - **NLP Processing**: `NLTK` (Lemmatization, Tokenization, Stopwords)
        - **Data Handling**: `Pandas`, `NumPy`
        
        #### How it works
        1. **Data Ingestion**: Learns from a dataset of 6,335 verified real and fake news articles.
        2. **NLP Preprocessing**: Unstructured text is cleaned by stripping punctuation, making lowercase, removing stopwords, and reducing words to their root form (lemmatization).
        3. **Vectorization**: The cleaned text is converted into a matrix of TF-IDF (Term Frequency-Inverse Document Frequency) features.
        4. **Model Inference**: The highest-performing model (determined via automated Grid Search cross-validation) analyzes the feature matrix to output a probability score.
        
        *Built for Placement Interviews and Portfolio Showcase.*
        """)

    # Footer
    st.markdown("<div class='footer'>Developed with ❤️ using Python and Streamlit | Fake News Detection System © 2026</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
