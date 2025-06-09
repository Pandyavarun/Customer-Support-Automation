import streamlit as st
import pandas as pd
import numpy as np
import os
import time
from datetime import datetime
import plotly.express as px
from processing.file_processor import LargeFileProcessor
from ml.model_builder import ModelBuilder
from ml.response_templates import ResponseTemplates
#from database.operations import DatabaseOperations
from config import config

# Set page config
st.set_page_config(
    page_title="Customer Support Automation",
    page_icon="🎧",
    layout="wide"
)

# Initialize components
if 'processor' not in st.session_state:
    st.session_state.processor = LargeFileProcessor()
if 'model_builder' not in st.session_state:
    st.session_state.model_builder = ModelBuilder()
if 'response_templates' not in st.session_state:
    st.session_state.response_templates = ResponseTemplates()
#if 'db_operations' not in st.session_state:
#    st.session_state.db_operations = DatabaseOperations()



# Main app
def main():
    st.title("🎧 Customer Support Automation System")
    st.markdown("---")
    
    # Sidebar navigation
    page = st.sidebar.selectbox("Navigation", [
        "Data Processing",
        "Model Training",
        "Message Analysis",
        "Analytics Dashboard"
    ])
    
    if page == "Data Processing":
        show_data_processing_page()
    elif page == "Model Training":
        show_model_training_page()
    elif page == "Message Analysis":
        show_message_analysis_page()
    elif page == "Analytics Dashboard":
        show_analytics_page()

def show_data_processing_page():
    st.header("📂 Data Processing")
    
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    
    if uploaded_file is not None:
        # Save uploaded file
        file_path = os.path.join(config.DATA_PATH, uploaded_file.name)
        with open(file_path, "wb") as f:
            uploaded_file.seek(0)
            f.write(uploaded_file.read())
        
        st.success(f"File saved: {file_path}")
        
        if st.button("Process Data"):
            with st.spinner("Processing data..."):
                try:
                    def progress_callback(chunk_num, chunk_size, total_rows):
                        progress = min(100, int((chunk_num * chunk_size) / total_rows * 100))
                        st.write(f"Processed {progress}%...")
                    
                    data = st.session_state.processor.process_large_csv(
                        file_path,
                        progress_callback
                    )
                    
                    st.session_state.processed_data = data
                    st.success(f"Successfully processed {len(data)} records!")
                    
                    # Show sample data
                    st.subheader("Sample Data")
                    st.dataframe(data.head())
                    
                except Exception as e:
                    st.error(f"Error processing file: {e}")

def show_model_training_page():
    st.header("🤖 Model Training")
    
    if 'processed_data' not in st.session_state:
        st.warning("Please process data first in the Data Processing page")
        return
    
    if st.button("Train Models"):
        with st.spinner("Training models..."):
            try:
                # Prepare features
                data = st.session_state.processed_data
                vectorizer = TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1, 2))
                text_features = vectorizer.fit_transform(data['clean_text'])
                
                # Create labels
                response_categories = []
                for text in data['clean_text']:
                    if any(word in text.lower() for word in ['private message', 'secure', 'details']):
                        response_categories.append('escalation')
                    elif any(word in text.lower() for word in ['understand', 'help', 'assist']):
                        response_categories.append('acknowledgment')
                    elif any(word in text.lower() for word in ['click', 'please', 'follow']):
                        response_categories.append('instruction')
                    else:
                        response_categories.append('general')
                
                # Train models
                st.session_state.model_builder.vectorizers['response'] = vectorizer
                st.session_state.model_builder.build_response_classifier(text_features, response_categories)
                st.session_state.model_builder.save_models()
                
                st.success("Models trained and saved successfully!")
                
            except Exception as e:
                st.error(f"Error training models: {e}")

def show_message_analysis_page():
    st.header("💬 Message Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Input Message")
        message = st.text_area("Customer Message", height=150)
        author_id = st.text_input("Customer ID")
        analyze_btn = st.button("Analyze Message", type="primary")
    
    with col2:
        st.subheader("Analysis Results")
        
        if analyze_btn and message:
            try:
                # Load models if not loaded
                if not st.session_state.model_builder.models:
                    st.session_state.model_builder.load_models()
                
                # Clean message
                clean_message = st.session_state.processor._clean_text(message)
                
                # Get predictions
                text_vector = st.session_state.model_builder.vectorizers['response'].transform([clean_message])
                category = st.session_state.model_builder.models['response_classifier'].predict(text_vector)[0]
                
                # Calculate urgency and frustration
                urgency_score = st.session_state.processor._calculate_urgency_score(clean_message)
                frustration_score = st.session_state.processor._calculate_frustration_score(clean_message)
                
                # Get response template
                response = st.session_state.response_templates.get_template(
                    category,
                    urgency_score,
                    frustration_score
                )
                
                # Display results
                st.success("Analysis complete!")
                
                cols = st.columns(2)
                with cols[0]:
                    st.metric("Category", category.title())
                    st.metric("Urgency Score", f"{urgency_score:.1f}")
                
                with cols[1]:
                    st.metric("Frustration", f"{frustration_score:.1f}")
                    st.metric("Priority", "High" if urgency_score > 5 else "Medium" if urgency_score > 2 else "Low")
                
                st.subheader("Recommended Response")
                st.info(response)
                
                if urgency_score > 5 or frustration_score > 4:
                    st.warning("⚠️ This message should be escalated!")
                
            except Exception as e:
                st.error(f"Error analyzing message: {e}")

def show_analytics_page():
    st.header("📊 Analytics Dashboard")
    
    if 'processed_data' not in st.session_state:
        st.warning("No data available for analytics")
        return
    
    data = st.session_state.processed_data
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Messages", len(data))
    with col2:
        st.metric("Unique Customers", data['author_id'].nunique())
    with col3:
        st.metric("Avg Urgency", f"{data['urgency_score'].mean():.1f}")
    with col4:
        st.metric("Avg Sentiment", f"{data['sentiment_score'].mean():.1f}")
    
    # Charts
    st.subheader("Message Categories")
    category_counts = data['response_category'].value_counts()
    fig1 = px.pie(category_counts, values=category_counts.values, names=category_counts.index)
    st.plotly_chart(fig1, use_container_width=True)
    
    st.subheader("Urgency Distribution")
    fig2 = px.histogram(data, x='urgency_score', nbins=10)
    st.plotly_chart(fig2, use_container_width=True)

def show_template_management_page():
    st.header("📝 Response Template Management")
    
    tab1, tab2 = st.tabs(["View Templates", "Add New Template"])
    
    with tab1:
        st.subheader("Current Templates")
        all_templates = st.session_state.response_templates.get_all_templates()
        
        st.write("#### Default Templates")
        st.json(all_templates['default'])
        
        st.write("#### Custom Templates")
        st.json(all_templates['custom'])
    
    with tab2:
        st.subheader("Add Custom Template")
        
        category = st.selectbox("Category", [
            'escalation', 'acknowledgment', 
            'instruction', 'followup', 'general'
        ])
        
        template_text = st.text_area("Template Text", height=100)
        
        col1, col2 = st.columns(2)
        with col1:
            urgency_threshold = st.number_input(
                "Minimum Urgency Score (optional)", 
                min_value=0, max_value=10, value=0
            )
        with col2:
            frustration_threshold = st.number_input(
                "Minimum Frustration Score (optional)", 
                min_value=0, max_value=10, value=0
            )
        
        if st.button("Add Template"):
            if template_text:
                st.session_state.response_templates.add_custom_template(
                    category=category,
                    template=template_text,
                    urgency_threshold=urgency_threshold if urgency_threshold > 0 else None,
                    frustration_threshold=frustration_threshold if frustration_threshold > 0 else None
                )
                
                # Save templates
                templates_file = os.path.join(config.MODEL_PATH, "custom_templates.json")
                st.session_state.response_templates.save_templates(templates_file)
                
                st.success("Template added successfully!")
            else:
                st.warning("Please enter template text")

if __name__ == "__main__":
    main()