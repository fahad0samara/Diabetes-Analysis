"""
Analytics component for DiabetesGuard Pro.
Author: Fahad
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import sys

# Add parent directory to Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from app.utils.data_loader import load_data, get_feature_descriptions, get_feature_ranges, get_risk_factors

def create_distribution_plot(df, column, title):
    """Create a distribution plot for a numerical column."""
    fig = px.histogram(df, x=column, title=title)
    fig.update_layout(showlegend=False)
    return fig

def create_correlation_plot(df, title):
    """Create a correlation heatmap."""
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    corr = df[numeric_cols].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr,
        x=corr.columns,
        y=corr.columns,
        colorscale='RdBu',
        zmin=-1,
        zmax=1
    ))
    
    fig.update_layout(
        title=title,
        height=600
    )
    
    return fig

def render_analytics():
    """Render the analytics page."""
    st.title("📊 Health Analytics Dashboard")
    
    try:
        # Load data
        df = load_data()
        feature_desc = get_feature_descriptions()
        feature_ranges = get_feature_ranges()
        risk_factors = get_risk_factors()
        
        # Dashboard sections
        st.markdown("## 📈 Data Overview")
        
        # Display basic statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", len(df))
        with col2:
            diabetes_rate = (df['Diabetes_Diagnosis'].mean() * 100).round(1)
            st.metric("Diabetes Rate", f"{diabetes_rate}%")
        with col3:
            avg_age = df['Age'].mean().round(1)
            st.metric("Average Age", f"{avg_age} years")
        
        # Feature Distributions
        st.markdown("## 📊 Feature Distributions")
        
        # Let user select a feature to visualize
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        selected_feature = st.selectbox(
            "Select a feature to visualize:",
            numeric_cols,
            format_func=lambda x: f"{x} - {feature_desc.get(x, '')}"
        )
        
        # Create and display distribution plot
        fig = create_distribution_plot(
            df, 
            selected_feature,
            f"Distribution of {selected_feature}"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Display normal range if available
        if selected_feature in feature_ranges:
            range_info = feature_ranges[selected_feature]
            st.info(f"Normal range for {selected_feature}: {range_info['normal']}")
        
        # Correlation Analysis
        st.markdown("## 🔄 Feature Correlations")
        correlation_fig = create_correlation_plot(df, "Feature Correlation Heatmap")
        st.plotly_chart(correlation_fig, use_container_width=True)
        
        # Risk Factor Analysis
        st.markdown("## ⚠️ Risk Factor Analysis")
        
        for factor in risk_factors:
            with st.expander(f"📌 {factor['factor']}"):
                st.markdown(f"**Description**: {factor['description']}")
                st.markdown(f"**Recommendation**: {factor['recommendation']}")
        
        # Data Quality
        st.markdown("## 🔍 Data Quality")
        
        # Calculate missing values
        missing_data = df.isnull().sum()
        if missing_data.any():
            st.warning("Some features have missing values:")
            for col in missing_data[missing_data > 0].index:
                st.write(f"- {col}: {missing_data[col]} missing values")
        else:
            st.success("No missing values found in the dataset!")
        
        # Add data timestamp
        st.markdown("---")
        st.caption("Data last updated: 2025-02-05")
        
    except Exception as e:
        st.error(f"An error occurred while loading the analytics: {str(e)}")
        st.markdown("""
        Please make sure:
        1. The dataset file exists in the correct location
        2. You have the required permissions to access the file
        3. The file is not corrupted
        """)

def render_footer():
    """Render the footer section."""
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; padding: 1rem;'>
        Made with ❤️ by Fahad
        </div>
        """,
        unsafe_allow_html=True
    )
