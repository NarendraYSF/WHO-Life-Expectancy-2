# Prevent PyArrow issues by setting environment variables before imports
import os
os.environ['PANDAS_USE_PYARROW'] = 'false'
os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
# Disable PyArrow completely
os.environ['ARROW_DISABLE'] = 'true'

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from datetime import datetime
import warnings

# Suppress PyArrow warnings and errors
warnings.filterwarnings('ignore', category=UserWarning, module='pyarrow')
warnings.filterwarnings('ignore', category=FutureWarning, module='pyarrow')

# Import custom modules
from data_preprocessing import DataPreprocessor
from model_training import LifeExpectancyModel
from visualization import LifeExpectancyVisualizer
from outlier_detection import OutlierDetectionSystem

# Import sklearn modules for linear regression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Import mapping libraries
import folium
from streamlit_folium import folium_static
from folium.plugins import MarkerCluster, HeatMap

# Configure pandas to handle problematic data types better
pd.options.mode.use_inf_as_na = True

# Disable PyArrow in pandas completely
try:
    import pyarrow
    # This will prevent pandas from using PyArrow
    pd.options.io.parquet.engine = 'fastparquet'
except ImportError:
    pass

# Page configuration
st.set_page_config(
    page_title="Life Expectancy Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add a note about PyArrow bypass
st.markdown("""
<div style="background-color: #e8f4fd; padding: 10px; border-radius: 5px; margin-bottom: 20px;">
    <strong>Note:</strong> This application uses alternative display methods to avoid PyArrow compatibility issues. 
    Data is displayed using HTML tables for better compatibility. Any PyArrow-related errors in the console can be safely ignored.
</div>
""", unsafe_allow_html=True)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .sidebar-header {
        font-size: 1.5rem;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .outlier-info {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and cache the dataset."""
    preprocessor = DataPreprocessor()
    data_file = "Dataset/WHO Life Expectancy Descriptive Statistics - Raw Data.csv"
    
    if os.path.exists(data_file):
        try:
            df = preprocessor.load_and_clean_data(data_file)
            
            # Additional sanitization to prevent PyArrow issues
            df_clean = df.copy()
            
            # Handle any remaining problematic data types
            for col in df_clean.columns:
                try:
                    # Convert any problematic numeric columns to float, handling errors
                    if df_clean[col].dtype in ['object', 'string']:
                        # Try to convert to numeric first
                        numeric_series = pd.to_numeric(df_clean[col], errors='coerce')
                        if not numeric_series.isna().all():  # If conversion was successful
                            df_clean[col] = numeric_series
                        else:
                            # Keep as string but clean it
                            df_clean[col] = df_clean[col].astype(str)
                            df_clean[col] = df_clean[col].replace(['nan', 'None', 'NULL', ''], 'N/A')
                    elif df_clean[col].dtype in ['float64', 'int64']:
                        # Handle infinite values
                        df_clean[col] = df_clean[col].replace([np.inf, -np.inf], np.nan)
                        # Ensure proper numeric type
                        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                except Exception as e:
                    # If conversion fails, convert to string
                    df_clean[col] = df_clean[col].astype(str)
                    df_clean[col] = df_clean[col].replace(['nan', 'None', 'NULL', ''], 'N/A')
            
            return df_clean, preprocessor
            
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return None, None
    else:
        st.error(f"Data file not found at {data_file}")
        return None, None

@st.cache_resource
def train_models(X, y, feature_names):
    """Train and cache models."""
    model_trainer = LifeExpectancyModel()
    results = model_trainer.train_models(X, y, feature_names)
    return results, model_trainer

def show_detailed_outlier_info(df, outlier_analysis):
    """Show detailed information about detected outliers."""
    
    st.subheader("📋 Detailed Outlier Information")
    
    # Create tabs for different types of outlier information
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Target Outliers", "📊 Feature Outliers", "🌐 Multivariate Outliers", "📈 Impact Analysis"])
    
    with tab1:
        st.write("**Life Expectancy Outliers by Detection Method**")
        
        for method, results in outlier_analysis['target_outliers'].items():
            with st.expander(f"{method.upper()} - {results['outlier_count']} outliers ({results['outlier_percentage']:.2f}%)"):
                if results['outlier_count'] > 0:
                    # Get outlier data
                    target_data = df['Life expectancy'].dropna()
                    outlier_indices = results['outlier_indices']
                    outlier_values = target_data.iloc[outlier_indices]
                    outlier_scores = results['outlier_scores'][outlier_indices]
                    
                    # Create detailed outlier table
                    outlier_details = []
                    for i, (idx, value, score) in enumerate(zip(outlier_indices, outlier_values, outlier_scores)):
                        # Find original row
                        original_idx = df.index[df['Life expectancy'] == value].tolist()
                        if original_idx:
                            row = df.loc[original_idx[0]]
                            outlier_details.append({
                                'Index': i + 1,
                                'Country': row['Country'],
                                'Year': row['Year'],
                                'Status': row['Status'],
                                'Region': row['Region'],
                                'Life Expectancy': f"{value:.1f}",
                                'Outlier Score': f"{score:.3f}",
                                'Original Index': original_idx[0]
                            })
                    
                    if outlier_details:
                        outlier_df = pd.DataFrame(outlier_details)
                        safe_display_dataframe(outlier_df, use_container_width=True)
                        
                        # Show statistics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Min Outlier", f"{outlier_values.min():.1f}")
                        with col2:
                            st.metric("Max Outlier", f"{outlier_values.max():.1f}")
                        with col3:
                            st.metric("Mean Outlier", f"{outlier_values.mean():.1f}")
                        
                        # Show outlier distribution
                        fig = px.histogram(outlier_values, title=f"Distribution of {method} Outliers")
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No outliers detected with this method.")
    
    with tab2:
        st.write("**Feature-wise Outlier Analysis**")
        
        # Select feature to analyze
        feature_options = list(outlier_analysis['feature_outliers'].keys())
        selected_feature = st.selectbox("Select a feature:", feature_options)
        
        if selected_feature:
            feature_methods = outlier_analysis['feature_outliers'][selected_feature]
            
            for method, results in feature_methods.items():
                with st.expander(f"{selected_feature} - {method.upper()}"):
                    if results['outlier_count'] > 0:
                        # Get feature data
                        feature_data = df[selected_feature].dropna()
                        outlier_indices = results['outlier_indices']
                        outlier_values = feature_data.iloc[outlier_indices]
                        
                        st.write(f"**Outliers detected:** {results['outlier_count']} ({results['outlier_percentage']:.2f}%)")
                        
                        # Show outlier statistics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Outlier Range", f"{outlier_values.min():.3f} - {outlier_values.max():.3f}")
                        with col2:
                            st.metric("Overall Range", f"{feature_data.min():.3f} - {feature_data.max():.3f}")
                        with col3:
                            st.metric("Mean Outlier", f"{outlier_values.mean():.3f}")
                        
                        # Show outlier distribution
                        fig = px.histogram(outlier_values, title=f"Distribution of {selected_feature} Outliers")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No outliers detected for this feature.")
    
    with tab3:
        st.write("**Multivariate Outlier Analysis**")
        
        for method, results in outlier_analysis['multivariate_outliers'].items():
            with st.expander(f"{method.upper()} - {results['outlier_count']} outliers ({results['outlier_percentage']:.2f}%)"):
                if results['outlier_count'] > 0:
                    # Get outlier indices
                    outlier_indices = results['outlier_indices']
                    
                    # Create detailed table
                    multivariate_details = []
                    for i, idx in enumerate(outlier_indices[:20]):  # Show first 20
                        row = df.iloc[idx]
                        multivariate_details.append({
                            'Index': i + 1,
                            'Country': row['Country'],
                            'Year': row['Year'],
                            'Status': row['Status'],
                            'Region': row['Region'],
                            'Life Expectancy': f"{row['Life expectancy']:.1f}",
                            'Outlier Score': f"{results['outlier_scores'][idx]:.3f}",
                            'Original Index': idx
                        })
                    
                    if multivariate_details:
                        multivariate_df = pd.DataFrame(multivariate_details)
                        safe_display_dataframe(multivariate_df, use_container_width=True)
                        
                        if len(outlier_indices) > 20:
                            st.info(f"... and {len(outlier_indices) - 20} more outliers")
                        
                        # Show 2D scatter plot of first two numerical features
                        numerical_cols = df.select_dtypes(include=[np.number]).columns[:2]
                        if len(numerical_cols) >= 2:
                            fig = px.scatter(
                                df, 
                                x=numerical_cols[0], 
                                y=numerical_cols[1],
                                title=f"Multivariate Outliers: {numerical_cols[0]} vs {numerical_cols[1]}"
                            )
                            
                            # Highlight outliers
                            outlier_df = df.iloc[outlier_indices]
                            fig.add_trace(go.Scatter(
                                x=outlier_df[numerical_cols[0]],
                                y=outlier_df[numerical_cols[1]],
                                mode='markers',
                                marker=dict(color='red', size=8),
                                name='Outliers'
                            ))
                            
                            st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No multivariate outliers detected.")
    
    with tab4:
        st.write("**Impact Analysis of Outlier Removal**")
        
        # Show impact for each method
        for method, results in outlier_analysis['target_outliers'].items():
            if results['outlier_count'] > 0:
                with st.expander(f"Impact of removing {method} outliers"):
                    # Calculate statistics
                    target_data = df['Life expectancy'].dropna()
                    
                    stats_before = {
                        'count': len(target_data),
                        'mean': target_data.mean(),
                        'std': target_data.std(),
                        'median': target_data.median(),
                        'min': target_data.min(),
                        'max': target_data.max()
                    }
                    
                    outlier_indices = results['outlier_indices']
                    clean_data = np.delete(target_data.values, outlier_indices)
                    
                    stats_after = {
                        'count': len(clean_data),
                        'mean': np.mean(clean_data),
                        'std': np.std(clean_data),
                        'median': np.median(clean_data),
                        'min': np.min(clean_data),
                        'max': np.max(clean_data)
                    }
                    
                    # Show comparison
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Records", f"{stats_before['count']}", f"{stats_after['count'] - stats_before['count']}")
                    
                    with col2:
                        mean_change = stats_after['mean'] - stats_before['mean']
                        st.metric("Mean", f"{stats_before['mean']:.2f}", f"{mean_change:+.2f}")
                    
                    with col3:
                        std_change = stats_after['std'] - stats_before['std']
                        st.metric("Std Dev", f"{stats_before['std']:.2f}", f"{std_change:+.2f}")
                    
                    # Show before/after comparison chart
                    comparison_data = pd.DataFrame({
                        'Metric': ['Mean', 'Median', 'Std Dev', 'Min', 'Max'],
                        'Before': [stats_before['mean'], stats_before['median'], stats_before['std'], 
                                 stats_before['min'], stats_before['max']],
                        'After': [stats_after['mean'], stats_after['median'], stats_after['std'], 
                                stats_after['min'], stats_after['max']]
                    })
                    
                    fig = px.bar(comparison_data, x='Metric', y=['Before', 'After'], 
                               title=f"Statistics Before vs After Removing {method} Outliers",
                               barmode='group')
                    st.plotly_chart(fig, use_container_width=True)

def sanitize_dataframe_for_display(df):
    """
    Sanitize DataFrame to prevent PyArrow conversion errors.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        
    Returns:
        pd.DataFrame: Sanitized DataFrame safe for display
    """
    try:
        df_clean = df.copy()
        
        # Handle problematic data types
        for col in df_clean.columns:
            try:
                # Check if column contains problematic values
                if df_clean[col].dtype == 'object':
                    # Convert object columns to string, handling NaN values
                    df_clean[col] = df_clean[col].astype(str)
                    # Replace problematic string representations
                    df_clean[col] = df_clean[col].replace(['nan', 'None', 'NULL', ''], 'N/A')
                elif df_clean[col].dtype == 'float64':
                    # Handle infinite values in float columns
                    df_clean[col] = df_clean[col].replace([np.inf, -np.inf], np.nan)
                    # Convert to string if there are still issues
                    if df_clean[col].isnull().any():
                        df_clean[col] = df_clean[col].astype(str)
                        df_clean[col] = df_clean[col].replace('nan', 'N/A')
                elif df_clean[col].dtype == 'int64':
                    # Handle integer columns with NaN values
                    if df_clean[col].isnull().any():
                        df_clean[col] = df_clean[col].astype(str)
                        df_clean[col] = df_clean[col].replace('nan', 'N/A')
            except Exception as e:
                # If any conversion fails, convert to string
                df_clean[col] = df_clean[col].astype(str)
                df_clean[col] = df_clean[col].replace(['nan', 'None', 'NULL', ''], 'N/A')
        
        return df_clean
    except Exception as e:
        # If all else fails, return a basic string representation
        st.error(f"Critical error in data sanitization: {e}")
        return pd.DataFrame({'Error': ['Data could not be sanitized for display']})

def alternative_display_dataframe(df, title="Data"):
    """
    Display DataFrame using alternative method when PyArrow fails completely.
    
    Args:
        df (pd.DataFrame): DataFrame to display
        title (str): Title for the display
    """
    st.write(f"**{title}**")
    st.write(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    
    # Show data types
    st.write("**Data Types:**")
    dtype_info = df.dtypes.value_counts()
    for dtype, count in dtype_info.items():
        st.write(f"- {dtype}: {count} columns")
    
    # Show first few rows as HTML table
    if len(df) > 0:
        st.write("**First 10 rows:**")
        
        # Display using Streamlit-friendly approach instead of HTML
        first_10_df = df.head(10)
        
        # Show the data using st.write which handles DataFrames properly
        st.write(first_10_df)
        
        if len(df) > 10:
            st.info(f"... and {len(df) - 10} more rows")
            
        # Show summary statistics for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            st.write("**Summary Statistics (Numeric Columns):**")
            summary_stats = df[numeric_cols].describe()
            
            # Show the summary statistics using st.write
            st.write(summary_stats)
    else:
        st.info("No data to display")

def safe_display_dataframe(df, use_container_width=True):
    """
    Safely display a DataFrame by completely bypassing PyArrow.
    
    Args:
        df (pd.DataFrame): DataFrame to display
        use_container_width (bool): Whether to use container width (ignored for alternative display)
    """
    # Always use alternative display method to completely avoid PyArrow
    alternative_display_dataframe(df, "DataFrame Contents")

def get_country_coordinates(country_name):
    """
    Get country coordinates using a mapping dictionary.
    
    Args:
        country_name (str): Country name
        
    Returns:
        tuple: (latitude, longitude) or None if not found
    """
    # Country coordinates mapping
    country_coordinates = {
        'Afghanistan': (33.9391, 67.7100),
        'Albania': (41.1533, 20.1683),
        'Algeria': (28.0339, 1.6596),
        'Andorra': (42.5063, 1.5218),
        'Angola': (-11.2027, 17.8739),
        'Antigua and Barbuda': (17.0608, -61.7964),
        'Argentina': (-38.4161, -63.6167),
        'Armenia': (40.0691, 45.0382),
        'Australia': (-25.2744, 133.7751),
        'Austria': (47.5162, 14.5501),
        'Azerbaijan': (40.1431, 47.5769),
        'Bahamas': (25.0343, -77.3963),
        'Bahrain': (26.0667, 50.5577),
        'Bangladesh': (23.6850, 90.3563),
        'Barbados': (13.1939, -59.5432),
        'Belarus': (53.7098, 27.9534),
        'Belgium': (50.8503, 4.3517),
        'Belize': (17.1899, -88.4976),
        'Benin': (9.3077, 2.3158),
        'Bhutan': (27.5142, 90.4336),
        'Bolivia': (-16.2902, -63.5887),
        'Bosnia and Herzegovina': (43.9159, 17.6791),
        'Botswana': (-22.3285, 24.6849),
        'Brazil': (-14.2350, -51.9253),
        'Brunei': (4.5353, 114.7277),
        'Bulgaria': (42.7339, 25.4858),
        'Burkina Faso': (12.2383, -1.5616),
        'Burundi': (-3.3731, 29.9189),
        'Cambodia': (12.5657, 104.9910),
        'Cameroon': (7.3697, 12.3547),
        'Canada': (56.1304, -106.3468),
        'Cape Verde': (16.5388, -23.0418),
        'Central African Republic': (6.6111, 20.9394),
        'Chad': (15.4542, 18.7322),
        'Chile': (-35.6751, -71.5430),
        'China': (35.8617, 104.1954),
        'Colombia': (4.5709, -74.2973),
        'Comoros': (-11.6455, 43.3333),
        'Congo': (-0.2280, 15.8277),
        'Costa Rica': (9.9281, -84.0907),
        'Croatia': (45.1000, 15.2000),
        'Cuba': (21.5218, -77.7812),
        'Cyprus': (35.1264, 33.4299),
        'Czech Republic': (49.8175, 15.4730),
        'Democratic Republic of the Congo': (-4.0383, 21.7587),
        'Denmark': (56.2639, 9.5018),
        'Djibouti': (11.8251, 42.5903),
        'Dominica': (15.4150, -61.3710),
        'Dominican Republic': (18.7357, -70.1627),
        'East Timor': (-8.8742, 125.7275),
        'Ecuador': (-1.8312, -78.1834),
        'Egypt': (26.8206, 30.8025),
        'El Salvador': (13.7942, -88.8965),
        'Equatorial Guinea': (1.6508, 10.2679),
        'Eritrea': (15.1794, 39.7823),
        'Estonia': (58.5953, 25.0136),
        'Eswatini': (-26.5225, 31.4659),
        'Ethiopia': (9.1450, 40.4897),
        'Fiji': (-17.7134, 178.0650),
        'Finland': (61.9241, 25.7482),
        'France': (46.2276, 2.2137),
        'Gabon': (-0.8037, 11.6094),
        'Gambia': (13.4432, -15.3101),
        'Georgia': (42.3154, 43.3569),
        'Germany': (51.1657, 10.4515),
        'Ghana': (7.9465, -1.0232),
        'Greece': (39.0742, 21.8243),
        'Grenada': (12.1165, -61.6790),
        'Guatemala': (15.7835, -90.2308),
        'Guinea': (9.9456, -9.6966),
        'Guinea-Bissau': (11.8037, -15.1804),
        'Guyana': (4.8604, -58.9302),
        'Haiti': (18.9712, -72.2852),
        'Honduras': (15.1999, -86.2419),
        'Hungary': (47.1625, 19.5033),
        'Iceland': (64.9631, -19.0208),
        'India': (20.5937, 78.9629),
        'Indonesia': (-0.7893, 113.9213),
        'Iran': (32.4279, 53.6880),
        'Iraq': (33.2232, 43.6793),
        'Ireland': (53.1424, -7.6921),
        'Israel': (31.0461, 34.8516),
        'Italy': (41.8719, 12.5674),
        'Ivory Coast': (7.5400, -5.5471),
        'Jamaica': (18.1096, -77.2975),
        'Japan': (36.2048, 138.2529),
        'Jordan': (30.5852, 36.2384),
        'Kazakhstan': (48.0196, 66.9237),
        'Kenya': (-0.0236, 37.9062),
        'Kiribati': (-3.3704, -168.7340),
        'Kuwait': (29.3117, 47.4818),
        'Kyrgyzstan': (41.2044, 74.7661),
        'Laos': (19.8563, 102.4955),
        'Latvia': (56.8796, 24.6032),
        'Lebanon': (33.8547, 35.8623),
        'Lesotho': (-29.6099, 28.2336),
        'Liberia': (6.4281, -9.4295),
        'Libya': (26.3351, 17.2283),
        'Lithuania': (55.1694, 23.8813),
        'Luxembourg': (49.8153, 6.1296),
        'Madagascar': (-18.7669, 46.8691),
        'Malawi': (-13.2543, 34.3015),
        'Malaysia': (4.2105, 108.9758),
        'Maldives': (3.2028, 73.2207),
        'Mali': (17.5707, -3.9962),
        'Malta': (35.9375, 14.3754),
        'Marshall Islands': (7.1315, 171.1845),
        'Mauritania': (21.0079, -10.9408),
        'Mauritius': (-20.3484, 57.5522),
        'Mexico': (23.6345, -102.5528),
        'Micronesia': (7.4256, 150.5508),
        'Moldova': (47.4116, 28.3699),
        'Monaco': (43.7384, 7.4246),
        'Mongolia': (46.8625, 103.8467),
        'Montenegro': (42.7087, 19.3744),
        'Morocco': (31.7917, -7.0926),
        'Mozambique': (-18.6657, 35.5296),
        'Myanmar': (21.9162, 95.9560),
        'Namibia': (-22.9576, 18.4904),
        'Nauru': (-0.5228, 166.9315),
        'Nepal': (28.3949, 84.1240),
        'Netherlands': (52.1326, 5.2913),
        'New Zealand': (-40.9006, 174.8860),
        'Nicaragua': (12.8654, -85.2072),
        'Niger': (17.6078, 8.0817),
        'Nigeria': (9.0820, 8.6753),
        'North Korea': (40.3399, 127.5101),
        'Norway': (60.4720, 8.4689),
        'Oman': (21.4735, 55.9754),
        'Pakistan': (30.3753, 69.3451),
        'Palau': (7.5150, 134.5825),
        'Panama': (8.5380, -80.7821),
        'Papua New Guinea': (-6.3150, 143.9555),
        'Paraguay': (-23.4425, -58.4438),
        'Peru': (-9.1900, -75.0152),
        'Philippines': (12.8797, 121.7740),
        'Poland': (51.9194, 19.1451),
        'Portugal': (39.3999, -8.2245),
        'Qatar': (25.3548, 51.1839),
        'Republic of the Congo': (-0.2280, 15.8277),
        'Romania': (45.9432, 24.9668),
        'Russia': (61.5240, 105.3188),
        'Rwanda': (-1.9403, 29.8739),
        'Saint Kitts and Nevis': (17.3578, -62.7830),
        'Saint Lucia': (13.9094, -60.9789),
        'Saint Vincent and the Grenadines': (12.9843, -61.2872),
        'Samoa': (-13.7590, -172.1046),
        'San Marino': (43.9424, 12.4578),
        'Sao Tome and Principe': (0.1864, 6.6131),
        'Saudi Arabia': (23.8859, 45.0792),
        'Senegal': (14.4974, -14.4524),
        'Serbia': (44.0165, 21.0059),
        'Seychelles': (-4.6796, 55.4920),
        'Sierra Leone': (8.4606, -11.7799),
        'Singapore': (1.3521, 103.8198),
        'Slovakia': (48.6690, 19.6990),
        'Slovenia': (46.0569, 14.5058),
        'Solomon Islands': (-9.6457, 160.1562),
        'Somalia': (5.1521, 46.1996),
        'South Africa': (-30.5595, 22.9375),
        'South Korea': (35.9078, 127.7669),
        'South Sudan': (6.8770, 31.3070),
        'Spain': (40.4637, -3.7492),
        'Sri Lanka': (7.8731, 80.7718),
        'Sudan': (12.8628, 30.2176),
        'Suriname': (3.9193, -56.0278),
        'Sweden': (60.1282, 18.6435),
        'Switzerland': (46.8182, 8.2275),
        'Syria': (34.8021, 38.9968),
        'Tanzania': (-6.3690, 34.8888),
        'Thailand': (15.8700, 100.9925),
        'Togo': (8.6195, 0.8248),
        'Tonga': (-21.1790, -175.1982),
        'Trinidad and Tobago': (10.6598, -61.5190),
        'Tunisia': (33.8869, 9.5375),
        'Turkey': (38.9637, 35.2433),
        'Turkmenistan': (38.9697, 59.5563),
        'Tuvalu': (-7.1095, 177.6493),
        'Uganda': (1.3733, 32.2903),
        'Ukraine': (48.3794, 31.1656),
        'United Arab Emirates': (24.0002, 54.0000),
        'United Kingdom': (55.3781, -3.4360),
        'United States': (37.0902, -95.7129),
        'Uruguay': (-32.5228, -55.7658),
        'Uzbekistan': (41.3775, 64.5853),
        'Vanuatu': (-15.3767, 166.9592),
        'Venezuela': (6.4238, -66.5897),
        'Vietnam': (14.0583, 108.2772),
        'Yemen': (15.5527, 48.5164),
        'Zambia': (-13.1339, 27.8493),
        'Zimbabwe': (-19.0154, 29.1549)
    }
    
    # Common country name mappings for WHO dataset
    country_mapping = {
        'United States of America': 'United States',
        'United Kingdom of Great Britain and Northern Ireland': 'United Kingdom',
        'Russian Federation': 'Russia',
        'Iran (Islamic Republic of)': 'Iran',
        'Venezuela (Bolivarian Republic of)': 'Venezuela',
        'Syrian Arab Republic': 'Syria',
        'Korea, Republic of': 'South Korea',
        'Korea, Democratic People\'s Republic of': 'North Korea',
        'Lao People\'s Democratic Republic': 'Laos',
        'Brunei Darussalam': 'Brunei',
        'Côte d\'Ivoire': 'Ivory Coast',
        'Congo': 'Republic of the Congo',
        'Congo, Democratic Republic of the': 'Democratic Republic of the Congo',
        'Tanzania, United Republic of': 'Tanzania',
        'Central African Republic': 'Central African Republic',
        'Equatorial Guinea': 'Equatorial Guinea',
        'Guinea-Bissau': 'Guinea-Bissau',
        'Sao Tome and Principe': 'Sao Tome and Principe',
        'Cabo Verde': 'Cape Verde',
        'Eswatini': 'Eswatini',
        'Timor-Leste': 'East Timor'
    }
    
    # Try to find the country coordinates
    # First try with the original name
    if country_name in country_coordinates:
        return country_coordinates[country_name]
    
    # Try with mapped name
    mapped_name = country_mapping.get(country_name, country_name)
    if mapped_name in country_coordinates:
        return country_coordinates[mapped_name]
    
    # Try fuzzy matching for close matches
    for known_country, coords in country_coordinates.items():
        if country_name.lower() in known_country.lower() or known_country.lower() in country_name.lower():
            return coords
    
    return None

def create_interactive_map(df, metric='Life expectancy', year=None, region=None):
    """
    Create an interactive map visualization.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        metric (str): Metric to visualize on the map
        year (int): Specific year to filter (optional)
        region (str): Specific region to filter (optional)
        
    Returns:
        folium.Map: Interactive map
    """
    # Filter data based on parameters
    map_data = df.copy()
    
    if year is not None:
        map_data = map_data[map_data['Year'] == year]
    
    if region is not None:
        map_data = map_data[map_data['Region'] == region]
    
    # Group by country and calculate average for the metric
    country_data = map_data.groupby('Country').agg({
        metric: 'mean',
        'Status': 'first',
        'Region': 'first',
        'Year': ['min', 'max', 'count']
    }).reset_index()
    
    # Flatten column names
    country_data.columns = ['Country', f'{metric}_avg', 'Status', 'Region', 'Year_min', 'Year_max', 'Year_count']
    
    # Remove rows with missing metric values
    country_data = country_data.dropna(subset=[f'{metric}_avg'])
    
    if len(country_data) == 0:
        return None
    
    # Create base map
    center_lat, center_lon = 20, 0
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=2,
        tiles='OpenStreetMap'
    )
    
    # Add different tile layers
    folium.TileLayer('OpenStreetMap', name='OpenStreetMap').add_to(m)
    folium.TileLayer('CartoDB positron', name='CartoDB Positron').add_to(m)
    folium.TileLayer('CartoDB dark_matter', name='CartoDB Dark').add_to(m)
    
    # Create marker cluster
    marker_cluster = MarkerCluster().add_to(m)
    
    # Color mapping for status
    status_colors = {
        'Developed': 'green',
        'Developing': 'orange'
    }
    
    # Add markers for each country
    for idx, row in country_data.iterrows():
        country_name = row['Country']
        metric_value = row[f'{metric}_avg']
        status = row['Status']
        region = row['Region']
        year_range = f"{row['Year_min']}-{row['Year_max']}"
        data_points = row['Year_count']
        
        # Get coordinates
        coords = get_country_coordinates(country_name)
        
        if coords:
            lat, lon = coords
            
            # Create popup content
            popup_content = f"""
            <div style="width: 200px;">
                <h4>{country_name}</h4>
                <p><strong>{metric}:</strong> {metric_value:.2f}</p>
                <p><strong>Status:</strong> {status}</p>
                <p><strong>Region:</strong> {region}</p>
                <p><strong>Years:</strong> {year_range}</p>
                <p><strong>Data Points:</strong> {data_points}</p>
            </div>
            """
            
            # Create marker
            folium.Marker(
                location=[lat, lon],
                popup=folium.Popup(popup_content, max_width=300),
                tooltip=f"{country_name}: {metric_value:.2f}",
                icon=folium.Icon(
                    color=status_colors.get(status, 'blue'),
                    icon='info-sign'
                )
            ).add_to(marker_cluster)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    return m

def create_choropleth_map(df, metric='Life expectancy', year=None):
    """
    Create a choropleth map using Plotly.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        metric (str): Metric to visualize
        year (int): Specific year to filter (optional)
        
    Returns:
        plotly.graph_objects.Figure: Choropleth map
    """
    # Filter data based on parameters
    map_data = df.copy()
    
    if year is not None:
        map_data = map_data[map_data['Year'] == year]
    
    # Group by country and calculate average for the metric
    country_data = map_data.groupby('Country').agg({
        metric: 'mean',
        'Status': 'first',
        'Region': 'first'
    }).reset_index()
    
    # Remove rows with missing metric values
    country_data = country_data.dropna(subset=[metric])
    
    if len(country_data) == 0:
        return None
    
    # Create choropleth map
    fig = px.choropleth(
        country_data,
        locations='Country',
        locationmode='country names',
        color=metric,
        hover_name='Country',
        hover_data=['Status', 'Region'],
        color_continuous_scale='Viridis',
        title=f'Global {metric} Distribution',
        labels={metric: metric}
    )
    
    fig.update_layout(
        geo=dict(
            showframe=False,
            showcoastlines=True,
            projection_type='equirectangular'
        ),
        height=600
    )
    
    return fig

def create_heatmap_data(df, metric='Life expectancy', year=None):
    """
    Create heatmap data for visualization.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        metric (str): Metric to visualize
        year (int): Specific year to filter (optional)
        
    Returns:
        list: List of [lat, lon, value] for heatmap
    """
    # Filter data based on parameters
    map_data = df.copy()
    
    if year is not None:
        map_data = map_data[map_data['Year'] == year]
    
    # Group by country and calculate average for the metric
    country_data = map_data.groupby('Country').agg({
        metric: 'mean'
    }).reset_index()
    
    # Remove rows with missing metric values
    country_data = country_data.dropna(subset=[metric])
    
    heatmap_data = []
    
    for idx, row in country_data.iterrows():
        country_name = row['Country']
        metric_value = row[metric]
        
        # Get coordinates
        coords = get_country_coordinates(country_name)
        
        if coords:
            lat, lon = coords
            heatmap_data.append([lat, lon, metric_value])
    
    return heatmap_data

def show_map_visualizations(df):
    """
    Display comprehensive map visualizations.
    
    Args:
        df (pd.DataFrame): Input DataFrame
    """
    st.markdown("## 🌍 Interactive Map Visualizations")
    
    # Sidebar controls
    st.sidebar.markdown("### Map Controls")
    
    # Metric selection
    metric_options = ['Life expectancy'] + [col for col in df.select_dtypes(include=[np.number]).columns 
                                          if col not in ['Year', 'Life expectancy']]
    selected_metric = st.sidebar.selectbox("Select Metric:", metric_options)
    
    # Year filter
    years = sorted(df['Year'].unique())
    year_filter = st.sidebar.selectbox("Select Year (Optional):", ['All Years'] + list(years))
    selected_year = None if year_filter == 'All Years' else int(year_filter)
    
    # Region filter
    regions = sorted(df['Region'].unique())
    region_filter = st.sidebar.selectbox("Select Region (Optional):", ['All Regions'] + list(regions))
    selected_region = None if region_filter == 'All Regions' else region_filter
    
    # Map type selection
    map_type = st.sidebar.selectbox("Map Type:", ["Interactive Markers", "Choropleth", "Heatmap"])
    
    # Create tabs for different map views
    tab1, tab2, tab3 = st.tabs(["🌍 Interactive Map", "📊 Statistics", "📈 Trends"])
    
    with tab1:
        st.subheader(f"🌍 {selected_metric} Distribution")
        
        if map_type == "Interactive Markers":
            # Create interactive map with markers
            map_obj = create_interactive_map(df, selected_metric, selected_year, selected_region)
            
            if map_obj:
                folium_static(map_obj, width=800, height=600)
            else:
                st.warning("No data available for the selected filters.")
        
        elif map_type == "Choropleth":
            # Create choropleth map
            fig = create_choropleth_map(df, selected_metric, selected_year)
            
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No data available for the selected filters.")
        
        elif map_type == "Heatmap":
            # Create heatmap
            heatmap_data = create_heatmap_data(df, selected_metric, selected_year)
            
            if heatmap_data:
                # Create base map for heatmap
                center_lat, center_lon = 20, 0
                m = folium.Map(
                    location=[center_lat, center_lon],
                    zoom_start=2,
                    tiles='CartoDB dark_matter'
                )
                
                # Add heatmap layer
                HeatMap(heatmap_data, radius=15).add_to(m)
                
                folium_static(m, width=800, height=600)
            else:
                st.warning("No data available for the selected filters.")
    
    with tab2:
        st.subheader("📊 Geographic Statistics")
        
        # Filter data based on selections
        filtered_data = df.copy()
        if selected_year:
            filtered_data = filtered_data[filtered_data['Year'] == selected_year]
        if selected_region:
            filtered_data = filtered_data[filtered_data['Region'] == selected_region]
        
        # Calculate statistics by region
        region_stats = filtered_data.groupby('Region').agg({
            selected_metric: ['mean', 'std', 'min', 'max', 'count']
        }).round(2)
        
        # Flatten column names
        region_stats.columns = [f'{col[0]}_{col[1]}' for col in region_stats.columns]
        region_stats = region_stats.reset_index()
        
        # Display region statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Regional Statistics:**")
            safe_display_dataframe(region_stats, use_container_width=True)
        
        with col2:
            # Create bar chart of regional averages
            fig = px.bar(
                region_stats, 
                x='Region', 
                y=f'{selected_metric}_mean',
                title=f'Average {selected_metric} by Region',
                error_y=f'{selected_metric}_std'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Top and bottom countries
        country_stats = filtered_data.groupby('Country').agg({
            selected_metric: 'mean'
        }).reset_index()
        country_stats = country_stats.dropna()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Top 10 Countries:**")
            top_countries = country_stats.nlargest(10, selected_metric)
            safe_display_dataframe(top_countries, use_container_width=True)
        
        with col2:
            st.write("**Bottom 10 Countries:**")
            bottom_countries = country_stats.nsmallest(10, selected_metric)
            safe_display_dataframe(bottom_countries, use_container_width=True)
    
    with tab3:
        st.subheader("📈 Temporal Trends")
        
        # Time series analysis
        if selected_year is None:  # Only show trends if no specific year is selected
            # Calculate trends by region
            trend_data = df.groupby(['Year', 'Region'])[selected_metric].mean().reset_index()
            
            # Create trend plot
            fig = px.line(
                trend_data,
                x='Year',
                y=selected_metric,
                color='Region',
                title=f'{selected_metric} Trends by Region Over Time'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Calculate growth rates
            st.write("**Growth Analysis:**")
            
            # Calculate average annual growth by region
            growth_data = []
            for region in df['Region'].unique():
                region_data = df[df['Region'] == region].groupby('Year')[selected_metric].mean()
                if len(region_data) > 1:
                    growth_rate = (region_data.iloc[-1] - region_data.iloc[0]) / (region_data.index[-1] - region_data.index[0])
                    growth_data.append({
                        'Region': region,
                        'Growth Rate': growth_rate,
                        'Start Value': region_data.iloc[0],
                        'End Value': region_data.iloc[-1]
                    })
            
            if growth_data:
                growth_df = pd.DataFrame(growth_data)
                growth_df = growth_df.sort_values('Growth Rate', ascending=False)
                safe_display_dataframe(growth_df, use_container_width=True)
        else:
            st.info("Select 'All Years' to view temporal trends.")

def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">📊 Life Expectancy Prediction</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    st.sidebar.markdown('<h2 class="sidebar-header">Navigation</h2>', unsafe_allow_html=True)
    
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🏠 Home", "📈 Data Analysis", "🌍 Map Visualization", "🔍 Outlier Detection", "📋 Outlier Details", "🤖 Model Training", "📊 Results", "🔮 Predictions", "📋 About"]
    )
    
    # Load data
    with st.spinner("Loading dataset..."):
        df, preprocessor = load_data()
    
    if df is None:
        st.error("Failed to load dataset. Please check the file path.")
        return
    
    # Home page
    if page == "🏠 Home":
        st.markdown("## Welcome to Life Expectancy Prediction")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            This application uses machine learning to predict life expectancy based on various health, 
            economic, and social indicators from the World Health Organization (WHO) dataset.
            
            ### Features:
            - **Data Preprocessing**: Handle missing values, data cleaning, and feature engineering
            - **Comprehensive Outlier Detection**: Multiple detection methods with detailed analysis
            - **Multiple Models**: Linear Regression, Ridge, Lasso, Elastic Net, Random Forest, Gradient Boosting
            - **Model Evaluation**: Cross-validation, hyperparameter tuning, and performance metrics
            - **Advanced Visualization**: Interactive plots and comprehensive analysis
            - **Predictions**: Make predictions on new data
            
            ### Dataset Overview:
            """)
            
            # Dataset info
            st.info(f"📊 **Dataset Shape**: {df.shape[0]} rows × {df.shape[1]} columns")
            st.info(f"🌍 **Countries**: {df['Country'].nunique()}")
            st.info(f"📅 **Years**: {df['Year'].min()} - {df['Year'].max()}")
            st.info(f"🎯 **Target Variable**: Life expectancy")
        
        with col2:
            st.markdown("### Quick Stats")
            
            # Life expectancy stats
            life_exp = df['Life expectancy'].dropna()
            col2.metric("Average Life Expectancy", f"{life_exp.mean():.1f} years")
            col2.metric("Min Life Expectancy", f"{life_exp.min():.1f} years")
            col2.metric("Max Life Expectancy", f"{life_exp.max():.1f} years")
    
    # Data Analysis page
    elif page == "📈 Data Analysis":
        st.markdown("## Data Analysis")
        
        # Data overview
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Dataset Information")
            st.write(f"**Shape**: {df.shape}")
            st.write(f"**Columns**: {list(df.columns)}")
            
            # Missing values
            missing_data = df.isnull().sum()
            missing_df = pd.DataFrame({
                'Column': missing_data.index,
                'Missing Values': missing_data.values,
                'Percentage': (missing_data.values / len(df)) * 100
            }).sort_values('Missing Values', ascending=False)
            
            st.subheader("Missing Values Analysis")
            safe_display_dataframe(missing_df, use_container_width=True)
        
        with col2:
            st.subheader("Data Types")
            st.write(df.dtypes.value_counts())
            
            # Target distribution
            st.subheader("Life Expectancy Distribution")
            fig = px.histogram(df, x='Life expectancy', nbins=30, 
                             title="Distribution of Life Expectancy")
            st.plotly_chart(fig, use_container_width=True)
        
        # Correlation analysis
        st.subheader("Correlation Analysis")
        
        # Select numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        corr_matrix = df[numerical_cols].corr()
        
        # Create correlation heatmap
        fig = px.imshow(corr_matrix, 
                       title="Correlation Matrix",
                       color_continuous_scale='RdBu',
                       aspect='auto')
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature vs target analysis
        st.subheader("Feature vs Target Analysis")
        
        # Select feature to analyze
        feature_cols = [col for col in numerical_cols if col != 'Life expectancy']
        selected_feature = st.selectbox("Select a feature to analyze:", feature_cols)
        
        if selected_feature:
            # Create scatter plot with country information (without trendline)
            fig = px.scatter(
                df, 
                x=selected_feature, 
                y='Life expectancy',
                color='Country',  # Color by country
                hover_data=['Country', 'Year', 'Status', 'Region'],  # Show additional info on hover
                title=f"Life Expectancy vs {selected_feature} (Colored by Country)"
            )
            
            # Add linear regression line for prediction
            # Prepare data for linear regression
            X = df[[selected_feature]].dropna()
            y = df['Life expectancy'].dropna()
            
            # Align the data
            common_indices = X.index.intersection(y.index)
            X = X.loc[common_indices]
            y = y.loc[common_indices]
            
            if len(X) > 0:
                try:
                    # Fit linear regression
                    lr = LinearRegression()
                    lr.fit(X, y)
                    
                    # Generate points for the regression line
                    x_min, x_max = X[selected_feature].min(), X[selected_feature].max()
                    x_range = np.linspace(x_min, x_max, 100)
                    y_pred = lr.predict(x_range.reshape(-1, 1))
                    
                    # Add regression line to the plot
                    fig.add_trace(go.Scatter(
                        x=x_range,
                        y=y_pred,
                        mode='lines',
                        name='Linear Regression (Prediction)',
                        line=dict(color='red', width=3, dash='dash'),
                        showlegend=True
                    ))
                    
                    # Calculate R² score
                    y_pred_all = lr.predict(X)
                    r2_score = lr.score(X, y)
                    
                    # Add R² information to the plot
                    fig.add_annotation(
                        x=0.05, y=0.95,
                        xref='paper', yref='paper',
                        text=f'R² = {r2_score:.3f}',
                        showarrow=False,
                        font=dict(size=14, color='red'),
                        bgcolor='rgba(255,255,255,0.8)',
                        bordercolor='red',
                        borderwidth=1
                    )
                except Exception as e:
                    st.warning(f"⚠️ Could not calculate linear regression: {e}")
                    st.info("The scatter plot is displayed without the regression line.")
            
            # Update layout for better readability
            fig.update_layout(
                height=600,
                showlegend=True,  # Show legend for the regression line
                hovermode='closest'
            )
            
            # Add a note about hover information
            st.info("💡 **Hover over any point to see the country, year, status, and region information**")
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show country statistics for the selected feature
            st.subheader(f"Country Statistics for {selected_feature}")
            
            # Calculate statistics by country
            country_stats = df.groupby('Country').agg({
                selected_feature: ['mean', 'std', 'min', 'max'],
                'Life expectancy': ['mean', 'std', 'min', 'max']
            }).round(2)
            
            # Flatten column names
            country_stats.columns = [f"{col[0]}_{col[1]}" for col in country_stats.columns]
            country_stats = country_stats.reset_index()
            
            # Show top and bottom countries by the selected feature
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Top 10 Countries by Average " + selected_feature + ":**")
                top_countries = country_stats.nlargest(10, f'{selected_feature}_mean')
                safe_display_dataframe(top_countries[['Country', f'{selected_feature}_mean', f'{selected_feature}_std']], use_container_width=True)
            
            with col2:
                st.write("**Bottom 10 Countries by Average " + selected_feature + ":**")
                bottom_countries = country_stats.nsmallest(10, f'{selected_feature}_mean')
                safe_display_dataframe(bottom_countries[['Country', f'{selected_feature}_mean', f'{selected_feature}_std']], use_container_width=True)
    
    # Map Visualization page
    elif page == "🌍 Map Visualization":
        show_map_visualizations(df)
    
    # Outlier Detection page
    elif page == "🔍 Outlier Detection":
        st.markdown("## Comprehensive Outlier Detection")
        
        # Initialize outlier detection system
        outlier_system = OutlierDetectionSystem()
        
        # Detection methods selection
        st.subheader("Detection Methods")
        
        col1, col2 = st.columns(2)
        
        with col1:
            univariate_methods = st.multiselect(
                "Select univariate detection methods:",
                ['zscore', 'iqr', 'modified_zscore', 'isolation_forest'],
                default=['zscore', 'iqr']
            )
            
            zscore_threshold = st.slider("Z-score threshold", 2.0, 5.0, 3.0, 0.1)
            iqr_multiplier = st.slider("IQR multiplier", 1.0, 3.0, 1.5, 0.1)
        
        with col2:
            multivariate_methods = st.multiselect(
                "Select multivariate detection methods:",
                ['isolation_forest', 'local_outlier_factor', 'elliptic_envelope'],
                default=['isolation_forest']
            )
            
            contamination = st.slider("Contamination rate", 0.01, 0.2, 0.1, 0.01)
        
        # Run outlier analysis
        if st.button("🔍 Run Outlier Analysis", type="primary"):
            with st.spinner("Performing comprehensive outlier analysis..."):
                # Perform analysis
                outlier_analysis = outlier_system.comprehensive_outlier_analysis(
                    df, 
                    target_column='Life expectancy',
                    methods=univariate_methods
                )
                
                # Store in session state
                st.session_state.outlier_analysis = outlier_analysis
                st.session_state.outlier_system = outlier_system
                
                st.success("✅ Outlier analysis completed!")
        
        # Display results if available
        if 'outlier_analysis' in st.session_state:
            outlier_analysis = st.session_state.outlier_analysis
            outlier_system = st.session_state.outlier_system
            
            # Summary metrics
            st.subheader("Outlier Detection Summary")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                total_methods = len(outlier_analysis['target_outliers'])
                st.metric("Detection Methods", total_methods)
            
            with col2:
                total_outliers = sum([results['outlier_count'] for results in outlier_analysis['target_outliers'].values()])
                st.metric("Total Outliers Detected", total_outliers)
            
            with col3:
                avg_percentage = np.mean([results['outlier_percentage'] for results in outlier_analysis['target_outliers'].values()])
                st.metric("Average Outlier %", f"{avg_percentage:.2f}%")
            
            # Detailed results
            st.subheader("Detailed Results by Method")
            
            # Create results table
            results_data = []
            for method, results in outlier_analysis['target_outliers'].items():
                results_data.append({
                    'Method': method,
                    'Outlier Count': results['outlier_count'],
                    'Outlier Percentage': f"{results['outlier_percentage']:.2f}%",
                    'Threshold': results['threshold']
                })
            
            results_df = pd.DataFrame(results_data)
            safe_display_dataframe(results_df, use_container_width=True)
            
            # Recommendations
            st.subheader("Recommendations")
            summary = outlier_analysis['summary']
            
            for rec in summary['recommendations']:
                if "High outlier percentage" in rec:
                    st.warning(rec)
                elif "Moderate outlier percentage" in rec:
                    st.info(rec)
                else:
                    st.success(rec)
            
            # Visualizations
            st.subheader("Outlier Analysis Visualizations")
            
            # Generate visualizations
            outlier_fig = outlier_system.visualize_outlier_analysis(
                df, 
                outlier_analysis, 
                target_column='Life expectancy'
            )
            
            st.plotly_chart(outlier_fig, use_container_width=True)
            
            # Interactive outlier handling
            st.subheader("Interactive Outlier Handling")
            
            handling_options = outlier_system.interactive_outlier_handling(
                df, 
                outlier_analysis, 
                target_column='Life expectancy'
            )
            
            # Display handling options
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Outlier Counts by Method:**")
                for method, stats in handling_options['outlier_counts'].items():
                    st.write(f"- {method}: {stats['count']} ({stats['percentage']:.2f}%)")
            
            with col2:
                st.write("**Impact Analysis:**")
                for method, option in handling_options['removal_options'].items():
                    impact = option['impact']
                    st.write(f"- {method}: {impact['records_removed']} records, mean change: {impact['mean_change']:.3f}")
            
            # Outlier removal options
            st.subheader("Outlier Removal Options")
            
            removal_choice = st.selectbox(
                "Choose outlier handling strategy:",
                ["Keep all outliers", "Remove outliers using recommended method", "Remove outliers using specific method"]
            )
            
            if removal_choice == "Remove outliers using recommended method":
                best_method = min(outlier_analysis['target_outliers'].keys(), 
                                 key=lambda x: outlier_analysis['target_outliers'][x]['outlier_count'])
                
                if st.button(f"Remove outliers using {best_method}"):
                    outlier_indices = outlier_analysis['target_outliers'][best_method]['outlier_indices']
                    df_clean, df_removed, removal_summary = outlier_system.remove_outliers(
                        df, outlier_indices, best_method
                    )
                    
                    st.session_state.df_clean = df_clean
                    st.session_state.df_removed = df_removed
                    st.session_state.removal_summary = removal_summary
                    
                    st.success(f"✅ Removed {removal_summary['removed_count']} outliers using {best_method}")
                    
            elif removal_choice == "Remove outliers using specific method":
                method_choice = st.selectbox(
                    "Select method:",
                    list(handling_options['detection_methods'])
                )
                
                if st.button(f"Remove outliers using {method_choice}"):
                    outlier_indices = outlier_analysis['target_outliers'][method_choice]['outlier_indices']
                    df_clean, df_removed, removal_summary = outlier_system.remove_outliers(
                        df, outlier_indices, method_choice
                    )
                    
                    st.session_state.df_clean = df_clean
                    st.session_state.df_removed = df_removed
                    st.session_state.removal_summary = removal_summary
                    
                    st.success(f"✅ Removed {removal_summary['removed_count']} outliers using {method_choice}")
            
            # Show removal summary if available
            if 'removal_summary' in st.session_state:
                st.subheader("Outlier Removal Summary")
                
                summary = st.session_state.removal_summary
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Method Used", summary['method'])
                
                with col2:
                    st.metric("Records Removed", summary['removed_count'])
                
                with col3:
                    st.metric("Removal %", f"{summary['removal_percentage']:.2f}%")
                
                # Show before/after comparison
                if 'df_clean' in st.session_state:
                    st.subheader("Before vs After Outlier Removal")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Before Removal:**")
                        st.write(f"Records: {summary['original_count']}")
                        st.write(f"Mean Life Expectancy: {df['Life expectancy'].mean():.2f}")
                        st.write(f"Std Life Expectancy: {df['Life expectancy'].std():.2f}")
                    
                    with col2:
                        st.write("**After Removal:**")
                        st.write(f"Records: {summary['remaining_count']}")
                        st.write(f"Mean Life Expectancy: {st.session_state.df_clean['Life expectancy'].mean():.2f}")
                        st.write(f"Std Life Expectancy: {st.session_state.df_clean['Life expectancy'].std():.2f}")
    
    # Outlier Details page
    elif page == "📋 Outlier Details":
        st.markdown("## Detailed Outlier Information")
        
        if 'outlier_analysis' not in st.session_state:
            st.warning("Please run outlier analysis first in the Outlier Detection page.")
            return
        
        outlier_analysis = st.session_state.outlier_analysis
        
        # Show detailed outlier information
        show_detailed_outlier_info(df, outlier_analysis)
    
    # Model Training page
    elif page == "🤖 Model Training":
        st.markdown("## Model Training")
        
        # Check if cleaned data is available
        if 'df_clean' in st.session_state:
            df_to_use = st.session_state.df_clean
            st.info("Using outlier-cleaned dataset for model training")
        else:
            df_to_use = df
            st.info("Using original dataset for model training")
        
        # Preprocessing options
        st.subheader("Preprocessing Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            imputation_strategy = st.selectbox(
                "Imputation Strategy",
                ["median", "mean", "most_frequent"]
            )
        
        with col2:
            test_size = st.slider("Test Size", 0.1, 0.5, 0.2, 0.05)
            random_state = st.number_input("Random State", value=42, min_value=0)
        
        # Train models button
        if st.button("🚀 Train Models", type="primary"):
            with st.spinner("Training models..."):
                # Preprocess data
                X_scaled, y, feature_names, df_processed = preprocessor.prepare_final_dataset(df_to_use)
                
                # Train models
                results, model_trainer = train_models(X_scaled, y, feature_names)
                
                # Store in session state
                st.session_state.results = results
                st.session_state.model_trainer = model_trainer
                st.session_state.feature_names = feature_names
                st.session_state.X_scaled = X_scaled
                st.session_state.y = y
                st.session_state.df_processed = df_processed
                st.session_state.preprocessor = preprocessor
                
                st.success("✅ Models trained successfully!")
        
        # Display results if available
        if 'results' in st.session_state:
            st.subheader("Model Performance")
            
            # Create performance comparison
            results = st.session_state.results
            
            # Performance metrics table
            performance_data = []
            for name, result in results.items():
                performance_data.append({
                    'Model': name,
                    'Test R²': f"{result['test_r2']:.4f}",
                    'Test RMSE': f"{result['test_rmse']:.4f}",
                    'CV R²': f"{result['cv_mean']:.4f} (±{result['cv_std']:.4f})"
                })
            
            performance_df = pd.DataFrame(performance_data)
            safe_display_dataframe(performance_df, use_container_width=True)
            
            # Performance visualization
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('R² Scores', 'RMSE Scores'),
                specs=[[{"type": "bar"}, {"type": "bar"}]]
            )
            
            model_names = list(results.keys())
            test_r2 = [results[name]['test_r2'] for name in model_names]
            test_rmse = [results[name]['test_rmse'] for name in model_names]
            
            fig.add_trace(
                go.Bar(x=model_names, y=test_r2, name='Test R²'),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Bar(x=model_names, y=test_rmse, name='Test RMSE'),
                row=1, col=2
            )
            
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    # Results page
    elif page == "📊 Results":
        st.markdown("## Model Results")
        
        if 'results' not in st.session_state:
            st.warning("Please train models first in the Model Training page.")
            return
        
        results = st.session_state.results
        model_trainer = st.session_state.model_trainer
        
        # Best model info
        best_model_name = model_trainer.best_model_name
        st.success(f"🏆 **Best Model**: {best_model_name}")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Best Test R²", f"{results[best_model_name]['test_r2']:.4f}")
        
        with col2:
            st.metric("Best Test RMSE", f"{results[best_model_name]['test_rmse']:.4f}")
        
        with col3:
            st.metric("Best CV R²", f"{results[best_model_name]['cv_mean']:.4f}")
        
        # Actual vs Predicted plot
        st.subheader("Actual vs Predicted Values")
        
        y_test = results[best_model_name]['y_test']
        y_pred = results[best_model_name]['y_pred_test']
        
        fig = px.scatter(x=y_test, y=y_pred, 
                        title=f"Actual vs Predicted - {best_model_name}",
                        labels={'x': 'Actual Values', 'y': 'Predicted Values'})
        
        # Add perfect prediction line
        min_val, max_val = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
        fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val], 
                                mode='lines', name='Perfect Prediction',
                                line=dict(color='red', dash='dash')))
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature importance
        st.subheader("Feature Importance")
        
        importance_df = model_trainer.analyze_feature_importance()
        
        if importance_df is not None:
            # Top 10 features
            top_10 = importance_df.head(10)
            
            fig = px.bar(top_10, x='Importance', y='Feature', orientation='h',
                        title="Top 10 Most Important Features")
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature importance table
            safe_display_dataframe(importance_df, use_container_width=True)
    
    # Predictions page
    elif page == "🔮 Predictions":
        st.markdown("## Make Predictions")
        
        if 'model_trainer' not in st.session_state:
            st.warning("Please train models first in the Model Training page.")
            return
        
        model_trainer = st.session_state.model_trainer
        feature_names = st.session_state.feature_names
        
        st.subheader("Input Parameters")
        
        # Create input form
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                adult_mortality = st.number_input("Adult Mortality", value=100.0, min_value=0.0)
                infant_deaths = st.number_input("Infant Deaths", value=10.0, min_value=0.0)
                alcohol = st.number_input("Alcohol Consumption", value=5.0, min_value=0.0)
                percentage_expenditure = st.number_input("Percentage Expenditure", value=1000.0, min_value=0.0)
                hepatitis_b = st.number_input("Hepatitis B Coverage", value=80.0, min_value=0.0, max_value=100.0)
                measles = st.number_input("Measles Cases", value=100.0, min_value=0.0)
                bmi = st.number_input("BMI", value=25.0, min_value=10.0, max_value=50.0)
                under_five_deaths = st.number_input("Under-Five Deaths", value=15.0, min_value=0.0)
                polio = st.number_input("Polio Coverage", value=80.0, min_value=0.0, max_value=100.0)
                total_expenditure = st.number_input("Total Expenditure", value=5.0, min_value=0.0)
            
            with col2:
                diphtheria = st.number_input("Diphtheria Coverage", value=80.0, min_value=0.0, max_value=100.0)
                hiv_aids = st.number_input("HIV/AIDS", value=0.1, min_value=0.0)
                gdp = st.number_input("GDP", value=5000.0, min_value=0.0)
                population = st.number_input("Population", value=1000000.0, min_value=0.0)
                thinness_1_19 = st.number_input("Thinness 1-19 Years", value=5.0, min_value=0.0)
                thinness_5_9 = st.number_input("Thinness 5-9 Years", value=5.0, min_value=0.0)
                income_composition = st.number_input("Income Composition", value=0.5, min_value=0.0, max_value=1.0)
                schooling = st.number_input("Schooling", value=12.0, min_value=0.0, max_value=25.0)
                year = st.number_input("Year", value=2015, min_value=2000, max_value=2030)
                status = st.selectbox("Status", ["Developing", "Developed"])
                region = st.selectbox("Region", ["Africa", "Asia", "Europe", "North America", "South America", "Oceania"])
            
            submitted = st.form_submit_button("🔮 Predict Life Expectancy", type="primary")
        
        if submitted:
            try:
                # Create a DataFrame with the input data
                input_data = pd.DataFrame({
                    'Adult Mortality': [adult_mortality],
                    'infant deaths': [infant_deaths],
                    'Alcohol': [alcohol],
                    'percentage expenditure': [percentage_expenditure],
                    'Hepatitis B': [hepatitis_b],
                    'Measles': [measles],
                    'BMI': [bmi],
                    'under-five deaths': [under_five_deaths],
                    'Polio': [polio],
                    'Total expenditure': [total_expenditure],
                    'Diphtheria': [diphtheria],
                    'HIV/AIDS': [hiv_aids],
                    'GDP': [gdp],
                    'Population': [population],
                    'thinness 1-19 years': [thinness_1_19],
                    'thinness 5-9 years': [thinness_5_9],
                    'Income composition of resources': [income_composition],
                    'Schooling': [schooling],
                    'Year': [year],
                    'Status': [status],
                    'Region': [region]
                })
                
                # Get the preprocessor from session state
                if 'preprocessor' in st.session_state:
                    preprocessor = st.session_state.preprocessor
                else:
                    # If preprocessor not in session state, create a new one
                    preprocessor = DataPreprocessor()
                
                # Apply the same preprocessing pipeline used during training
                # Step 1: Feature engineering
                input_engineered = preprocessor.feature_engineering(input_data)
                
                # Step 2: Encode categorical features
                input_encoded = preprocessor.encode_categorical_features(input_engineered)
                
                # Step 3: Handle missing values (if any)
                input_imputed = preprocessor.handle_missing_values(input_encoded)
                
                # Step 4: Prepare features for prediction (same as training)
                # Get the feature columns that were used during training
                feature_columns = [col for col in input_imputed.columns if col not in 
                                  ['Country', 'Year', 'Status', 'Region', 'Life expectancy']]
                
                # Add engineered features that might be missing
                if 'is_developed' not in input_imputed.columns:
                    input_imputed['is_developed'] = (input_imputed['Status'] == 'Developed').astype(int)
                
                if 'year_normalized' not in input_imputed.columns:
                    input_imputed['year_normalized'] = (input_imputed['Year'] - 2000) / 15
                
                # Create composite features if they don't exist
                if 'health_expenditure_per_capita' not in input_imputed.columns:
                    input_imputed['health_expenditure_per_capita'] = (
                        input_imputed['percentage expenditure'] / input_imputed['Population']
                    ).fillna(0)
                
                if 'total_mortality_rate' not in input_imputed.columns:
                    input_imputed['total_mortality_rate'] = (
                        input_imputed['Adult Mortality'] + input_imputed['infant deaths'] + 
                        input_imputed['under-five deaths']
                    )
                
                if 'vaccination_coverage' not in input_imputed.columns:
                    input_imputed['vaccination_coverage'] = (
                        input_imputed['Hepatitis B'] + input_imputed['Polio'] + input_imputed['Diphtheria']
                    ) / 3
                
                if 'thinness_composite' not in input_imputed.columns:
                    input_imputed['thinness_composite'] = (
                        input_imputed['thinness 1-19 years'] + input_imputed['thinness 5-9 years']
                    ) / 2
                
                if 'gdp_per_capita' not in input_imputed.columns:
                    input_imputed['gdp_per_capita'] = (
                        input_imputed['GDP'] / input_imputed['Population']
                    ).fillna(0)
                
                # Get the final feature columns (excluding target and categorical)
                final_feature_columns = [col for col in input_imputed.columns if col not in 
                                        ['Country', 'Year', 'Status', 'Region', 'Life expectancy']]
                
                # Select only the features that were used during training
                X_input = input_imputed[final_feature_columns].select_dtypes(include=[np.number])
                
                # Scale the features using the same scaler from training
                if hasattr(preprocessor, 'scaler') and preprocessor.is_fitted:
                    X_scaled = preprocessor.scaler.transform(X_input)
                else:
                    # If scaler not fitted, use the one from the model trainer
                    if hasattr(model_trainer, 'scaler'):
                        X_scaled = model_trainer.scaler.transform(X_input)
                    else:
                        # Fallback: use the features as-is
                        X_scaled = X_input.values
                
                # Make prediction using the best model
                best_model = model_trainer.best_model
                prediction = best_model.predict(X_scaled)[0]
                
                # Display debug information (can be removed in production)
                with st.expander("🔧 Debug Information"):
                    st.write(f"**Model Used**: {model_trainer.best_model_name}")
                    st.write(f"**Number of Features**: {X_scaled.shape[1]}")
                    st.write(f"**Feature Names**: {list(X_input.columns)}")
                    st.write(f"**Scaled Features Shape**: {X_scaled.shape}")
                
                # Display the prediction
                st.success(f"🎯 **Predicted Life Expectancy**: {prediction:.1f} years")
                
                # Add some context
                if prediction < 60:
                    st.warning("⚠️ Low life expectancy - may indicate health challenges")
                elif prediction > 80:
                    st.success("✅ High life expectancy - good health indicators")
                else:
                    st.info("ℹ️ Moderate life expectancy")
                
                # Show input summary
                st.subheader("Input Summary")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Health Indicators:**")
                    st.write(f"- Adult Mortality: {adult_mortality}")
                    st.write(f"- Infant Deaths: {infant_deaths}")
                    st.write(f"- Under-Five Deaths: {under_five_deaths}")
                    st.write(f"- BMI: {bmi}")
                    st.write(f"- HIV/AIDS: {hiv_aids}")
                
                with col2:
                    st.write("**Economic & Social:**")
                    st.write(f"- GDP: {gdp:,.0f}")
                    st.write(f"- Schooling: {schooling} years")
                    st.write(f"- Status: {status}")
                    st.write(f"- Region: {region}")
                    st.write(f"- Year: {year}")
                
            except Exception as e:
                st.error(f"❌ Error making prediction: {e}")
                st.info("Please ensure you have trained the models first and that all input values are valid.")
    
    # About page
    elif page == "📋 About":
        st.markdown("## About This Application")
        
        st.markdown("""
        ### Overview
        This Life Expectancy Prediction application uses machine learning to predict life expectancy 
        based on various health, economic, and social indicators from the World Health Organization (WHO) dataset.
        
        ### Features Implemented
        
        #### 1. Data Preprocessing
        - **Missing Value Handling**: Imputation strategies (median, mean, most frequent)
        - **Data Cleaning**: Handle inconsistencies, data type conversions
        - **Feature Engineering**: Create composite features, encode categorical variables
        - **Scaling**: Standardize numerical features
        
        #### 2. Comprehensive Outlier Detection
        - **Multiple Detection Methods**: Z-score, IQR, Modified Z-score, Isolation Forest
        - **Multivariate Analysis**: Isolation Forest, Local Outlier Factor, Elliptic Envelope
        - **Interactive Handling**: Choose detection methods and removal strategies
        - **Impact Analysis**: Assess the effect of outlier removal on data statistics
        - **Visualization**: Comprehensive plots showing outlier distribution and impact
        - **Detailed Information**: Show exactly what data was recognized as outliers
        
        #### 3. Model Training
        - **Multiple Algorithms**: Linear Regression, Ridge, Lasso, Elastic Net, Random Forest, Gradient Boosting
        - **Cross-Validation**: 5-fold cross-validation for robust evaluation
        - **Hyperparameter Tuning**: Grid search for optimal parameters
        - **Performance Comparison**: Side-by-side model evaluation
        
        #### 4. Model Evaluation
        - **Performance Metrics**: R², RMSE, MAE
        - **Model Comparison**: Side-by-side performance analysis
        - **Feature Importance**: Identify key predictors
        - **Residual Analysis**: Check model assumptions
        
        #### 5. Visualization
        - **Data Distribution**: Histograms and box plots
        - **Correlation Analysis**: Heatmaps and scatter plots
        - **Outlier Analysis**: Comprehensive outlier detection visualizations
        - **Model Performance**: Comparison charts and actual vs predicted plots
        - **Interactive Dashboard**: Plotly-based visualizations
        
        ### Technical Details
        
        #### Dataset
        - **Source**: World Health Organization (WHO)
        - **Size**: ~2,900 records
        - **Features**: 22 variables including health, economic, and social indicators
        - **Time Period**: 2000-2015
        - **Countries**: 193 countries
        
        #### Outlier Detection Methods
        1. **Z-score**: Standard deviation-based detection
        2. **IQR**: Interquartile range method
        3. **Modified Z-score**: Robust to extreme values
        4. **Isolation Forest**: Machine learning-based detection
        5. **Local Outlier Factor**: Density-based detection
        6. **Elliptic Envelope**: Statistical outlier detection
        
        #### Models Used
        1. **Linear Regression**: Baseline model
        2. **Ridge Regression**: L2 regularization for multicollinearity
        3. **Lasso Regression**: L1 regularization for feature selection
        4. **Elastic Net**: Combined L1 and L2 regularization
        5. **Random Forest**: Ensemble method for non-linear relationships
        6. **Gradient Boosting**: Advanced ensemble method
        
        #### Key Features
        - **Adult Mortality**: Deaths per 1000 population
        - **Infant Deaths**: Deaths per 1000 live births
        - **Alcohol**: Alcohol consumption per capita
        - **GDP**: Gross Domestic Product per capita
        - **Schooling**: Average years of schooling
        - **BMI**: Body Mass Index
        - **Vaccination Coverage**: Hepatitis B, Polio, Diphtheria
        - **Economic Indicators**: Income composition, expenditure
        
        ### Usage Instructions
        
        1. **Data Analysis**: Explore the dataset, check distributions and correlations
        2. **Outlier Detection**: Use comprehensive outlier detection with multiple methods
        3. **Outlier Details**: View detailed information about detected outliers
        4. **Model Training**: Configure preprocessing options and train models
        5. **Results**: View model performance and feature importance
        6. **Predictions**: Make predictions on new data (simplified interface)
        
        ### Technologies Used
        - **Python**: Core programming language
        - **Pandas**: Data manipulation and analysis
        - **NumPy**: Numerical computing
        - **Scikit-learn**: Machine learning algorithms
        - **Matplotlib/Seaborn**: Static visualizations
        - **Plotly**: Interactive visualizations
        - **Streamlit**: Web application framework
        
        ### Future Enhancements
        - Real-time data updates
        - More advanced models (Neural Networks, XGBoost)
        - Geographic visualization
        - Time series analysis
        - API integration for real-world predictions
        
        ### Contact
        This application was developed as a comprehensive machine learning project for life expectancy prediction.
        """)

if __name__ == "__main__":
    main() 