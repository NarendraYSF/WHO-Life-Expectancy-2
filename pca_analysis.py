import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class PCAAnalysis:
    """
    A comprehensive PCA analysis class for life expectancy prediction.
    Includes basic PCA analysis, visualization, and feature selection capabilities.
    """
    
    def __init__(self):
        self.pca = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.analysis_results = {}
        
    def prepare_data_for_pca(self, df, target_column='Life expectancy'):
        """
        Prepare data for PCA analysis by selecting numerical features and scaling.
        
        Args:
            df (pd.DataFrame): Input dataframe
            target_column (str): Name of the target column to exclude
            
        Returns:
            tuple: (X_scaled, feature_names, target_values)
        """
        # Select numerical features excluding target and categorical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in numerical_cols if col != target_column and col not in 
                       ['Year', 'is_developed']]  # Exclude year and binary features
        
        # Prepare feature matrix
        X = df[feature_cols].dropna()
        y = df[target_column].dropna()
        
        # Align data
        common_indices = X.index.intersection(y.index)
        X = X.loc[common_indices]
        y = y.loc[common_indices]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        self.feature_names = feature_cols
        
        return X_scaled, feature_cols, y.values
    
    def perform_pca_analysis(self, X, n_components=None, explained_variance_threshold=0.95):
        """
        Perform PCA analysis on the data.
        
        Args:
            X (array): Scaled feature matrix
            n_components (int): Number of components (None for auto)
            explained_variance_threshold (float): Threshold for explained variance
            
        Returns:
            dict: PCA analysis results
        """
        # Determine number of components if not specified
        if n_components is None:
            # Use enough components to explain the threshold variance
            temp_pca = PCA()
            temp_pca.fit(X)
            cumulative_variance = np.cumsum(temp_pca.explained_variance_ratio_)
            n_components = np.argmax(cumulative_variance >= explained_variance_threshold) + 1
            n_components = max(2, min(n_components, X.shape[1]))  # At least 2, at most all features
        
        # Perform PCA
        self.pca = PCA(n_components=n_components)
        X_pca = self.pca.fit_transform(X)
        
        # Calculate metrics
        explained_variance_ratio = self.pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance_ratio)
        total_variance_explained = cumulative_variance[-1]
        
        # Store results
        self.analysis_results = {
            'n_components': n_components,
            'explained_variance_ratio': explained_variance_ratio,
            'cumulative_variance': cumulative_variance,
            'total_variance_explained': total_variance_explained,
            'X_pca': X_pca,
            'X_original': X,
            'feature_names': self.feature_names,
            'loadings': self.pca.components_,
            'singular_values': self.pca.singular_values_
        }
        
        return self.analysis_results
    
    def create_scree_plot(self):
        """
        Create a scree plot showing explained variance ratio.
        
        Returns:
            plotly.graph_objects.Figure: Scree plot
        """
        if self.pca is None:
            return None
        
        explained_variance_ratio = self.analysis_results['explained_variance_ratio']
        cumulative_variance = self.analysis_results['cumulative_variance']
        
        # Create subplot with both individual and cumulative variance
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Individual Explained Variance', 'Cumulative Explained Variance'),
            specs=[[{"type": "bar"}, {"type": "scatter"}]]
        )
        
        # Individual explained variance (bar plot)
        components = list(range(1, len(explained_variance_ratio) + 1))
        fig.add_trace(
            go.Bar(
                x=components,
                y=explained_variance_ratio,
                name='Individual Variance',
                marker_color='lightblue',
                text=[f'{v:.3f}' for v in explained_variance_ratio],
                textposition='auto'
            ),
            row=1, col=1
        )
        
        # Cumulative explained variance (line plot)
        fig.add_trace(
            go.Scatter(
                x=components,
                y=cumulative_variance,
                mode='lines+markers',
                name='Cumulative Variance',
                line=dict(color='red', width=3),
                marker=dict(size=8)
            ),
            row=1, col=2
        )
        
        # Add threshold line
        threshold = 0.95
        fig.add_hline(
            y=threshold,
            line_dash="dash",
            line_color="orange",
            annotation_text=f"95% Threshold",
            row=1, col=2
        )
        
        # Update layout
        fig.update_layout(
            title="PCA Scree Plot - Explained Variance Analysis",
            height=500,
            showlegend=True
        )
        
        fig.update_xaxes(title_text="Principal Components", row=1, col=1)
        fig.update_xaxes(title_text="Principal Components", row=1, col=2)
        fig.update_yaxes(title_text="Explained Variance Ratio", row=1, col=1)
        fig.update_yaxes(title_text="Cumulative Explained Variance", row=1, col=2)
        
        return fig
    
    def create_component_loadings_heatmap(self, top_features=10):
        """
        Create a heatmap showing component loadings.
        
        Args:
            top_features (int): Number of top features to show per component
            
        Returns:
            plotly.graph_objects.Figure: Loadings heatmap
        """
        if self.pca is None:
            return None
        
        loadings = self.analysis_results['loadings']
        feature_names = self.analysis_results['feature_names']
        
        # Get top features for each component
        top_loadings = []
        top_feature_names = []
        
        for i, component in enumerate(loadings):
            # Get indices of top features by absolute loading value
            top_indices = np.argsort(np.abs(component))[-top_features:]
            
            # Sort by absolute loading value
            top_indices = sorted(top_indices, key=lambda x: abs(component[x]), reverse=True)
            
            top_loadings.append(component[top_indices])
            top_feature_names.append([feature_names[j] for j in top_indices])
        
        # Create heatmap data
        heatmap_data = np.array(top_loadings).T  # Transpose for better visualization
        
        # Create heatmap
        fig = px.imshow(
            heatmap_data,
            x=[f'PC{i+1}' for i in range(len(loadings))],
            y=[f'Top {i+1}' for i in range(top_features)],
            color_continuous_scale='RdBu',
            aspect='auto',
            title="PCA Component Loadings Heatmap (Top Features)"
        )
        
        # Add annotations
        annotations = []
        for i in range(heatmap_data.shape[0]):
            for j in range(heatmap_data.shape[1]):
                annotations.append(
                    dict(
                        x=j,
                        y=i,
                        text=f'{heatmap_data[i, j]:.3f}',
                        showarrow=False,
                        font=dict(size=10)
                    )
                )
        
        fig.update_layout(
            annotations=annotations,
            height=600
        )
        
        return fig
    
    def create_2d_pca_visualization(self, target_values=None, color_by='target'):
        """
        Create 2D visualization of PCA results.
        
        Args:
            target_values (array): Target values for coloring
            color_by (str): What to color by ('target', 'component')
            
        Returns:
            plotly.graph_objects.Figure: 2D PCA plot
        """
        if self.pca is None:
            return None
        
        X_pca = self.analysis_results['X_pca']
        
        if X_pca.shape[1] < 2:
            return None
        
        # Create DataFrame for plotting
        plot_data = pd.DataFrame({
            'PC1': X_pca[:, 0],
            'PC2': X_pca[:, 1]
        })
        
        if target_values is not None and color_by == 'target':
            plot_data['Life Expectancy'] = target_values
            
            fig = px.scatter(
                plot_data,
                x='PC1',
                y='PC2',
                color='Life Expectancy',
                title="PCA 2D Visualization - Colored by Life Expectancy",
                color_continuous_scale='viridis',
                hover_data=['Life Expectancy']
            )
        else:
            fig = px.scatter(
                plot_data,
                x='PC1',
                y='PC2',
                title="PCA 2D Visualization",
                color_discrete_sequence=['blue']
            )
        
        # Add explained variance to axis labels
        explained_var = self.analysis_results['explained_variance_ratio']
        fig.update_xaxes(title_text=f"PC1 ({explained_var[0]:.1%} variance)")
        fig.update_yaxes(title_text=f"PC2 ({explained_var[1]:.1%} variance)")
        
        return fig
    
    def create_feature_importance_plot(self, top_features=15):
        """
        Create a plot showing feature importance based on PCA loadings.
        
        Args:
            top_features (int): Number of top features to show
            
        Returns:
            plotly.graph_objects.Figure: Feature importance plot
        """
        if self.pca is None:
            return None
        
        loadings = self.analysis_results['loadings']
        feature_names = self.analysis_results['feature_names']
        
        # Calculate feature importance as sum of squared loadings across components
        feature_importance = np.sum(loadings ** 2, axis=0)
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importance
        }).sort_values('Importance', ascending=False)
        
        # Select top features
        top_importance = importance_df.head(top_features)
        
        # Create bar plot
        fig = px.bar(
            top_importance,
            x='Importance',
            y='Feature',
            orientation='h',
            title=f"Feature Importance in PCA (Top {top_features})",
            color='Importance',
            color_continuous_scale='viridis'
        )
        
        fig.update_layout(height=600)
        
        return fig, importance_df
    
    def get_pca_summary(self):
        """
        Get a summary of PCA analysis results.
        
        Returns:
            dict: Summary statistics
        """
        if self.pca is None:
            return None
        
        explained_variance_ratio = self.analysis_results['explained_variance_ratio']
        cumulative_variance = self.analysis_results['cumulative_variance']
        
        summary = {
            'n_components': self.analysis_results['n_components'],
            'total_variance_explained': self.analysis_results['total_variance_explained'],
            'variance_explained_per_component': explained_variance_ratio.tolist(),
            'cumulative_variance': cumulative_variance.tolist(),
            'reduction_ratio': 1 - (self.analysis_results['n_components'] / len(self.feature_names)),
            'feature_names': self.feature_names
        }
        
        return summary
    
    def get_top_contributing_features(self, component_idx=0, top_n=5):
        """
        Get top contributing features for a specific component.
        
        Args:
            component_idx (int): Index of the component (0-based)
            top_n (int): Number of top features to return
            
        Returns:
            list: List of tuples (feature_name, loading_value)
        """
        if self.pca is None or component_idx >= self.analysis_results['n_components']:
            return []
        
        loadings = self.analysis_results['loadings'][component_idx]
        feature_names = self.analysis_results['feature_names']
        
        # Get indices of top features by absolute loading value
        top_indices = np.argsort(np.abs(loadings))[-top_n:]
        
        # Sort by absolute loading value
        top_indices = sorted(top_indices, key=lambda x: abs(loadings[x]), reverse=True)
        
        top_features = [(feature_names[i], loadings[i]) for i in top_indices]
        
        return top_features
    
    def transform_data(self, X):
        """
        Transform new data using fitted PCA.
        
        Args:
            X (array): New data to transform
            
        Returns:
            array: Transformed data
        """
        if self.pca is None:
            raise ValueError("PCA has not been fitted yet. Call perform_pca_analysis first.")
        
        # Scale the data using the fitted scaler
        X_scaled = self.scaler.transform(X)
        
        # Transform using PCA
        X_pca = self.pca.transform(X_scaled)
        
        return X_pca
    
    def inverse_transform(self, X_pca):
        """
        Transform PCA data back to original feature space.
        
        Args:
            X_pca (array): PCA transformed data
            
        Returns:
            array: Data in original feature space
        """
        if self.pca is None:
            raise ValueError("PCA has not been fitted yet. Call perform_pca_analysis first.")
        
        # Inverse transform using PCA
        X_scaled = self.pca.inverse_transform(X_pca)
        
        # Inverse scale using the fitted scaler
        X_original = self.scaler.inverse_transform(X_scaled)
        
        return X_original 