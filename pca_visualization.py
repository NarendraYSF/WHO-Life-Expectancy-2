import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns

class PCAVisualizer:
    """
    A comprehensive PCA visualization class with multiple plotting options.
    """
    
    def __init__(self):
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                      '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    def create_comprehensive_pca_dashboard(self, pca_analysis, target_values=None):
        """
        Create a comprehensive PCA dashboard with multiple visualizations.
        
        Args:
            pca_analysis (PCAAnalysis): Fitted PCA analysis object
            target_values (array): Target values for coloring
            
        Returns:
            dict: Dictionary containing all plot figures
        """
        dashboard = {}
        
        # 1. Scree Plot
        dashboard['scree_plot'] = pca_analysis.create_scree_plot()
        
        # 2. Component Loadings Heatmap
        dashboard['loadings_heatmap'] = pca_analysis.create_component_loadings_heatmap()
        
        # 3. 2D PCA Visualization
        dashboard['pca_2d'] = pca_analysis.create_2d_pca_visualization(target_values)
        
        # 4. Feature Importance Plot
        dashboard['feature_importance'], dashboard['importance_df'] = pca_analysis.create_feature_importance_plot()
        
        # 5. Component Interpretation Table
        dashboard['component_interpretation'] = self.create_component_interpretation_table(pca_analysis)
        
        # 6. Variance Explained Summary
        dashboard['variance_summary'] = self.create_variance_summary_plot(pca_analysis)
        
        return dashboard
    
    def create_component_interpretation_table(self, pca_analysis, top_features=5):
        """
        Create a table showing interpretation of each PCA component.
        
        Args:
            pca_analysis (PCAAnalysis): Fitted PCA analysis object
            top_features (int): Number of top features per component
            
        Returns:
            plotly.graph_objects.Figure: Component interpretation table
        """
        if pca_analysis.pca is None:
            return None
        
        try:
            n_components = pca_analysis.analysis_results['n_components']
            explained_variance = pca_analysis.analysis_results['explained_variance_ratio']
            loadings = pca_analysis.analysis_results['loadings']
            feature_names = pca_analysis.analysis_results['feature_names']
            
            # Create table data
            table_data = []
            for i in range(n_components):
                try:
                    # Get top contributing features for this component
                    component_loadings = loadings[i]
                    
                    # Create pairs of (feature_name, loading_value)
                    feature_loading_pairs = list(zip(feature_names, component_loadings))
                    
                    # Sort by absolute loading value
                    feature_loading_pairs.sort(key=lambda x: abs(x[1]), reverse=True)
                    
                    # Take top features
                    top_features_list = feature_loading_pairs[:top_features]
                    
                    # Create feature string with better formatting
                    feature_strings = []
                    for name, loading in top_features_list:
                        sign = "+" if loading >= 0 else ""
                        feature_strings.append(f"{name}: {sign}{loading:.3f}")
                    
                    feature_string = "<br>".join(feature_strings)
                    
                except Exception as e:
                    # Fallback if there's an error getting features
                    feature_string = f"Error: {str(e)}"
                
                table_data.append({
                    'Component': f'PC{i+1}',
                    'Explained Variance': f'{explained_variance[i]:.3f}',
                    'Top Features': feature_string
                })
            
            # Create table with better styling
            fig = go.Figure(data=[go.Table(
                header=dict(
                    values=['Component', 'Explained Variance', 'Top Contributing Features'],
                    fill_color='#1f77b4',
                    align='left',
                    font=dict(size=14, color='white', weight='bold')
                ),
                cells=dict(
                    values=[[row['Component'] for row in table_data],
                           [row['Explained Variance'] for row in table_data],
                           [row['Top Features'] for row in table_data]],
                    fill_color=[['#f0f2f6', '#ffffff'] * (len(table_data) // 2 + 1)][:len(table_data)],
                    align='left',
                    font=dict(size=11, color='black'),
                    height=35
                )
            )])
            
            fig.update_layout(
                title="PCA Component Interpretation",
                height=min(400, 50 + len(table_data) * 35),
                margin=dict(l=20, r=20, t=40, b=20)
            )
            
            return fig
            
        except Exception as e:
            # Return a simple error table if something goes wrong
            fig = go.Figure(data=[go.Table(
                header=dict(
                    values=['Error'],
                    fill_color='#ff6b6b',
                    align='center',
                    font=dict(size=14, color='white')
                ),
                cells=dict(
                    values=[[f"Could not create component interpretation table: {str(e)}"]],
                    fill_color='white',
                    align='center',
                    font=dict(size=12, color='black')
                )
            )])
            
            fig.update_layout(
                title="PCA Component Interpretation - Error",
                height=200
            )
            
            return fig
    
    def create_variance_summary_plot(self, pca_analysis):
        """
        Create a summary plot showing variance explained by PCA.
        
        Args:
            pca_analysis (PCAAnalysis): Fitted PCA analysis object
            
        Returns:
            plotly.graph_objects.Figure: Variance summary plot
        """
        if pca_analysis.pca is None:
            return None
        
        explained_variance = pca_analysis.analysis_results['explained_variance_ratio']
        cumulative_variance = pca_analysis.analysis_results['cumulative_variance']
        
        # Create subplot
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Variance Explained by Component', 'Cumulative Variance'),
            specs=[[{"type": "pie"}, {"type": "bar"}]]
        )
        
        # Pie chart for individual components
        components = [f'PC{i+1}' for i in range(len(explained_variance))]
        fig.add_trace(
            go.Pie(
                labels=components,
                values=explained_variance,
                name="Individual Variance",
                textinfo='label+percent',
                textposition='inside'
            ),
            row=1, col=1
        )
        
        # Bar chart for cumulative variance
        fig.add_trace(
            go.Bar(
                x=components,
                y=cumulative_variance,
                name="Cumulative Variance",
                text=[f'{v:.1%}' for v in cumulative_variance],
                textposition='auto',
                marker_color='lightgreen'
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title="PCA Variance Summary",
            height=500,
            showlegend=False
        )
        
        return fig
    
    def create_3d_pca_visualization(self, pca_analysis, target_values=None):
        """
        Create 3D visualization of PCA results.
        
        Args:
            pca_analysis (PCAAnalysis): Fitted PCA analysis object
            target_values (array): Target values for coloring
            
        Returns:
            plotly.graph_objects.Figure: 3D PCA plot
        """
        if pca_analysis.pca is None:
            return None
        
        X_pca = pca_analysis.analysis_results['X_pca']
        
        if X_pca.shape[1] < 3:
            return None
        
        # Create DataFrame for plotting
        plot_data = pd.DataFrame({
            'PC1': X_pca[:, 0],
            'PC2': X_pca[:, 1],
            'PC3': X_pca[:, 2]
        })
        
        if target_values is not None:
            plot_data['Life Expectancy'] = target_values
            
            fig = px.scatter_3d(
                plot_data,
                x='PC1',
                y='PC2',
                z='PC3',
                color='Life Expectancy',
                title="PCA 3D Visualization - Colored by Life Expectancy",
                color_continuous_scale='viridis'
            )
        else:
            fig = px.scatter_3d(
                plot_data,
                x='PC1',
                y='PC2',
                z='PC3',
                title="PCA 3D Visualization",
                color_discrete_sequence=['blue']
            )
        
        # Add explained variance to axis labels
        explained_var = pca_analysis.analysis_results['explained_variance_ratio']
        fig.update_layout(
            scene=dict(
                xaxis_title=f"PC1 ({explained_var[0]:.1%} variance)",
                yaxis_title=f"PC2 ({explained_var[1]:.1%} variance)",
                zaxis_title=f"PC3 ({explained_var[2]:.1%} variance)"
            )
        )
        
        return fig
    
    def create_biplot(self, pca_analysis, target_values=None, top_features=10):
        """
        Create a biplot showing both data points and feature vectors.
        
        Args:
            pca_analysis (PCAAnalysis): Fitted PCA analysis object
            target_values (array): Target values for coloring
            top_features (int): Number of top features to show as vectors
            
        Returns:
            plotly.graph_objects.Figure: Biplot
        """
        if pca_analysis.pca is None:
            return None
        
        X_pca = pca_analysis.analysis_results['X_pca']
        loadings = pca_analysis.analysis_results['loadings']
        feature_names = pca_analysis.analysis_results['feature_names']
        
        if X_pca.shape[1] < 2:
            return None
        
        # Create base scatter plot
        plot_data = pd.DataFrame({
            'PC1': X_pca[:, 0],
            'PC2': X_pca[:, 1]
        })
        
        if target_values is not None:
            plot_data['Life Expectancy'] = target_values
            fig = px.scatter(
                plot_data,
                x='PC1',
                y='PC2',
                color='Life Expectancy',
                title="PCA Biplot - Data Points and Feature Vectors",
                color_continuous_scale='viridis'
            )
        else:
            fig = px.scatter(
                plot_data,
                x='PC1',
                y='PC2',
                title="PCA Biplot - Data Points and Feature Vectors",
                color_discrete_sequence=['blue']
            )
        
        # Add feature vectors
        # Get top features by importance
        feature_importance = np.sum(loadings ** 2, axis=0)
        top_indices = np.argsort(feature_importance)[-top_features:]
        
        # Scale factor for vectors
        scale_factor = 3
        
        for idx in top_indices:
            # Get loading values for PC1 and PC2
            loading_pc1 = loadings[0, idx] * scale_factor
            loading_pc2 = loadings[1, idx] * scale_factor
            
            # Add arrow
            fig.add_annotation(
                x=loading_pc1,
                y=loading_pc2,
                xref="x",
                yref="y",
                text=feature_names[idx],
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor="red",
                ax=0,
                ay=-40,
                font=dict(size=10, color="red")
            )
        
        # Add explained variance to axis labels
        explained_var = pca_analysis.analysis_results['explained_variance_ratio']
        fig.update_xaxes(title_text=f"PC1 ({explained_var[0]:.1%} variance)")
        fig.update_yaxes(title_text=f"PC2 ({explained_var[1]:.1%} variance)")
        
        return fig
    
    def create_correlation_heatmap_pca(self, pca_analysis):
        """
        Create a correlation heatmap between original features and PCA components.
        
        Args:
            pca_analysis (PCAAnalysis): Fitted PCA analysis object
            
        Returns:
            plotly.graph_objects.Figure: Correlation heatmap
        """
        if pca_analysis.pca is None:
            return None
        
        loadings = pca_analysis.analysis_results['loadings']
        feature_names = pca_analysis.analysis_results['feature_names']
        
        # Create correlation matrix
        n_components = loadings.shape[0]
        n_features = loadings.shape[1]
        
        # Create heatmap
        fig = px.imshow(
            loadings,
            x=feature_names,
            y=[f'PC{i+1}' for i in range(n_components)],
            color_continuous_scale='RdBu',
            aspect='auto',
            title="Correlation between Original Features and PCA Components"
        )
        
        # Add annotations
        annotations = []
        for i in range(n_components):
            for j in range(n_features):
                annotations.append(
                    dict(
                        x=j,
                        y=i,
                        text=f'{loadings[i, j]:.3f}',
                        showarrow=False,
                        font=dict(size=8)
                    )
                )
        
        fig.update_layout(
            annotations=annotations,
            height=500,
            xaxis_title="Original Features",
            yaxis_title="PCA Components"
        )
        
        return fig
    
    def create_pca_performance_comparison(self, original_results, pca_results):
        """
        Create a comparison plot between original features and PCA features performance.
        
        Args:
            original_results (dict): Results from models trained on original features
            pca_results (dict): Results from models trained on PCA features
            
        Returns:
            plotly.graph_objects.Figure: Performance comparison plot
        """
        # Extract model names and metrics
        model_names = list(original_results.keys())
        
        # Prepare data for comparison
        comparison_data = []
        for model_name in model_names:
            if model_name in original_results and model_name in pca_results:
                comparison_data.append({
                    'Model': model_name,
                    'Original R²': original_results[model_name]['test_r2'],
                    'PCA R²': pca_results[model_name]['test_r2'],
                    'Original RMSE': original_results[model_name]['test_rmse'],
                    'PCA RMSE': pca_results[model_name]['test_rmse']
                })
        
        if not comparison_data:
            return None
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Create subplots
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('R² Score Comparison', 'RMSE Comparison'),
            specs=[[{"type": "bar"}, {"type": "bar"}]]
        )
        
        # R² comparison
        fig.add_trace(
            go.Bar(
                x=comparison_df['Model'],
                y=comparison_df['Original R²'],
                name='Original Features',
                marker_color='lightblue'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(
                x=comparison_df['Model'],
                y=comparison_df['PCA R²'],
                name='PCA Features',
                marker_color='lightcoral'
            ),
            row=1, col=1
        )
        
        # RMSE comparison
        fig.add_trace(
            go.Bar(
                x=comparison_df['Model'],
                y=comparison_df['Original RMSE'],
                name='Original Features',
                marker_color='lightblue',
                showlegend=False
            ),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Bar(
                x=comparison_df['Model'],
                y=comparison_df['PCA RMSE'],
                name='PCA Features',
                marker_color='lightcoral',
                showlegend=False
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title="Model Performance: Original vs PCA Features",
            height=500,
            barmode='group'
        )
        
        return fig 