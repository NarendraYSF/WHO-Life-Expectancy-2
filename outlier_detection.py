import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.cluster import DBSCAN
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

class OutlierDetectionSystem:
    """
    A comprehensive outlier detection system with multiple detection methods,
    visualization capabilities, and interactive handling options.
    """
    
    def __init__(self):
        self.outlier_methods = {}
        self.outlier_results = {}
        self.feature_outliers = {}
        self.multivariate_outliers = {}
        
    def detect_univariate_outliers(self, data, method='zscore', threshold=3, **kwargs):
        """
        Detect univariate outliers using various statistical methods.
        
        Args:
            data (array-like): Input data
            method (str): Detection method ('zscore', 'iqr', 'modified_zscore', 'isolation_forest')
            threshold (float): Threshold for outlier detection
            **kwargs: Additional parameters for specific methods
            
        Returns:
            dict: Outlier detection results
        """
        data = np.array(data)
        results = {
            'method': method,
            'threshold': threshold,
            'outlier_indices': [],
            'outlier_scores': [],
            'outlier_count': 0,
            'outlier_percentage': 0
        }
        
        if method == 'zscore':
            z_scores = np.abs(stats.zscore(data, nan_policy='omit'))
            outlier_mask = z_scores > threshold
            outlier_scores = z_scores
            
        elif method == 'modified_zscore':
            median = np.nanmedian(data)
            mad = np.nanmedian(np.abs(data - median))
            modified_z_scores = 0.6745 * (data - median) / mad
            outlier_mask = np.abs(modified_z_scores) > threshold
            outlier_scores = np.abs(modified_z_scores)
            
        elif method == 'iqr':
            Q1 = np.nanpercentile(data, 25)
            Q3 = np.nanpercentile(data, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outlier_mask = (data < lower_bound) | (data > upper_bound)
            outlier_scores = np.abs(data - np.nanmedian(data)) / IQR
            
        elif method == 'isolation_forest':
            # Reshape data for sklearn
            X = data.reshape(-1, 1)
            iso_forest = IsolationForest(
                contamination=kwargs.get('contamination', 0.1),
                random_state=kwargs.get('random_state', 42)
            )
            outlier_labels = iso_forest.fit_predict(X)
            outlier_mask = outlier_labels == -1
            outlier_scores = -iso_forest.score_samples(X)
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        results['outlier_indices'] = np.where(outlier_mask)[0]
        results['outlier_scores'] = outlier_scores
        results['outlier_count'] = len(results['outlier_indices'])
        results['outlier_percentage'] = (results['outlier_count'] / len(data)) * 100
        
        return results
    
    def detect_multivariate_outliers(self, X, method='isolation_forest', **kwargs):
        """
        Detect multivariate outliers using various methods.
        
        Args:
            X (array-like): Feature matrix
            method (str): Detection method ('isolation_forest', 'local_outlier_factor', 'elliptic_envelope', 'dbscan')
            **kwargs: Additional parameters for specific methods
            
        Returns:
            dict: Multivariate outlier detection results
        """
        X = np.array(X)
        results = {
            'method': method,
            'outlier_indices': [],
            'outlier_scores': [],
            'outlier_count': 0,
            'outlier_percentage': 0
        }
        
        if method == 'isolation_forest':
            iso_forest = IsolationForest(
                contamination=kwargs.get('contamination', 0.1),
                random_state=kwargs.get('random_state', 42)
            )
            outlier_labels = iso_forest.fit_predict(X)
            outlier_scores = -iso_forest.score_samples(X)
            
        elif method == 'local_outlier_factor':
            lof = LocalOutlierFactor(
                contamination=kwargs.get('contamination', 0.1),
                n_neighbors=kwargs.get('n_neighbors', 20)
            )
            outlier_labels = lof.fit_predict(X)
            outlier_scores = -lof.negative_outlier_factor_
            
        elif method == 'elliptic_envelope':
            envelope = EllipticEnvelope(
                contamination=kwargs.get('contamination', 0.1),
                random_state=kwargs.get('random_state', 42)
            )
            outlier_labels = envelope.fit_predict(X)
            outlier_scores = -envelope.score_samples(X)
            
        elif method == 'dbscan':
            dbscan = DBSCAN(
                eps=kwargs.get('eps', 0.5),
                min_samples=kwargs.get('min_samples', 5)
            )
            outlier_labels = dbscan.fit_predict(X)
            outlier_scores = np.zeros(len(X))
            outlier_scores[outlier_labels == -1] = 1
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        results['outlier_indices'] = np.where(outlier_labels == -1)[0]
        results['outlier_scores'] = outlier_scores
        results['outlier_count'] = len(results['outlier_indices'])
        results['outlier_percentage'] = (results['outlier_count'] / len(X)) * 100
        
        return results
    
    def comprehensive_outlier_analysis(self, df, target_column='Life expectancy', 
                                     numerical_columns=None, methods=None):
        """
        Perform comprehensive outlier analysis on the dataset.
        
        Args:
            df (pd.DataFrame): Input dataframe
            target_column (str): Name of the target column
            numerical_columns (list): List of numerical columns to analyze
            methods (list): List of detection methods to use
            
        Returns:
            dict: Comprehensive outlier analysis results
        """
        if numerical_columns is None:
            numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if methods is None:
            methods = ['zscore', 'iqr', 'modified_zscore', 'isolation_forest']
        
        print("=== Comprehensive Outlier Analysis ===")
        
        # Initialize results
        analysis_results = {
            'target_outliers': {},
            'feature_outliers': {},
            'multivariate_outliers': {},
            'summary': {}
        }
        
        # 1. Target variable outlier analysis
        print(f"\n1. Analyzing target variable: {target_column}")
        target_data = df[target_column].dropna()
        
        for method in methods:
            try:
                results = self.detect_univariate_outliers(target_data, method=method)
                analysis_results['target_outliers'][method] = results
                print(f"   {method}: {results['outlier_count']} outliers ({results['outlier_percentage']:.2f}%)")
            except Exception as e:
                print(f"   {method}: Error - {e}")
        
        # 2. Feature-wise outlier analysis
        print(f"\n2. Analyzing individual features")
        feature_columns = [col for col in numerical_columns if col != target_column]
        
        for feature in feature_columns[:10]:  # Limit to first 10 features for performance
            feature_data = df[feature].dropna()
            if len(feature_data) > 0:
                analysis_results['feature_outliers'][feature] = {}
                
                for method in methods[:2]:  # Use first 2 methods for features
                    try:
                        results = self.detect_univariate_outliers(feature_data, method=method)
                        analysis_results['feature_outliers'][feature][method] = results
                    except Exception as e:
                        print(f"   {feature} - {method}: Error - {e}")
        
        # 3. Multivariate outlier analysis
        print(f"\n3. Analyzing multivariate outliers")
        X = df[numerical_columns].dropna()
        
        multivariate_methods = ['isolation_forest', 'local_outlier_factor']
        for method in multivariate_methods:
            try:
                results = self.detect_multivariate_outliers(X, method=method)
                analysis_results['multivariate_outliers'][method] = results
                print(f"   {method}: {results['outlier_count']} outliers ({results['outlier_percentage']:.2f}%)")
            except Exception as e:
                print(f"   {method}: Error - {e}")
        
        # 4. Summary statistics
        analysis_results['summary'] = self._generate_outlier_summary(analysis_results)
        
        return analysis_results
    
    def _generate_outlier_summary(self, analysis_results):
        """Generate summary statistics for outlier analysis."""
        summary = {
            'total_features_analyzed': len(analysis_results['feature_outliers']),
            'methods_used': list(analysis_results['target_outliers'].keys()),
            'target_outlier_summary': {},
            'feature_outlier_summary': {},
            'recommendations': []
        }
        
        # Target outlier summary
        for method, results in analysis_results['target_outliers'].items():
            summary['target_outlier_summary'][method] = {
                'count': results['outlier_count'],
                'percentage': results['outlier_percentage']
            }
        
        # Feature outlier summary
        feature_summary = {}
        for feature, methods in analysis_results['feature_outliers'].items():
            feature_summary[feature] = {}
            for method, results in methods.items():
                feature_summary[feature][method] = {
                    'count': results['outlier_count'],
                    'percentage': results['outlier_percentage']
                }
        summary['feature_outlier_summary'] = feature_summary
        
        # Generate recommendations
        recommendations = []
        
        # Check for high outlier percentages
        for method, results in analysis_results['target_outliers'].items():
            if results['outlier_percentage'] > 10:
                recommendations.append(f"High outlier percentage ({results['outlier_percentage']:.1f}%) detected in target using {method}")
        
        # Check for features with many outliers
        for feature, methods in analysis_results['feature_outliers'].items():
            for method, results in methods.items():
                if results['outlier_percentage'] > 15:
                    recommendations.append(f"Feature '{feature}' has high outlier percentage ({results['outlier_percentage']:.1f}%) using {method}")
        
        if not recommendations:
            recommendations.append("No significant outlier issues detected")
        
        summary['recommendations'] = recommendations
        
        return summary
    
    def visualize_outlier_analysis(self, df, analysis_results, target_column='Life expectancy'):
        """
        Create comprehensive visualizations for outlier analysis.
        
        Args:
            df (pd.DataFrame): Input dataframe
            analysis_results (dict): Results from comprehensive_outlier_analysis
            target_column (str): Name of the target column
        """
        print("\n=== Generating Outlier Visualizations ===")
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Target Variable Distribution', 'Target Outlier Scores',
                'Feature Outlier Summary', 'Multivariate Outlier Analysis',
                'Outlier Detection Comparison', 'Outlier Impact Analysis'
            ),
            specs=[[{"type": "histogram"}, {"type": "scatter"}],
                   [{"type": "bar"}, {"type": "scatter"}],
                   [{"type": "bar"}, {"type": "scatter"}]]
        )
        
        # 1. Target variable distribution with outliers
        target_data = df[target_column].dropna()
        fig.add_trace(
            go.Histogram(x=target_data, name='All Data', nbinsx=30),
            row=1, col=1
        )
        
        # Highlight outliers from different methods
        colors = ['red', 'orange', 'purple', 'brown']
        for i, (method, results) in enumerate(analysis_results['target_outliers'].items()):
            if len(results['outlier_indices']) > 0:
                outlier_data = target_data.iloc[results['outlier_indices']]
                fig.add_trace(
                    go.Histogram(x=outlier_data, name=f'{method} Outliers', 
                               marker_color=colors[i % len(colors)]),
                    row=1, col=1
                )
        
        # 2. Target outlier scores
        for method, results in analysis_results['target_outliers'].items():
            fig.add_trace(
                go.Scatter(x=list(range(len(target_data))), y=results['outlier_scores'],
                          mode='markers', name=f'{method} Scores',
                          marker=dict(size=4)),
                row=1, col=2
            )
        
        # 3. Feature outlier summary
        feature_names = list(analysis_results['feature_outliers'].keys())[:10]
        outlier_counts = []
        for feature in feature_names:
            if 'zscore' in analysis_results['feature_outliers'][feature]:
                count = analysis_results['feature_outliers'][feature]['zscore']['outlier_count']
                outlier_counts.append(count)
        
        fig.add_trace(
            go.Bar(x=feature_names, y=outlier_counts, name='Outlier Count'),
            row=2, col=1
        )
        
        # 4. Multivariate outlier analysis
        for method, results in analysis_results['multivariate_outliers'].items():
            if len(results['outlier_indices']) > 0:
                # Use first two features for 2D visualization
                X = df[df.select_dtypes(include=[np.number]).columns[:2]].dropna()
                fig.add_trace(
                    go.Scatter(x=X.iloc[:, 0], y=X.iloc[:, 1], mode='markers',
                              name=f'{method} Normal', marker=dict(color='blue', size=4)),
                    row=2, col=2
                )
                
                outlier_X = X.iloc[results['outlier_indices']]
                fig.add_trace(
                    go.Scatter(x=outlier_X.iloc[:, 0], y=outlier_X.iloc[:, 1], 
                              mode='markers', name=f'{method} Outliers',
                              marker=dict(color='red', size=6)),
                    row=2, col=2
                )
        
        # 5. Outlier detection comparison
        methods = list(analysis_results['target_outliers'].keys())
        outlier_percentages = [analysis_results['target_outliers'][m]['outlier_percentage'] 
                              for m in methods]
        
        fig.add_trace(
            go.Bar(x=methods, y=outlier_percentages, name='Outlier Percentage'),
            row=3, col=1
        )
        
        # 6. Outlier impact analysis (before/after removal)
        target_data = df[target_column].dropna()
        
        # Calculate statistics before outlier removal
        stats_before = {
            'mean': np.mean(target_data),
            'std': np.std(target_data),
            'median': np.median(target_data)
        }
        
        # Calculate statistics after removing outliers (using most conservative method)
        best_method = min(analysis_results['target_outliers'].keys(), 
                         key=lambda x: analysis_results['target_outliers'][x]['outlier_count'])
        outlier_indices = analysis_results['target_outliers'][best_method]['outlier_indices']
        clean_data = np.delete(target_data, outlier_indices)
        
        stats_after = {
            'mean': np.mean(clean_data),
            'std': np.std(clean_data),
            'median': np.median(clean_data)
        }
        
        # Plot impact
        metrics = ['mean', 'std', 'median']
        before_values = [stats_before[m] for m in metrics]
        after_values = [stats_after[m] for m in metrics]
        
        fig.add_trace(
            go.Bar(x=metrics, y=before_values, name='Before Outlier Removal'),
            row=3, col=2
        )
        fig.add_trace(
            go.Bar(x=metrics, y=after_values, name='After Outlier Removal'),
            row=3, col=2
        )
        
        # Update layout
        fig.update_layout(
            title_text="Comprehensive Outlier Analysis Dashboard",
            showlegend=True,
            height=1200
        )
        
        return fig
    
    def interactive_outlier_handling(self, df, analysis_results, target_column='Life expectancy'):
        """
        Provide interactive options for outlier handling.
        
        Args:
            df (pd.DataFrame): Input dataframe
            analysis_results (dict): Results from comprehensive_outlier_analysis
            target_column (str): Name of the target column
            
        Returns:
            dict: Outlier handling options and recommendations
        """
        handling_options = {
            'detection_methods': list(analysis_results['target_outliers'].keys()),
            'outlier_counts': {},
            'removal_options': {},
            'recommendations': []
        }
        
        # Calculate outlier counts for each method
        for method, results in analysis_results['target_outliers'].items():
            handling_options['outlier_counts'][method] = {
                'count': results['outlier_count'],
                'percentage': results['outlier_percentage']
            }
        
        # Generate removal options
        for method, results in analysis_results['target_outliers'].items():
            if results['outlier_count'] > 0:
                handling_options['removal_options'][method] = {
                    'description': f"Remove {results['outlier_count']} outliers ({results['outlier_percentage']:.1f}%) detected by {method}",
                    'outlier_indices': results['outlier_indices'],
                    'impact': self._assess_removal_impact(df, results['outlier_indices'], target_column)
                }
        
        # Generate recommendations
        recommendations = []
        
        # Check for extreme outliers
        for method, results in analysis_results['target_outliers'].items():
            if results['outlier_percentage'] > 20:
                recommendations.append(f"⚠️ High outlier percentage ({results['outlier_percentage']:.1f}%) - consider data quality issues")
            elif results['outlier_percentage'] > 5:
                recommendations.append(f"ℹ️ Moderate outlier percentage ({results['outlier_percentage']:.1f}%) - consider removal for robust modeling")
            else:
                recommendations.append(f"✅ Low outlier percentage ({results['outlier_percentage']:.1f}%) - outliers may be legitimate")
        
        # Recommend best method
        best_method = min(analysis_results['target_outliers'].keys(), 
                         key=lambda x: analysis_results['target_outliers'][x]['outlier_count'])
        recommendations.append(f"🎯 Recommended method: {best_method} (most conservative)")
        
        handling_options['recommendations'] = recommendations
        
        return handling_options
    
    def _assess_removal_impact(self, df, outlier_indices, target_column):
        """Assess the impact of removing outliers on data statistics."""
        target_data = df[target_column].dropna()
        
        # Statistics before removal
        stats_before = {
            'count': len(target_data),
            'mean': np.mean(target_data),
            'std': np.std(target_data),
            'median': np.median(target_data),
            'min': np.min(target_data),
            'max': np.max(target_data)
        }
        
        # Statistics after removal
        clean_data = np.delete(target_data, outlier_indices)
        stats_after = {
            'count': len(clean_data),
            'mean': np.mean(clean_data),
            'std': np.std(clean_data),
            'median': np.median(clean_data),
            'min': np.min(clean_data),
            'max': np.max(clean_data)
        }
        
        # Calculate changes
        impact = {
            'records_removed': stats_before['count'] - stats_after['count'],
            'mean_change': stats_after['mean'] - stats_before['mean'],
            'std_change': stats_after['std'] - stats_before['std'],
            'range_change': (stats_after['max'] - stats_after['min']) - (stats_before['max'] - stats_before['min'])
        }
        
        return impact
    
    def remove_outliers(self, df, outlier_indices, method_name='custom'):
        """
        Remove outliers from the dataset.
        
        Args:
            df (pd.DataFrame): Input dataframe
            outlier_indices (array): Indices of outliers to remove
            method_name (str): Name of the method used for detection
            
        Returns:
            tuple: (cleaned_dataframe, removed_dataframe, removal_summary)
        """
        print(f"\n=== Removing Outliers (Method: {method_name}) ===")
        
        # Create copies
        df_clean = df.copy()
        df_removed = df.iloc[outlier_indices].copy()
        
        # Remove outliers
        df_clean = df_clean.drop(df_clean.index[outlier_indices])
        
        # Generate summary
        removal_summary = {
            'method': method_name,
            'original_count': len(df),
            'removed_count': len(outlier_indices),
            'remaining_count': len(df_clean),
            'removal_percentage': (len(outlier_indices) / len(df)) * 100
        }
        
        print(f"Original dataset: {removal_summary['original_count']} records")
        print(f"Removed outliers: {removal_summary['removed_count']} records")
        print(f"Remaining data: {removal_summary['remaining_count']} records")
        print(f"Removal percentage: {removal_summary['removal_percentage']:.2f}%")
        
        return df_clean, df_removed, removal_summary 