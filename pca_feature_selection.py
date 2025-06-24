import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

class PCAFeatureSelector:
    """
    A PCA-based feature selection class that can be integrated with the existing model training pipeline.
    """
    
    def __init__(self):
        self.pca = None
        self.scaler = StandardScaler()
        self.selected_features = None
        self.feature_importance = None
        self.reduction_summary = {}
        
    def select_features_by_variance(self, X, y, variance_threshold=0.95, max_components=None):
        """
        Select features using PCA based on explained variance threshold.
        
        Args:
            X (array): Feature matrix
            y (array): Target values
            variance_threshold (float): Minimum explained variance to retain
            max_components (int): Maximum number of components to consider
            
        Returns:
            tuple: (X_selected, selected_features, reduction_summary)
        """
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Determine number of components
        if max_components is None:
            max_components = min(X.shape[1], X.shape[0] - 1)
        
        # Fit PCA
        self.pca = PCA(n_components=max_components)
        self.pca.fit(X_scaled)
        
        # Find number of components needed for variance threshold
        cumulative_variance = np.cumsum(self.pca.explained_variance_ratio_)
        n_components = np.argmax(cumulative_variance >= variance_threshold) + 1
        n_components = max(2, min(n_components, max_components))  # At least 2 components
        
        # Refit PCA with optimal number of components
        self.pca = PCA(n_components=n_components)
        X_pca = self.pca.fit_transform(X_scaled)
        
        # Calculate feature importance based on loadings
        self.feature_importance = np.sum(self.pca.components_ ** 2, axis=0)
        
        # Create feature names for PCA components
        self.selected_features = [f'PC{i+1}' for i in range(n_components)]
        
        # Store reduction summary
        self.reduction_summary = {
            'original_features': X.shape[1],
            'selected_features': n_components,
            'reduction_ratio': 1 - (n_components / X.shape[1]),
            'explained_variance': cumulative_variance[n_components - 1],
            'variance_threshold': variance_threshold,
            'feature_importance': self.feature_importance
        }
        
        return X_pca, self.selected_features, self.reduction_summary
    
    def select_features_by_importance(self, X, y, top_n=None, importance_threshold=None):
        """
        Select original features based on their importance in PCA.
        
        Args:
            X (array): Feature matrix
            y (array): Target values
            top_n (int): Number of top features to select
            importance_threshold (float): Minimum importance threshold
            
        Returns:
            tuple: (X_selected, selected_features, reduction_summary)
        """
        # First, perform PCA to get feature importance
        X_scaled = self.scaler.fit_transform(X)
        
        # Use enough components to explain 95% variance
        temp_pca = PCA()
        temp_pca.fit(X_scaled)
        cumulative_variance = np.cumsum(temp_pca.explained_variance_ratio_)
        n_components = np.argmax(cumulative_variance >= 0.95) + 1
        
        # Fit PCA with optimal components
        self.pca = PCA(n_components=n_components)
        self.pca.fit(X_scaled)
        
        # Calculate feature importance
        self.feature_importance = np.sum(self.pca.components_ ** 2, axis=0)
        
        # Select features based on criteria
        if top_n is not None:
            # Select top N features
            top_indices = np.argsort(self.feature_importance)[-top_n:]
        elif importance_threshold is not None:
            # Select features above threshold
            top_indices = np.where(self.feature_importance >= importance_threshold)[0]
        else:
            # Default: select features that contribute to 80% of total importance
            sorted_importance = np.sort(self.feature_importance)[::-1]
            cumulative_importance = np.cumsum(sorted_importance)
            cumulative_importance = cumulative_importance / cumulative_importance[-1]
            n_features = np.argmax(cumulative_importance >= 0.8) + 1
            top_indices = np.argsort(self.feature_importance)[-n_features:]
        
        # Sort indices by importance
        top_indices = sorted(top_indices, key=lambda x: self.feature_importance[x], reverse=True)
        
        # Select features
        X_selected = X[:, top_indices]
        self.selected_features = top_indices
        
        # Store reduction summary
        self.reduction_summary = {
            'original_features': X.shape[1],
            'selected_features': len(top_indices),
            'reduction_ratio': 1 - (len(top_indices) / X.shape[1]),
            'importance_threshold': importance_threshold,
            'top_n': top_n,
            'feature_importance': self.feature_importance[top_indices]
        }
        
        return X_selected, top_indices, self.reduction_summary
    
    def evaluate_feature_selection(self, X, y, feature_names=None, test_size=0.2, random_state=42):
        """
        Evaluate the impact of feature selection on model performance.
        
        Args:
            X (array): Feature matrix
            y (array): Target values
            feature_names (list): List of feature names
            test_size (float): Proportion of data for testing
            random_state (int): Random seed
            
        Returns:
            dict: Evaluation results
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Train model on original features
        lr_original = LinearRegression()
        lr_original.fit(X_train, y_train)
        y_pred_original = lr_original.predict(X_test)
        
        r2_original = r2_score(y_test, y_pred_original)
        rmse_original = np.sqrt(mean_squared_error(y_test, y_pred_original))
        
        # Perform PCA feature selection
        X_pca, pca_features, pca_summary = self.select_features_by_variance(
            X_train, y_train, variance_threshold=0.95
        )
        
        # Transform test data
        X_test_scaled = self.scaler.transform(X_test)
        X_test_pca = self.pca.transform(X_test_scaled)
        
        # Train model on PCA features
        lr_pca = LinearRegression()
        lr_pca.fit(X_pca, y_train)
        y_pred_pca = lr_pca.predict(X_test_pca)
        
        r2_pca = r2_score(y_test, y_pred_pca)
        rmse_pca = np.sqrt(mean_squared_error(y_test, y_pred_pca))
        
        # Calculate performance comparison
        r2_change = r2_pca - r2_original
        rmse_change = rmse_pca - rmse_original
        
        evaluation_results = {
            'original_features': {
                'n_features': X.shape[1],
                'r2_score': r2_original,
                'rmse': rmse_original
            },
            'pca_features': {
                'n_features': len(pca_features),
                'r2_score': r2_pca,
                'rmse': rmse_pca,
                'explained_variance': pca_summary['explained_variance']
            },
            'comparison': {
                'r2_change': r2_change,
                'rmse_change': rmse_change,
                'feature_reduction': pca_summary['reduction_ratio'],
                'performance_maintained': abs(r2_change) < 0.05  # Within 5% threshold
            },
            'pca_summary': pca_summary
        }
        
        return evaluation_results
    
    def get_feature_importance_dataframe(self, feature_names=None):
        """
        Get feature importance as a DataFrame.
        
        Args:
            feature_names (list): List of feature names
            
        Returns:
            pd.DataFrame: Feature importance DataFrame
        """
        if self.feature_importance is None:
            return None
        
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(len(self.feature_importance))]
        
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': self.feature_importance
        })
        
        importance_df = importance_df.sort_values('Importance', ascending=False)
        return importance_df
    
    def create_feature_selection_summary(self, evaluation_results):
        """
        Create a summary of feature selection results.
        
        Args:
            evaluation_results (dict): Results from evaluate_feature_selection
            
        Returns:
            dict: Summary statistics
        """
        original = evaluation_results['original_features']
        pca = evaluation_results['pca_features']
        comparison = evaluation_results['comparison']
        
        summary = {
            'feature_reduction': {
                'original_count': original['n_features'],
                'reduced_count': pca['n_features'],
                'reduction_percentage': comparison['feature_reduction'] * 100,
                'explained_variance': pca['explained_variance']
            },
            'performance_impact': {
                'r2_original': original['r2_score'],
                'r2_pca': pca['r2_score'],
                'r2_change': comparison['r2_change'],
                'rmse_original': original['rmse'],
                'rmse_pca': pca['rmse'],
                'rmse_change': comparison['rmse_change'],
                'performance_maintained': comparison['performance_maintained']
            },
            'recommendation': self._generate_recommendation(evaluation_results)
        }
        
        return summary
    
    def _generate_recommendation(self, evaluation_results):
        """
        Generate recommendation based on evaluation results.
        
        Args:
            evaluation_results (dict): Evaluation results
            
        Returns:
            str: Recommendation
        """
        comparison = evaluation_results['comparison']
        pca_summary = evaluation_results['pca_summary']
        
        if comparison['performance_maintained']:
            if comparison['feature_reduction'] > 0.5:
                return "✅ Strong recommendation: Use PCA features. Significant feature reduction (>50%) with maintained performance."
            elif comparison['feature_reduction'] > 0.2:
                return "✅ Good recommendation: Use PCA features. Moderate feature reduction (20-50%) with maintained performance."
            else:
                return "⚠️ Consider PCA: Small feature reduction with maintained performance. May not be worth the complexity."
        else:
            if comparison['r2_change'] < -0.1:
                return "❌ Not recommended: PCA significantly reduces performance. Stick with original features."
            else:
                return "⚠️ Consider trade-off: PCA reduces performance slightly but provides feature reduction. Evaluate based on your priorities."
    
    def transform_new_data(self, X_new):
        """
        Transform new data using fitted PCA.
        
        Args:
            X_new (array): New data to transform
            
        Returns:
            array: Transformed data
        """
        if self.pca is None:
            raise ValueError("PCA has not been fitted yet. Call select_features_by_variance or select_features_by_importance first.")
        
        # Scale the data
        X_new_scaled = self.scaler.transform(X_new)
        
        # Transform using PCA
        X_new_pca = self.pca.transform(X_new_scaled)
        
        return X_new_pca
    
    def get_pca_components_info(self):
        """
        Get information about PCA components.
        
        Returns:
            dict: PCA components information
        """
        if self.pca is None:
            return None
        
        components_info = {
            'n_components': self.pca.n_components_,
            'explained_variance_ratio': self.pca.explained_variance_ratio_.tolist(),
            'cumulative_variance': np.cumsum(self.pca.explained_variance_ratio_).tolist(),
            'singular_values': self.pca.singular_values_.tolist(),
            'components': self.pca.components_.tolist()
        }
        
        return components_info 