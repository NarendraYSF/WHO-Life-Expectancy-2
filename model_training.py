import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures
import joblib
import warnings
warnings.filterwarnings('ignore')

class LifeExpectancyModel:
    """
    A comprehensive model training class for life expectancy prediction.
    Includes multiple regression models, cross-validation, and regularization.
    """
    
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.feature_names = None
        self.scaler = None
        
    def train_models(self, X, y, feature_names, test_size=0.2, random_state=42):
        """
        Train multiple regression models and compare their performance.
        
        Args:
            X (array): Scaled feature matrix
            y (array): Target values
            feature_names (list): List of feature names
            test_size (float): Proportion of data for testing
            random_state (int): Random seed for reproducibility
            
        Returns:
            dict: Model performance metrics
        """
        print("=== Model Training ===")
        
        self.feature_names = feature_names
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Define models to train
        models_to_train = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=0.1),
            'Elastic Net': ElasticNet(alpha=0.1, l1_ratio=0.5),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=random_state),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=random_state)
        }
        
        # Train and evaluate models
        results = {}
        
        for name, model in models_to_train.items():
            print(f"\nTraining {name}...")
            
            # Train the model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Calculate metrics
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            train_mae = mean_absolute_error(y_train, y_pred_train)
            test_mae = mean_absolute_error(y_test, y_pred_test)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
            
            # Store results
            results[name] = {
                'model': model,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'train_mae': train_mae,
                'test_mae': test_mae,
                'cv_mean': cv_mean,
                'cv_std': cv_std,
                'y_pred_test': y_pred_test,
                'y_test': y_test
            }
            
            self.models[name] = model
            
            print(f"  Train R²: {train_r2:.4f}")
            print(f"  Test R²: {test_r2:.4f}")
            print(f"  Test RMSE: {test_rmse:.4f}")
            print(f"  CV R²: {cv_mean:.4f} (±{cv_std:.4f})")
        
        # Find the best model based on test R²
        self.best_model_name = max(results.keys(), key=lambda k: results[k]['test_r2'])
        self.best_model = results[self.best_model_name]['model']
        
        print(f"\nBest Model: {self.best_model_name}")
        print(f"Best Test R²: {results[self.best_model_name]['test_r2']:.4f}")
        
        return results
    
    def hyperparameter_tuning(self, X, y, model_name='Ridge Regression', cv=5):
        """
        Perform hyperparameter tuning for the specified model.
        
        Args:
            X (array): Feature matrix
            y (array): Target values
            model_name (str): Name of the model to tune
            cv (int): Number of cross-validation folds
            
        Returns:
            dict: Best parameters and scores
        """
        print(f"\n=== Hyperparameter Tuning for {model_name} ===")
        
        # Define parameter grids for different models
        param_grids = {
            'Ridge Regression': {
                'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
            },
            'Lasso Regression': {
                'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
            },
            'Elastic Net': {
                'alpha': [0.001, 0.01, 0.1, 1.0],
                'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
            },
            'Random Forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10]
            },
            'Gradient Boosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            }
        }
        
        if model_name not in param_grids:
            print(f"No parameter grid defined for {model_name}")
            return None
        
        # Get the base model
        base_models = {
            'Ridge Regression': Ridge(),
            'Lasso Regression': Lasso(),
            'Elastic Net': ElasticNet(),
            'Random Forest': RandomForestRegressor(random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(random_state=42)
        }
        
        # Perform grid search
        grid_search = GridSearchCV(
            base_models[model_name],
            param_grids[model_name],
            cv=cv,
            scoring='r2',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X, y)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")
        
        # Update the model with best parameters
        self.models[model_name] = grid_search.best_estimator_
        
        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'best_estimator': grid_search.best_estimator_
        }
    
    def analyze_feature_importance(self, model_name=None):
        """
        Analyze feature importance for the specified model.
        
        Args:
            model_name (str): Name of the model to analyze
            
        Returns:
            pd.DataFrame: Feature importance dataframe
        """
        if model_name is None:
            model_name = self.best_model_name
        
        if model_name not in self.models:
            print(f"Model {model_name} not found!")
            return None
        
        model = self.models[model_name]
        
        # Get feature importance based on model type
        if hasattr(model, 'coef_'):
            # Linear models
            importance = np.abs(model.coef_)
        elif hasattr(model, 'feature_importances_'):
            # Tree-based models
            importance = model.feature_importances_
        else:
            print(f"Cannot extract feature importance from {model_name}")
            return None
        
        # Create importance dataframe
        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': importance
        })
        
        importance_df = importance_df.sort_values('Importance', ascending=False)
        
        return importance_df
    
    def detect_outliers(self, X, y, method='zscore', threshold=3):
        """
        Detect outliers in the dataset using various methods.
        
        Args:
            X (array): Feature matrix
            y (array): Target values
            method (str): Outlier detection method ('zscore', 'iqr')
            threshold (float): Threshold for outlier detection
            
        Returns:
            tuple: (outlier_indices, outlier_scores)
        """
        print(f"\n=== Outlier Detection ({method}) ===")
        
        if method == 'zscore':
            # Z-score method
            z_scores = np.abs((y - np.mean(y)) / np.std(y))
            outlier_indices = np.where(z_scores > threshold)[0]
            outlier_scores = z_scores
        elif method == 'iqr':
            # IQR method
            Q1 = np.percentile(y, 25)
            Q3 = np.percentile(y, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outlier_indices = np.where((y < lower_bound) | (y > upper_bound))[0]
            outlier_scores = np.abs(y - np.median(y)) / IQR
        else:
            print(f"Unknown outlier detection method: {method}")
            return None, None
        
        print(f"Detected {len(outlier_indices)} outliers out of {len(y)} samples")
        
        return outlier_indices, outlier_scores
    
    def remove_outliers(self, X, y, outlier_indices):
        """
        Remove outliers from the dataset.
        
        Args:
            X (array): Feature matrix
            y (array): Target values
            outlier_indices (array): Indices of outliers to remove
            
        Returns:
            tuple: (X_clean, y_clean)
        """
        # Create mask for non-outlier indices
        mask = np.ones(len(y), dtype=bool)
        mask[outlier_indices] = False
        
        X_clean = X[mask]
        y_clean = y[mask]
        
        print(f"Removed {len(outlier_indices)} outliers")
        print(f"New dataset shape: {X_clean.shape}")
        
        return X_clean, y_clean
    
    def save_model(self, model_name, filepath):
        """
        Save a trained model to disk.
        
        Args:
            model_name (str): Name of the model to save
            filepath (str): Path to save the model
        """
        if model_name in self.models:
            joblib.dump(self.models[model_name], filepath)
            print(f"Model {model_name} saved to {filepath}")
        else:
            print(f"Model {model_name} not found!")
    
    def load_model(self, model_name, filepath):
        """
        Load a trained model from disk.
        
        Args:
            model_name (str): Name for the loaded model
            filepath (str): Path to the saved model
        """
        try:
            model = joblib.load(filepath)
            self.models[model_name] = model
            print(f"Model {model_name} loaded from {filepath}")
        except Exception as e:
            print(f"Error loading model: {e}")
    
    def predict(self, X, model_name=None):
        """
        Make predictions using the specified model.
        
        Args:
            X (array): Feature matrix
            model_name (str): Name of the model to use
            
        Returns:
            array: Predictions
        """
        if model_name is None:
            model_name = self.best_model_name
        
        if model_name not in self.models:
            print(f"Model {model_name} not found!")
            return None
        
        return self.models[model_name].predict(X)
    
    def get_model_summary(self):
        """
        Get a summary of all trained models.
        
        Returns:
            pd.DataFrame: Model summary
        """
        summary_data = []
        
        for name, model in self.models.items():
            model_type = type(model).__name__
            summary_data.append({
                'Model Name': name,
                'Model Type': model_type,
                'Is Best Model': (name == self.best_model_name)
            })
        
        return pd.DataFrame(summary_data) 