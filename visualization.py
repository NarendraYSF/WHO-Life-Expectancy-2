import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Set style for matplotlib
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class LifeExpectancyVisualizer:
    """
    A comprehensive visualization class for life expectancy prediction analysis.
    Includes various plots for data exploration, model evaluation, and results presentation.
    """
    
    def __init__(self):
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
    def plot_data_distribution(self, df, target_column='Life expectancy', figsize=(15, 10)):
        """
        Plot distribution of key variables in the dataset.
        
        Args:
            df (pd.DataFrame): Input dataframe
            target_column (str): Name of the target column
            figsize (tuple): Figure size
        """
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle('Distribution of Key Variables', fontsize=16, fontweight='bold')
        
        # Target variable distribution
        axes[0, 0].hist(df[target_column].dropna(), bins=30, alpha=0.7, color=self.colors[0])
        axes[0, 0].set_title(f'{target_column} Distribution')
        axes[0, 0].set_xlabel(target_column)
        axes[0, 0].set_ylabel('Frequency')
        
        # Adult Mortality
        if 'Adult Mortality' in df.columns:
            axes[0, 1].hist(df['Adult Mortality'].dropna(), bins=30, alpha=0.7, color=self.colors[1])
            axes[0, 1].set_title('Adult Mortality Distribution')
            axes[0, 1].set_xlabel('Adult Mortality')
            axes[0, 1].set_ylabel('Frequency')
        
        # GDP
        if 'GDP' in df.columns:
            axes[0, 2].hist(df['GDP'].dropna(), bins=30, alpha=0.7, color=self.colors[2])
            axes[0, 2].set_title('GDP Distribution')
            axes[0, 2].set_xlabel('GDP')
            axes[0, 2].set_ylabel('Frequency')
        
        # BMI
        if 'BMI' in df.columns:
            axes[1, 0].hist(df['BMI'].dropna(), bins=30, alpha=0.7, color=self.colors[3])
            axes[1, 0].set_title('BMI Distribution')
            axes[1, 0].set_xlabel('BMI')
            axes[1, 0].set_ylabel('Frequency')
        
        # Schooling
        if 'Schooling' in df.columns:
            axes[1, 1].hist(df['Schooling'].dropna(), bins=30, alpha=0.7, color=self.colors[4])
            axes[1, 1].set_title('Schooling Distribution')
            axes[1, 1].set_xlabel('Schooling')
            axes[1, 1].set_ylabel('Frequency')
        
        # Alcohol
        if 'Alcohol' in df.columns:
            axes[1, 2].hist(df['Alcohol'].dropna(), bins=30, alpha=0.7, color=self.colors[5])
            axes[1, 2].set_title('Alcohol Consumption Distribution')
            axes[1, 2].set_xlabel('Alcohol')
            axes[1, 2].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.show()
    
    def plot_correlation_matrix(self, df, target_column='Life expectancy', figsize=(12, 10)):
        """
        Plot correlation matrix for numerical variables.
        
        Args:
            df (pd.DataFrame): Input dataframe
            target_column (str): Name of the target column
            figsize (tuple): Figure size
        """
        # Select numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if target_column not in numerical_cols:
            numerical_cols.append(target_column)
        
        # Calculate correlation matrix
        corr_matrix = df[numerical_cols].corr()
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        plt.figure(figsize=figsize)
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": .8})
        plt.title('Correlation Matrix of Numerical Variables', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def plot_target_vs_features(self, df, target_column='Life expectancy', n_features=6, figsize=(15, 10)):
        """
        Plot target variable against key features.
        
        Args:
            df (pd.DataFrame): Input dataframe
            target_column (str): Name of the target column
            n_features (int): Number of features to plot
            figsize (tuple): Figure size
        """
        # Select numerical features (excluding target)
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in numerical_cols if col != target_column]
        
        # Select top features based on correlation with target
        correlations = []
        for col in feature_cols:
            corr = df[col].corr(df[target_column])
            correlations.append((col, abs(corr)))
        
        correlations.sort(key=lambda x: x[1], reverse=True)
        top_features = [col for col, _ in correlations[:n_features]]
        
        # Create subplots
        n_rows = (n_features + 1) // 2
        fig, axes = plt.subplots(n_rows, 2, figsize=figsize)
        fig.suptitle(f'{target_column} vs Key Features', fontsize=16, fontweight='bold')
        
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, feature in enumerate(top_features):
            row = i // 2
            col = i % 2
            
            axes[row, col].scatter(df[feature], df[target_column], alpha=0.6, color=self.colors[i % len(self.colors)])
            axes[row, col].set_xlabel(feature)
            axes[row, col].set_ylabel(target_column)
            axes[row, col].set_title(f'{target_column} vs {feature}')
            
            # Add trend line
            z = np.polyfit(df[feature].dropna(), df[target_column].dropna(), 1)
            p = np.poly1d(z)
            axes[row, col].plot(df[feature], p(df[feature]), "r--", alpha=0.8)
        
        # Hide empty subplots
        for i in range(len(top_features), n_rows * 2):
            row = i // 2
            col = i % 2
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    def plot_model_comparison(self, results, figsize=(15, 10)):
        """
        Plot comparison of different models' performance.
        
        Args:
            results (dict): Model results dictionary
            figsize (tuple): Figure size
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        # Extract metrics
        model_names = list(results.keys())
        train_r2 = [results[name]['train_r2'] for name in model_names]
        test_r2 = [results[name]['test_r2'] for name in model_names]
        test_rmse = [results[name]['test_rmse'] for name in model_names]
        cv_mean = [results[name]['cv_mean'] for name in model_names]
        
        # R² Comparison
        x = np.arange(len(model_names))
        width = 0.35
        
        axes[0, 0].bar(x - width/2, train_r2, width, label='Train R²', alpha=0.8)
        axes[0, 0].bar(x + width/2, test_r2, width, label='Test R²', alpha=0.8)
        axes[0, 0].set_xlabel('Models')
        axes[0, 0].set_ylabel('R² Score')
        axes[0, 0].set_title('R² Score Comparison')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(model_names, rotation=45, ha='right')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # RMSE Comparison
        axes[0, 1].bar(model_names, test_rmse, alpha=0.8, color=self.colors[1])
        axes[0, 1].set_xlabel('Models')
        axes[0, 1].set_ylabel('RMSE')
        axes[0, 1].set_title('RMSE Comparison')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Cross-validation scores
        axes[1, 0].bar(model_names, cv_mean, alpha=0.8, color=self.colors[2])
        axes[1, 0].set_xlabel('Models')
        axes[1, 0].set_ylabel('CV R² Score')
        axes[1, 0].set_title('Cross-Validation R² Scores')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Performance summary
        summary_data = pd.DataFrame({
            'Model': model_names,
            'Test R²': test_r2,
            'Test RMSE': test_rmse,
            'CV R²': cv_mean
        })
        
        axes[1, 1].axis('tight')
        axes[1, 1].axis('off')
        table = axes[1, 1].table(cellText=summary_data.values, colLabels=summary_data.columns,
                                cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        axes[1, 1].set_title('Performance Summary')
        
        plt.tight_layout()
        plt.show()
    
    def plot_actual_vs_predicted(self, results, model_name=None, figsize=(15, 5)):
        """
        Plot actual vs predicted values for model evaluation.
        
        Args:
            results (dict): Model results dictionary
            model_name (str): Name of the model to plot
            figsize (tuple): Figure size
        """
        if model_name is None:
            # Use the best model
            model_name = max(results.keys(), key=lambda k: results[k]['test_r2'])
        
        if model_name not in results:
            print(f"Model {model_name} not found in results!")
            return
        
        y_test = results[model_name]['y_test']
        y_pred = results[model_name]['y_pred_test']
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        fig.suptitle(f'Actual vs Predicted - {model_name}', fontsize=16, fontweight='bold')
        
        # Scatter plot
        axes[0].scatter(y_test, y_pred, alpha=0.6, color=self.colors[0])
        axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[0].set_xlabel('Actual Values')
        axes[0].set_ylabel('Predicted Values')
        axes[0].set_title('Actual vs Predicted')
        axes[0].grid(True, alpha=0.3)
        
        # Residuals plot
        residuals = y_test - y_pred
        axes[1].scatter(y_pred, residuals, alpha=0.6, color=self.colors[1])
        axes[1].axhline(y=0, color='r', linestyle='--')
        axes[1].set_xlabel('Predicted Values')
        axes[1].set_ylabel('Residuals')
        axes[1].set_title('Residuals Plot')
        axes[1].grid(True, alpha=0.3)
        
        # Residuals distribution
        axes[2].hist(residuals, bins=30, alpha=0.7, color=self.colors[2])
        axes[2].set_xlabel('Residuals')
        axes[2].set_ylabel('Frequency')
        axes[2].set_title('Residuals Distribution')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_feature_importance(self, importance_df, top_n=15, figsize=(12, 8)):
        """
        Plot feature importance for the model.
        
        Args:
            importance_df (pd.DataFrame): Feature importance dataframe
            top_n (int): Number of top features to display
            figsize (tuple): Figure size
        """
        if importance_df is None or len(importance_df) == 0:
            print("No feature importance data available!")
            return
        
        # Select top features
        top_features = importance_df.head(top_n)
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle('Feature Importance Analysis', fontsize=16, fontweight='bold')
        
        # Horizontal bar plot
        axes[0].barh(range(len(top_features)), top_features['Importance'], color=self.colors[0])
        axes[0].set_yticks(range(len(top_features)))
        axes[0].set_yticklabels(top_features['Feature'])
        axes[0].set_xlabel('Importance')
        axes[0].set_title(f'Top {top_n} Feature Importance')
        axes[0].grid(True, alpha=0.3)
        
        # Pie chart for top 10 features
        top_10 = top_features.head(10)
        axes[1].pie(top_10['Importance'], labels=top_10['Feature'], autopct='%1.1f%%',
                   startangle=90, colors=plt.cm.Set3(np.linspace(0, 1, len(top_10))))
        axes[1].set_title('Top 10 Features Distribution')
        
        plt.tight_layout()
        plt.show()
    
    def plot_outlier_analysis(self, y, outlier_indices, outlier_scores, figsize=(15, 5)):
        """
        Plot outlier analysis results.
        
        Args:
            y (array): Target values
            outlier_indices (array): Indices of outliers
            outlier_scores (array): Outlier scores
            figsize (tuple): Figure size
        """
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        fig.suptitle('Outlier Analysis', fontsize=16, fontweight='bold')
        
        # Original distribution
        axes[0].hist(y, bins=30, alpha=0.7, color=self.colors[0], label='All Data')
        axes[0].hist(y[outlier_indices], bins=30, alpha=0.7, color=self.colors[1], label='Outliers')
        axes[0].set_xlabel('Life Expectancy')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Distribution with Outliers Highlighted')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Outlier scores
        axes[1].scatter(range(len(y)), outlier_scores, alpha=0.6, color=self.colors[2])
        axes[1].scatter(outlier_indices, outlier_scores[outlier_indices], 
                       color=self.colors[1], s=50, label='Outliers')
        axes[1].set_xlabel('Sample Index')
        axes[1].set_ylabel('Outlier Score')
        axes[1].set_title('Outlier Scores')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Box plot
        axes[2].boxplot(y, patch_artist=True, boxprops=dict(facecolor=self.colors[3], alpha=0.7))
        axes[2].set_ylabel('Life Expectancy')
        axes[2].set_title('Box Plot')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def create_interactive_dashboard(self, df, results, importance_df):
        """
        Create an interactive dashboard using Plotly.
        
        Args:
            df (pd.DataFrame): Preprocessed dataframe
            results (dict): Model results
            importance_df (pd.DataFrame): Feature importance dataframe
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Model Performance Comparison', 'Feature Importance',
                          'Actual vs Predicted', 'Target Distribution'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "histogram"}]]
        )
        
        # Model performance comparison
        model_names = list(results.keys())
        test_r2 = [results[name]['test_r2'] for name in model_names]
        
        fig.add_trace(
            go.Bar(x=model_names, y=test_r2, name='Test R²', marker_color=self.colors[0]),
            row=1, col=1
        )
        
        # Feature importance (top 10)
        if importance_df is not None and len(importance_df) > 0:
            top_10 = importance_df.head(10)
            fig.add_trace(
                go.Bar(x=top_10['Importance'], y=top_10['Feature'], 
                      orientation='h', name='Feature Importance', marker_color=self.colors[1]),
                row=1, col=2
            )
        
        # Actual vs Predicted (best model)
        best_model = max(results.keys(), key=lambda k: results[k]['test_r2'])
        y_test = results[best_model]['y_test']
        y_pred = results[best_model]['y_pred_test']
        
        fig.add_trace(
            go.Scatter(x=y_test, y=y_pred, mode='markers', name='Predictions',
                      marker=dict(color=self.colors[2], opacity=0.6)),
            row=2, col=1
        )
        
        # Add perfect prediction line
        min_val, max_val = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
        fig.add_trace(
            go.Scatter(x=[min_val, max_val], y=[min_val, max_val], 
                      mode='lines', name='Perfect Prediction',
                      line=dict(color='red', dash='dash')),
            row=2, col=1
        )
        
        # Target distribution
        fig.add_trace(
            go.Histogram(x=df['Life expectancy'], name='Life Expectancy',
                        marker_color=self.colors[3]),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title_text="Life Expectancy Prediction Dashboard",
            showlegend=True,
            height=800
        )
        
        fig.show()
    
    def save_plots(self, filename_prefix='life_expectancy_analysis'):
        """
        Save all current plots to files.
        
        Args:
            filename_prefix (str): Prefix for saved files
        """
        # This would save all current plots
        # Implementation depends on specific requirements
        print(f"Plots would be saved with prefix: {filename_prefix}") 