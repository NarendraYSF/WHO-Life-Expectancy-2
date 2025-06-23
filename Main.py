#!/usr/bin/env python3
"""
Life Expectancy Prediction Application
=====================================

A comprehensive Python application for predicting life expectancy based on various
health, economic, and social indicators using multiple linear regression models.

Features:
- Data preprocessing and cleaning
- Feature engineering
- Comprehensive outlier detection and handling
- Multiple regression models (Linear, Ridge, Lasso, Elastic Net, Random Forest, Gradient Boosting)
- Cross-validation and hyperparameter tuning
- Advanced visualization and analysis
- Model evaluation and comparison

Date: 2024
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

# Import custom modules
from data_preprocessing import DataPreprocessor
from model_training import LifeExpectancyModel
from visualization import LifeExpectancyVisualizer
from outlier_detection import OutlierDetectionSystem

def main():
    """
    Main function to run the complete life expectancy prediction pipeline.
    """
    print("=" * 60)
    print("LIFE EXPECTANCY PREDICTION APPLICATION")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Initialize components
    preprocessor = DataPreprocessor()
    model_trainer = LifeExpectancyModel()
    visualizer = LifeExpectancyVisualizer()
    outlier_system = OutlierDetectionSystem()
    
    # File path
    data_file = "Dataset/WHO Life Expectancy Descriptive Statistics - Raw Data.csv"
    
    try:
        # Step 1: Load and clean data
        print("STEP 1: DATA LOADING AND CLEANING")
        print("-" * 40)
        
        if not os.path.exists(data_file):
            print(f"Error: Data file not found at {data_file}")
            return
        
        df = preprocessor.load_and_clean_data(data_file)
        
        # Display basic information
        print(f"\nDataset Info:")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"Data types:\n{df.dtypes.value_counts()}")
        
        # Step 2: Analyze missing values
        print("\nSTEP 2: MISSING VALUES ANALYSIS")
        print("-" * 40)
        
        missing_stats = preprocessor.analyze_missing_values(df)
        
        # Step 3: Complete preprocessing pipeline
        print("\nSTEP 3: COMPLETE PREPROCESSING PIPELINE")
        print("-" * 40)
        
        X_scaled, y, feature_names, df_processed = preprocessor.prepare_final_dataset(df)
        
        print(f"Final dataset shape: {X_scaled.shape}")
        print(f"Number of features: {len(feature_names)}")
        print(f"Target variable range: {y.min():.2f} - {y.max():.2f}")
        
        # Step 4: Initial data visualization
        print("\nSTEP 4: INITIAL DATA VISUALIZATION")
        print("-" * 40)
        
        print("Generating data distribution plots...")
        visualizer.plot_data_distribution(df_processed)
        
        print("Generating correlation matrix...")
        visualizer.plot_correlation_matrix(df_processed)
        
        print("Generating target vs features plots...")
        visualizer.plot_target_vs_features(df_processed)
        
        # Step 5: Comprehensive Outlier Detection and Analysis
        print("\nSTEP 5: COMPREHENSIVE OUTLIER DETECTION AND ANALYSIS")
        print("-" * 40)
        
        # Perform comprehensive outlier analysis
        print("Performing comprehensive outlier analysis...")
        outlier_analysis = outlier_system.comprehensive_outlier_analysis(
            df_processed, 
            target_column='Life expectancy'
        )
        
        # Display outlier analysis summary
        print("\nOutlier Analysis Summary:")
        print("-" * 30)
        summary = outlier_analysis['summary']
        
        print(f"Methods used: {', '.join(summary['methods_used'])}")
        print(f"Features analyzed: {summary['total_features_analyzed']}")
        
        print("\nTarget Variable Outliers:")
        for method, stats in summary['target_outlier_summary'].items():
            print(f"  {method}: {stats['count']} outliers ({stats['percentage']:.2f}%)")
        
        print("\nRecommendations:")
        for rec in summary['recommendations']:
            print(f"  {rec}")
        
        # Generate outlier visualizations
        print("\nGenerating outlier analysis visualizations...")
        outlier_fig = outlier_system.visualize_outlier_analysis(
            df_processed, 
            outlier_analysis, 
            target_column='Life expectancy'
        )
        
        # Interactive outlier handling
        print("\nSTEP 6: INTERACTIVE OUTLIER HANDLING")
        print("-" * 40)
        
        handling_options = outlier_system.interactive_outlier_handling(
            df_processed, 
            outlier_analysis, 
            target_column='Life expectancy'
        )
        
        print("\nOutlier Handling Options:")
        print("-" * 30)
        
        for method, stats in handling_options['outlier_counts'].items():
            print(f"{method}: {stats['count']} outliers ({stats['percentage']:.2f}%)")
        
        print("\nRemoval Options:")
        for method, option in handling_options['removal_options'].items():
            print(f"  {option['description']}")
            impact = option['impact']
            print(f"    Impact: {impact['records_removed']} records removed")
            print(f"    Mean change: {impact['mean_change']:.3f}")
            print(f"    Std change: {impact['std_change']:.3f}")
        
        print("\nRecommendations:")
        for rec in handling_options['recommendations']:
            print(f"  {rec}")
        
        # Ask user for outlier handling preference
        print("\nOutlier Handling Options:")
        print("1. Remove outliers using recommended method")
        print("2. Remove outliers using specific method")
        print("3. Keep all outliers")
        print("4. Interactive selection")
        
        outlier_choice = input("\nEnter your choice (1-4): ").strip()
        
        if outlier_choice == "1":
            # Use recommended method (most conservative)
            best_method = min(outlier_analysis['target_outliers'].keys(), 
                             key=lambda x: outlier_analysis['target_outliers'][x]['outlier_count'])
            outlier_indices = outlier_analysis['target_outliers'][best_method]['outlier_indices']
            
            df_clean, df_removed, removal_summary = outlier_system.remove_outliers(
                df_processed, outlier_indices, best_method
            )
            
            # Re-preprocess cleaned data
            X_scaled, y, feature_names, df_processed = preprocessor.prepare_final_dataset(df_clean)
            print("Data reprocessed after outlier removal.")
            
        elif outlier_choice == "2":
            # Let user choose method
            print("\nAvailable methods:")
            for i, method in enumerate(handling_options['detection_methods'], 1):
                stats = handling_options['outlier_counts'][method]
                print(f"{i}. {method}: {stats['count']} outliers ({stats['percentage']:.2f}%)")
            
            method_choice = input("Enter method number: ").strip()
            try:
                method_idx = int(method_choice) - 1
                selected_method = handling_options['detection_methods'][method_idx]
                outlier_indices = outlier_analysis['target_outliers'][selected_method]['outlier_indices']
                
                df_clean, df_removed, removal_summary = outlier_system.remove_outliers(
                    df_processed, outlier_indices, selected_method
                )
                
                # Re-preprocess cleaned data
                X_scaled, y, feature_names, df_processed = preprocessor.prepare_final_dataset(df_clean)
                print("Data reprocessed after outlier removal.")
                
            except (ValueError, IndexError):
                print("Invalid choice. Keeping all outliers.")
                
        elif outlier_choice == "3":
            print("Keeping all outliers in the dataset.")
            
        elif outlier_choice == "4":
            print("Interactive outlier selection not implemented in command-line version.")
            print("Use the Streamlit app for interactive outlier handling.")
            print("Keeping all outliers for now.")
            
        else:
            print("Invalid choice. Keeping all outliers.")
        
        # Step 7: Model training and comparison
        print("\nSTEP 7: MODEL TRAINING AND COMPARISON")
        print("-" * 40)
        
        results = model_trainer.train_models(X_scaled, y, feature_names)
        
        # Step 8: Model visualization
        print("\nSTEP 8: MODEL VISUALIZATION")
        print("-" * 40)
        
        print("Generating model comparison plots...")
        visualizer.plot_model_comparison(results)
        
        print("Generating actual vs predicted plots...")
        visualizer.plot_actual_vs_predicted(results)
        
        # Step 9: Feature importance analysis
        print("\nSTEP 9: FEATURE IMPORTANCE ANALYSIS")
        print("-" * 40)
        
        importance_df = model_trainer.analyze_feature_importance()
        
        if importance_df is not None:
            print("Top 10 most important features:")
            print(importance_df.head(10))
            
            print("\nGenerating feature importance plots...")
            visualizer.plot_feature_importance(importance_df)
        
        # Step 10: Hyperparameter tuning (optional)
        print("\nSTEP 10: HYPERPARAMETER TUNING")
        print("-" * 40)
        
        tune_hyperparameters = input("Do you want to perform hyperparameter tuning? (y/n): ").lower().strip()
        
        if tune_hyperparameters == 'y':
            model_to_tune = input("Enter model name to tune (Ridge Regression/Lasso Regression/Elastic Net/Random Forest/Gradient Boosting): ").strip()
            
            if model_to_tune in results:
                print(f"Performing hyperparameter tuning for {model_to_tune}...")
                tuning_results = model_trainer.hyperparameter_tuning(X_scaled, y, model_to_tune)
                
                if tuning_results:
                    print(f"Best parameters: {tuning_results['best_params']}")
                    print(f"Best CV score: {tuning_results['best_score']:.4f}")
            else:
                print(f"Model {model_to_tune} not found in trained models.")
        
        # Step 11: Interactive dashboard
        print("\nSTEP 11: INTERACTIVE DASHBOARD")
        print("-" * 40)
        
        create_dashboard = input("Do you want to create an interactive dashboard? (y/n): ").lower().strip()
        
        if create_dashboard == 'y':
            print("Creating interactive dashboard...")
            visualizer.create_interactive_dashboard(df_processed, results, importance_df)
        
        # Step 12: Model saving
        print("\nSTEP 12: MODEL SAVING")
        print("-" * 40)
        
        save_models = input("Do you want to save the trained models? (y/n): ").lower().strip()
        
        if save_models == 'y':
            # Create models directory if it doesn't exist
            os.makedirs('models', exist_ok=True)
            
            # Save best model
            best_model_name = model_trainer.best_model_name
            model_trainer.save_model(best_model_name, f'models/{best_model_name.replace(" ", "_").lower()}.pkl')
            
            # Save all models
            for model_name in results.keys():
                model_trainer.save_model(model_name, f'models/{model_name.replace(" ", "_").lower()}.pkl')
            
            print("Models saved successfully!")
        
        # Step 13: Final summary
        print("\nSTEP 13: FINAL SUMMARY")
        print("-" * 40)
        
        print("Model Performance Summary:")
        print("-" * 30)
        
        summary_data = []
        for name, result in results.items():
            summary_data.append({
                'Model': name,
                'Test R²': f"{result['test_r2']:.4f}",
                'Test RMSE': f"{result['test_rmse']:.4f}",
                'CV R²': f"{result['cv_mean']:.4f} (±{result['cv_std']:.4f})"
            })
        
        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string(index=False))
        
        print(f"\nBest Model: {model_trainer.best_model_name}")
        print(f"Best Test R²: {results[model_trainer.best_model_name]['test_r2']:.4f}")
        print(f"Best Test RMSE: {results[model_trainer.best_model_name]['test_rmse']:.4f}")
        
        # Feature importance summary
        if importance_df is not None:
            print(f"\nTop 5 Most Important Features:")
            print("-" * 30)
            for i, row in importance_df.head(5).iterrows():
                print(f"{i+1}. {row['Feature']}: {row['Importance']:.4f}")
        
        # Outlier analysis summary
        if 'removal_summary' in locals():
            print(f"\nOutlier Removal Summary:")
            print("-" * 30)
            print(f"Method used: {removal_summary['method']}")
            print(f"Records removed: {removal_summary['removed_count']}")
            print(f"Removal percentage: {removal_summary['removal_percentage']:.2f}%")
        
        print("\n" + "=" * 60)
        print("LIFE EXPECTANCY PREDICTION COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        print("Please check the error and try again.")
        return

def run_streamlit_app():
    """
    Run the Streamlit web application.
    """
    try:
        import streamlit as st
        from streamlit_app import main as streamlit_main
        streamlit_main()
    except ImportError:
        print("Streamlit not installed. Install with: pip install streamlit")
        print("Then run: streamlit run streamlit_app.py")

if __name__ == "__main__":
    # Check command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "--streamlit":
        run_streamlit_app()
    else:
        main()
