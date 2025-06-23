#!/usr/bin/env python3
"""
Test script for Life Expectancy Prediction Application
=====================================================

This script tests the main components of the application to ensure everything works correctly.
"""

import sys
import os
import pandas as pd
import numpy as np

# Import custom modules
from data_preprocessing import DataPreprocessor
from model_training import LifeExpectancyModel
from visualization import LifeExpectancyVisualizer

def test_imports():
    """Test if all required modules can be imported."""
    print("Testing imports...")
    
    try:
        # Test if modules are available
        preprocessor = DataPreprocessor()
        print("✅ DataPreprocessor imported successfully")
    except Exception as e:
        print(f"❌ Failed to import DataPreprocessor: {e}")
        return False
    
    try:
        model_trainer = LifeExpectancyModel()
        print("✅ LifeExpectancyModel imported successfully")
    except Exception as e:
        print(f"❌ Failed to import LifeExpectancyModel: {e}")
        return False
    
    try:
        visualizer = LifeExpectancyVisualizer()
        print("✅ LifeExpectancyVisualizer imported successfully")
    except Exception as e:
        print(f"❌ Failed to import LifeExpectancyVisualizer: {e}")
        return False
    
    return True

def test_data_loading():
    """Test data loading functionality."""
    print("\nTesting data loading...")
    
    data_file = "Dataset/WHO Life Expectancy Descriptive Statistics - Raw Data.csv"
    
    if not os.path.exists(data_file):
        print(f"❌ Data file not found: {data_file}")
        return False
    
    try:
        preprocessor = DataPreprocessor()
        df = preprocessor.load_and_clean_data(data_file)
        
        print(f"✅ Data loaded successfully")
        print(f"   Shape: {df.shape}")
        print(f"   Columns: {len(df.columns)}")
        print(f"   Target variable present: {'Life expectancy' in df.columns}")
        
        return True, df, preprocessor
        
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        return False, None, None

def test_preprocessing(df, preprocessor):
    """Test preprocessing pipeline."""
    print("\nTesting preprocessing pipeline...")
    
    try:
        # Test missing value analysis
        missing_stats = preprocessor.analyze_missing_values(df)
        print("✅ Missing value analysis completed")
        
        # Test complete preprocessing
        X_scaled, y, feature_names, df_processed = preprocessor.prepare_final_dataset(df)
        
        print(f"✅ Preprocessing completed successfully")
        print(f"   Features shape: {X_scaled.shape}")
        print(f"   Target shape: {y.shape}")
        print(f"   Number of features: {len(feature_names)}")
        
        return True, X_scaled, y, feature_names, df_processed
        
    except Exception as e:
        print(f"❌ Preprocessing failed: {e}")
        return False, None, None, None, None

def test_model_training(X, y, feature_names):
    """Test model training functionality."""
    print("\nTesting model training...")
    
    try:
        model_trainer = LifeExpectancyModel()
        
        # Test with a smaller subset for faster testing
        if X.shape[0] > 1000:
            indices = np.random.choice(X.shape[0], 1000, replace=False)
            X_subset = X[indices]
            y_subset = y[indices]
        else:
            X_subset, y_subset = X, y
        
        results = model_trainer.train_models(X_subset, y_subset, feature_names)
        
        print("✅ Model training completed successfully")
        print(f"   Models trained: {len(results)}")
        print(f"   Best model: {model_trainer.best_model_name}")
        
        # Test feature importance
        importance_df = model_trainer.analyze_feature_importance()
        if importance_df is not None:
            print("✅ Feature importance analysis completed")
        
        return True, results, model_trainer
        
    except Exception as e:
        print(f"❌ Model training failed: {e}")
        return False, None, None

def test_visualization(df_processed, results):
    """Test visualization functionality."""
    print("\nTesting visualization...")
    
    try:
        visualizer = LifeExpectancyVisualizer()
        
        # Test basic plots (without displaying)
        print("✅ Visualization module initialized")
        
        # Note: We won't actually display plots in the test to avoid blocking
        print("   (Plots would be generated in full application)")
        
        return True
        
    except Exception as e:
        print(f"❌ Visualization failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("LIFE EXPECTANCY PREDICTION - TEST SCRIPT")
    print("=" * 60)
    
    # Test 1: Imports
    if not test_imports():
        print("\n❌ Import test failed. Please check dependencies.")
        return False
    
    # Test 2: Data loading
    data_success, df, preprocessor = test_data_loading()
    if not data_success:
        print("\n❌ Data loading test failed.")
        return False
    
    # Test 3: Preprocessing
    prep_success, X_scaled, y, feature_names, df_processed = test_preprocessing(df, preprocessor)
    if not prep_success:
        print("\n❌ Preprocessing test failed.")
        return False
    
    # Test 4: Model training
    model_success, results, model_trainer = test_model_training(X_scaled, y, feature_names)
    if not model_success:
        print("\n❌ Model training test failed.")
        return False
    
    # Test 5: Visualization
    viz_success = test_visualization(df_processed, results)
    if not viz_success:
        print("\n❌ Visualization test failed.")
        return False
    
    # All tests passed
    print("\n" + "=" * 60)
    print("🎉 ALL TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)
    print("\nThe application is ready to use!")
    print("\nTo run the full application:")
    print("  python main.py")
    print("\nTo run the web application:")
    print("  streamlit run streamlit_app.py")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 