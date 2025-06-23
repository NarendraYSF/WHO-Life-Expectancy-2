#!/usr/bin/env python3
"""
Test script for Outlier Detection System
========================================

This script tests the outlier detection functionality.
"""

import numpy as np
import pandas as pd
from outlier_detection import OutlierDetectionSystem

def test_outlier_detection():
    """Test the outlier detection system."""
    print("Testing Outlier Detection System...")
    
    # Create sample data with outliers
    np.random.seed(42)
    normal_data = np.random.normal(50, 10, 100)
    outlier_data = np.array([10, 90, 5, 95])  # Clear outliers
    test_data = np.concatenate([normal_data, outlier_data])
    
    # Initialize outlier detection system
    outlier_system = OutlierDetectionSystem()
    
    # Test univariate outlier detection
    print("\n1. Testing univariate outlier detection...")
    
    methods = ['zscore', 'iqr', 'modified_zscore']
    for method in methods:
        try:
            results = outlier_system.detect_univariate_outliers(test_data, method=method)
            print(f"   {method}: {results['outlier_count']} outliers detected")
        except Exception as e:
            print(f"   {method}: Error - {e}")
    
    # Test multivariate outlier detection
    print("\n2. Testing multivariate outlier detection...")
    
    # Create 2D data with outliers
    X = np.random.normal(0, 1, (100, 2))
    X_outliers = np.array([[5, 5], [-5, -5], [10, 0], [0, 10]])  # Clear outliers
    X_with_outliers = np.vstack([X, X_outliers])
    
    try:
        results = outlier_system.detect_multivariate_outliers(X_with_outliers, method='isolation_forest')
        print(f"   Isolation Forest: {results['outlier_count']} outliers detected")
    except Exception as e:
        print(f"   Isolation Forest: Error - {e}")
    
    print("\n✅ Outlier detection system test completed!")

if __name__ == "__main__":
    test_outlier_detection() 