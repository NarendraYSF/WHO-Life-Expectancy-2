#!/usr/bin/env python3
"""
Outlier Analysis Demonstration
==============================

This script demonstrates the outlier detection system and shows exactly what data
was recognized as outliers with detailed information.
"""

import pandas as pd
import numpy as np
from outlier_detection import OutlierDetectionSystem
from data_preprocessing import DataPreprocessor

def demonstrate_outlier_detection():
    """Demonstrate outlier detection with detailed outlier information."""
    
    print("=" * 60)
    print("OUTLIER DETECTION DEMONSTRATION")
    print("=" * 60)
    
    # Load and preprocess data
    print("\n1. Loading and preprocessing data...")
    preprocessor = DataPreprocessor()
    data_file = "Dataset/WHO Life Expectancy Descriptive Statistics - Raw Data.csv"
    
    df = preprocessor.load_and_clean_data(data_file)
    X_scaled, y, feature_names, df_processed = preprocessor.prepare_final_dataset(df)
    
    # Initialize outlier detection system
    outlier_system = OutlierDetectionSystem()
    
    # Perform comprehensive outlier analysis
    print("\n2. Performing comprehensive outlier analysis...")
    outlier_analysis = outlier_system.comprehensive_outlier_analysis(
        df_processed, 
        target_column='Life expectancy'
    )
    
    # Display detailed outlier information
    print("\n" + "=" * 60)
    print("DETAILED OUTLIER ANALYSIS RESULTS")
    print("=" * 60)
    
    # 1. Target Variable Outliers
    print("\n📊 TARGET VARIABLE OUTLIERS (Life Expectancy)")
    print("-" * 50)
    
    target_data = df_processed['Life expectancy'].dropna()
    
    for method, results in outlier_analysis['target_outliers'].items():
        print(f"\n🔍 Method: {method.upper()}")
        print(f"   Outliers detected: {results['outlier_count']} ({results['outlier_percentage']:.2f}%)")
        
        if results['outlier_count'] > 0:
            # Get outlier indices and values
            outlier_indices = results['outlier_indices']
            outlier_values = target_data.iloc[outlier_indices]
            outlier_scores = results['outlier_scores'][outlier_indices]
            
            print(f"   Outlier values:")
            for i, (idx, value, score) in enumerate(zip(outlier_indices, outlier_values, outlier_scores)):
                # Get country and year for this outlier
                original_idx = df_processed.index[df_processed['Life expectancy'] == value].tolist()
                if original_idx:
                    country = df_processed.loc[original_idx[0], 'Country']
                    year = df_processed.loc[original_idx[0], 'Year']
                    print(f"     {i+1}. {country} ({year}): {value:.1f} years (score: {score:.3f})")
                else:
                    print(f"     {i+1}. Index {idx}: {value:.1f} years (score: {score:.3f})")
            
            # Show statistics
            print(f"   Statistics:")
            print(f"     - Outlier range: {outlier_values.min():.1f} - {outlier_values.max():.1f} years")
            print(f"     - Mean outlier value: {outlier_values.mean():.1f} years")
            print(f"     - Overall data range: {target_data.min():.1f} - {target_data.max():.1f} years")
            print(f"     - Overall mean: {target_data.mean():.1f} years")
        else:
            print("   No outliers detected")
    
    # 2. Feature-wise Outliers
    print("\n\n📈 FEATURE-WISE OUTLIER ANALYSIS")
    print("-" * 50)
    
    for feature, methods in outlier_analysis['feature_outliers'].items():
        print(f"\n🔍 Feature: {feature}")
        
        for method, results in methods.items():
            if results['outlier_count'] > 0:
                print(f"   Method: {method}")
                print(f"   Outliers: {results['outlier_count']} ({results['outlier_percentage']:.2f}%)")
                
                # Get outlier values
                feature_data = df_processed[feature].dropna()
                outlier_indices = results['outlier_indices']
                outlier_values = feature_data.iloc[outlier_indices]
                
                print(f"   Outlier range: {outlier_values.min():.3f} - {outlier_values.max():.3f}")
                print(f"   Overall range: {feature_data.min():.3f} - {feature_data.max():.3f}")
    
    # 3. Multivariate Outliers
    print("\n\n🌐 MULTIVARIATE OUTLIER ANALYSIS")
    print("-" * 50)
    
    for method, results in outlier_analysis['multivariate_outliers'].items():
        print(f"\n🔍 Method: {method.upper()}")
        print(f"   Outliers detected: {results['outlier_count']} ({results['outlier_percentage']:.2f}%)")
        
        if results['outlier_count'] > 0:
            # Get outlier indices
            outlier_indices = results['outlier_indices']
            
            print(f"   Sample outlier records:")
            for i, idx in enumerate(outlier_indices[:5]):  # Show first 5 outliers
                country = df_processed.iloc[idx]['Country']
                year = df_processed.iloc[idx]['Year']
                life_exp = df_processed.iloc[idx]['Life expectancy']
                print(f"     {i+1}. {country} ({year}): Life expectancy = {life_exp:.1f} years")
            
            if len(outlier_indices) > 5:
                print(f"     ... and {len(outlier_indices) - 5} more outliers")
    
    # 4. Outlier Impact Analysis
    print("\n\n📊 OUTLIER IMPACT ANALYSIS")
    print("-" * 50)
    
    # Show impact of removing outliers using different methods
    for method, results in outlier_analysis['target_outliers'].items():
        if results['outlier_count'] > 0:
            print(f"\n🔍 Impact of removing {method} outliers:")
            
            # Calculate statistics before removal
            stats_before = {
                'count': len(target_data),
                'mean': target_data.mean(),
                'std': target_data.std(),
                'median': target_data.median(),
                'min': target_data.min(),
                'max': target_data.max()
            }
            
            # Calculate statistics after removal
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
            
            print(f"   Records: {stats_before['count']} → {stats_after['count']} (-{results['outlier_count']})")
            print(f"   Mean: {stats_before['mean']:.2f} → {stats_after['mean']:.2f} (change: {stats_after['mean'] - stats_before['mean']:+.2f})")
            print(f"   Std: {stats_before['std']:.2f} → {stats_after['std']:.2f} (change: {stats_after['std'] - stats_before['std']:+.2f})")
            print(f"   Range: {stats_before['max'] - stats_before['min']:.2f} → {stats_after['max'] - stats_after['min']:.2f}")
    
    # 5. Recommendations
    print("\n\n💡 RECOMMENDATIONS")
    print("-" * 50)
    
    summary = outlier_analysis['summary']
    for rec in summary['recommendations']:
        if "High outlier percentage" in rec:
            print(f"⚠️  {rec}")
        elif "Moderate outlier percentage" in rec:
            print(f"ℹ️  {rec}")
        else:
            print(f"✅ {rec}")
    
    # 6. Export outlier data
    print("\n\n💾 EXPORTING OUTLIER DATA")
    print("-" * 50)
    
    # Create detailed outlier report
    outlier_report = []
    
    for method, results in outlier_analysis['target_outliers'].items():
        if results['outlier_count'] > 0:
            outlier_indices = results['outlier_indices']
            outlier_values = target_data.iloc[outlier_indices]
            outlier_scores = results['outlier_scores'][outlier_indices]
            
            for i, (idx, value, score) in enumerate(zip(outlier_indices, outlier_values, outlier_scores)):
                original_idx = df_processed.index[df_processed['Life expectancy'] == value].tolist()
                if original_idx:
                    country = df_processed.loc[original_idx[0], 'Country']
                    year = df_processed.loc[original_idx[0], 'Year']
                    status = df_processed.loc[original_idx[0], 'Status']
                    region = df_processed.loc[original_idx[0], 'Region']
                    
                    outlier_report.append({
                        'Method': method,
                        'Country': country,
                        'Year': year,
                        'Status': status,
                        'Region': region,
                        'Life_Expectancy': value,
                        'Outlier_Score': score,
                        'Original_Index': original_idx[0]
                    })
    
    if outlier_report:
        outlier_df = pd.DataFrame(outlier_report)
        outlier_df.to_csv('outlier_report.csv', index=False)
        print(f"✅ Outlier report saved to 'outlier_report.csv'")
        print(f"   Total outlier records: {len(outlier_df)}")
        
        # Show summary by country
        print(f"\n   Outliers by country:")
        country_counts = outlier_df['Country'].value_counts()
        for country, count in country_counts.head(10).items():
            print(f"     {country}: {count} outliers")
    
    print("\n" + "=" * 60)
    print("OUTLIER ANALYSIS COMPLETED!")
    print("=" * 60)

if __name__ == "__main__":
    demonstrate_outlier_detection() 