#!/usr/bin/env python3
"""
Simple Outlier Detection Demonstration
=====================================

This script shows exactly what data was recognized as outliers.
"""

import pandas as pd
import numpy as np
from outlier_detection import OutlierDetectionSystem
from data_preprocessing import DataPreprocessor

def main():
    print("=" * 60)
    print("OUTLIER DETECTION DEMONSTRATION")
    print("=" * 60)
    
    # Load data
    print("\n1. Loading data...")
    preprocessor = DataPreprocessor()
    data_file = "Dataset/WHO Life Expectancy Descriptive Statistics - Raw Data.csv"
    
    df = preprocessor.load_and_clean_data(data_file)
    print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Initialize outlier detection
    outlier_system = OutlierDetectionSystem()
    
    # Run outlier analysis
    print("\n2. Running outlier analysis...")
    outlier_analysis = outlier_system.comprehensive_outlier_analysis(
        df, 
        target_column='Life expectancy'
    )
    
    # Show detailed outlier information
    print("\n" + "=" * 60)
    print("DETAILED OUTLIER INFORMATION")
    print("=" * 60)
    
    # Target variable outliers
    print("\n📊 LIFE EXPECTANCY OUTLIERS:")
    print("-" * 40)
    
    target_data = df['Life expectancy'].dropna()
    
    for method, results in outlier_analysis['target_outliers'].items():
        print(f"\n🔍 Method: {method.upper()}")
        print(f"   Outliers detected: {results['outlier_count']} ({results['outlier_percentage']:.2f}%)")
        
        if results['outlier_count'] > 0:
            outlier_indices = results['outlier_indices']
            outlier_values = target_data.iloc[outlier_indices]
            outlier_scores = results['outlier_scores'][outlier_indices]
            
            print(f"   Outlier details:")
            for i, (idx, value, score) in enumerate(zip(outlier_indices, outlier_values, outlier_scores)):
                # Find the original row
                original_idx = df.index[df['Life expectancy'] == value].tolist()
                if original_idx:
                    country = df.loc[original_idx[0], 'Country']
                    year = df.loc[original_idx[0], 'Year']
                    status = df.loc[original_idx[0], 'Status']
                    region = df.loc[original_idx[0], 'Region']
                    
                    print(f"     {i+1:2d}. {country:20s} ({year}) - {value:5.1f} years (score: {score:6.3f})")
                    print(f"         Status: {status}, Region: {region}")
                else:
                    print(f"     {i+1:2d}. Index {idx:3d} - {value:5.1f} years (score: {score:6.3f})")
            
            # Show statistics
            print(f"   Statistics:")
            print(f"     - Outlier range: {outlier_values.min():.1f} - {outlier_values.max():.1f} years")
            print(f"     - Mean outlier value: {outlier_values.mean():.1f} years")
            print(f"     - Overall data range: {target_data.min():.1f} - {target_data.max():.1f} years")
            print(f"     - Overall mean: {target_data.mean():.1f} years")
        else:
            print("   No outliers detected")
    
    # Feature-wise outliers
    print("\n\n📈 FEATURE-WISE OUTLIERS:")
    print("-" * 40)
    
    for feature, methods in outlier_analysis['feature_outliers'].items():
        print(f"\n🔍 Feature: {feature}")
        
        for method, results in methods.items():
            if results['outlier_count'] > 0:
                print(f"   Method: {method}")
                print(f"   Outliers: {results['outlier_count']} ({results['outlier_percentage']:.2f}%)")
                
                # Get outlier values
                feature_data = df[feature].dropna()
                outlier_indices = results['outlier_indices']
                outlier_values = feature_data.iloc[outlier_indices]
                
                print(f"   Outlier range: {outlier_values.min():.3f} - {outlier_values.max():.3f}")
                print(f"   Overall range: {feature_data.min():.3f} - {feature_data.max():.3f}")
    
    # Multivariate outliers
    print("\n\n🌐 MULTIVARIATE OUTLIERS:")
    print("-" * 40)
    
    for method, results in outlier_analysis['multivariate_outliers'].items():
        print(f"\n🔍 Method: {method.upper()}")
        print(f"   Outliers detected: {results['outlier_count']} ({results['outlier_percentage']:.2f}%)")
        
        if results['outlier_count'] > 0:
            outlier_indices = results['outlier_indices']
            
            print(f"   Sample outlier records:")
            for i, idx in enumerate(outlier_indices[:10]):  # Show first 10
                country = df.iloc[idx]['Country']
                year = df.iloc[idx]['Year']
                life_exp = df.iloc[idx]['Life expectancy']
                status = df.iloc[idx]['Status']
                region = df.iloc[idx]['Region']
                
                print(f"     {i+1:2d}. {country:20s} ({year}) - {life_exp:5.1f} years")
                print(f"         Status: {status}, Region: {region}")
            
            if len(outlier_indices) > 10:
                print(f"     ... and {len(outlier_indices) - 10} more outliers")
    
    # Create outlier report
    print("\n\n💾 CREATING OUTLIER REPORT...")
    print("-" * 40)
    
    outlier_report = []
    
    for method, results in outlier_analysis['target_outliers'].items():
        if results['outlier_count'] > 0:
            outlier_indices = results['outlier_indices']
            outlier_values = target_data.iloc[outlier_indices]
            outlier_scores = results['outlier_scores'][outlier_indices]
            
            for i, (idx, value, score) in enumerate(zip(outlier_indices, outlier_values, outlier_scores)):
                original_idx = df.index[df['Life expectancy'] == value].tolist()
                if original_idx:
                    country = df.loc[original_idx[0], 'Country']
                    year = df.loc[original_idx[0], 'Year']
                    status = df.loc[original_idx[0], 'Status']
                    region = df.loc[original_idx[0], 'Region']
                    
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
        outlier_df.to_csv('detailed_outlier_report.csv', index=False)
        print(f"✅ Detailed outlier report saved to 'detailed_outlier_report.csv'")
        print(f"   Total outlier records: {len(outlier_df)}")
        
        # Show summary by country
        print(f"\n   Outliers by country:")
        country_counts = outlier_df['Country'].value_counts()
        for country, count in country_counts.head(15).items():
            print(f"     {country:20s}: {count:2d} outliers")
        
        # Show summary by method
        print(f"\n   Outliers by detection method:")
        method_counts = outlier_df['Method'].value_counts()
        for method, count in method_counts.items():
            print(f"     {method:15s}: {count:2d} outliers")
        
        # Show summary by region
        print(f"\n   Outliers by region:")
        region_counts = outlier_df['Region'].value_counts()
        for region, count in region_counts.items():
            print(f"     {region:15s}: {count:2d} outliers")
    
    print("\n" + "=" * 60)
    print("OUTLIER ANALYSIS COMPLETED!")
    print("=" * 60)
    print("\nFiles created:")
    print("- detailed_outlier_report.csv: Complete outlier information")
    print("\nTo view the report, open 'detailed_outlier_report.csv' in Excel or any spreadsheet application.")

if __name__ == "__main__":
    main() 