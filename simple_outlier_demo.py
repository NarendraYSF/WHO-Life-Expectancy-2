#!/usr/bin/env python3
"""
Simple Outlier Detection Demo
============================

This script demonstrates outlier detection and shows exactly what data 
was recognized as outliers.
"""

import pandas as pd
import numpy as np
from outlier_detection import OutlierDetectionSystem
from data_preprocessing import DataPreprocessor

def main():
    print("=" * 70)
    print("SIMPLE OUTLIER DETECTION DEMONSTRATION")
    print("=" * 70)
    
    # Load data
    print("\n1. Loading dataset...")
    preprocessor = DataPreprocessor()
    data_file = "Dataset/WHO Life Expectancy Descriptive Statistics - Raw Data.csv"
    
    df = preprocessor.load_and_clean_data(data_file)
    print(f"✅ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Initialize outlier detection
    outlier_system = OutlierDetectionSystem()
    
    # Run outlier analysis
    print("\n2. Running outlier analysis...")
    outlier_analysis = outlier_system.comprehensive_outlier_analysis(
        df, 
        target_column='Life expectancy'
    )
    
    # Show results
    print("\n" + "=" * 70)
    print("OUTLIER DETECTION RESULTS")
    print("=" * 70)
    
    # Target variable outliers
    print("\n📊 LIFE EXPECTANCY OUTLIERS:")
    print("-" * 50)
    
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
    
    # Feature-wise outliers summary
    print("\n\n📈 FEATURE-WISE OUTLIERS SUMMARY:")
    print("-" * 50)
    
    for feature, methods in outlier_analysis['feature_outliers'].items():
        total_outliers = sum([results['outlier_count'] for results in methods.values()])
        if total_outliers > 0:
            print(f"\n🔍 Feature: {feature}")
            for method, results in methods.items():
                if results['outlier_count'] > 0:
                    print(f"   {method}: {results['outlier_count']} outliers ({results['outlier_percentage']:.2f}%)")
    
    # Multivariate outliers summary
    print("\n\n🌐 MULTIVARIATE OUTLIERS SUMMARY:")
    print("-" * 50)
    
    for method, results in outlier_analysis['multivariate_outliers'].items():
        print(f"\n🔍 Method: {method.upper()}")
        print(f"   Outliers detected: {results['outlier_count']} ({results['outlier_percentage']:.2f}%)")
        
        if results['outlier_count'] > 0:
            outlier_indices = results['outlier_indices']
            
            print(f"   Sample outlier records:")
            for i, idx in enumerate(outlier_indices[:5]):  # Show first 5
                country = df.iloc[idx]['Country']
                year = df.iloc[idx]['Year']
                life_exp = df.iloc[idx]['Life expectancy']
                status = df.iloc[idx]['Status']
                region = df.iloc[idx]['Region']
                
                print(f"     {i+1:2d}. {country:20s} ({year}) - {life_exp:5.1f} years")
                print(f"         Status: {status}, Region: {region}")
            
            if len(outlier_indices) > 5:
                print(f"     ... and {len(outlier_indices) - 5} more outliers")
    
    # Create detailed report
    print("\n\n💾 CREATING DETAILED OUTLIER REPORT...")
    print("-" * 50)
    
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
        outlier_df.to_csv('outlier_report.csv', index=False)
        print(f"✅ Detailed outlier report saved to 'outlier_report.csv'")
        print(f"   Total outlier records: {len(outlier_df)}")
        
        # Show summary by country
        print(f"\n   Top countries with outliers:")
        country_counts = outlier_df['Country'].value_counts()
        for country, count in country_counts.head(10).items():
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
    
    # Recommendations
    print("\n\n💡 RECOMMENDATIONS:")
    print("-" * 50)
    
    summary = outlier_analysis['summary']
    for rec in summary['recommendations']:
        if "High outlier percentage" in rec:
            print(f"⚠️  {rec}")
        elif "Moderate outlier percentage" in rec:
            print(f"ℹ️  {rec}")
        else:
            print(f"✅ {rec}")
    
    print("\n" + "=" * 70)
    print("OUTLIER ANALYSIS COMPLETED!")
    print("=" * 70)
    print("\nFiles created:")
    print("- outlier_report.csv: Complete outlier information")
    print("\nTo view the report, open 'outlier_report.csv' in Excel or any spreadsheet application.")

if __name__ == "__main__":
    main() 