import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    """
    A comprehensive data preprocessing class for the WHO Life Expectancy dataset.
    Handles missing values, data type conversions, feature engineering, and scaling.
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder_status = LabelEncoder()
        self.label_encoder_region = LabelEncoder()
        self.imputer = SimpleImputer(strategy='median')
        self.is_fitted = False
        
    def load_and_clean_data(self, file_path):
        """
        Load the dataset and perform initial cleaning.
        
        Args:
            file_path (str): Path to the CSV file
            
        Returns:
            pd.DataFrame: Cleaned dataset
        """
        print("Loading and cleaning dataset...")
        
        # Load the dataset with explicit data types to avoid PyArrow issues
        try:
            # First try to load with default settings
            df = pd.read_csv(file_path)
        except Exception as e:
            print(f"Warning: Error loading with default settings: {e}")
            # Try with different engine
            try:
                df = pd.read_csv(file_path, engine='python')
            except Exception as e2:
                print(f"Warning: Error with python engine: {e2}")
                # Last resort: try with error handling
                df = pd.read_csv(file_path, on_bad_lines='skip', encoding='utf-8')
        
        # Display initial info
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Clean column names
        df.columns = df.columns.str.strip()
        
        # Handle numeric values with commas (European format)
        numeric_columns = ['Life expectancy', 'Adult Mortality', 'infant deaths', 'Alcohol', 
                          'percentage expenditure', 'Hepatitis B', 'Measles', 'BMI', 
                          'under-five deaths', 'Polio', 'Total expenditure', 'Diphtheria', 
                          'HIV/AIDS', 'GDP', 'Population', 'thinness 1-19 years', 
                          'thinness 5-9 years', 'Income composition of resources', 'Schooling']
        
        for col in numeric_columns:
            if col in df.columns:
                try:
                    # Convert to string first, then handle commas
                    df[col] = df[col].astype(str)
                    # Replace commas with dots for decimal values
                    df[col] = df[col].str.replace(',', '.')
                    # Handle any remaining problematic characters
                    df[col] = df[col].str.replace(' ', '')  # Remove spaces
                    df[col] = df[col].str.replace('nan', '')  # Handle 'nan' strings
                    # Convert to numeric, errors='coerce' will handle non-numeric values
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except Exception as e:
                    print(f"Warning: Error converting column {col}: {e}")
                    # Set to NaN if conversion fails
                    df[col] = np.nan
        
        # Handle specific problematic values
        df = self._handle_special_values(df)
        
        # Ensure all numeric columns are properly typed
        for col in df.select_dtypes(include=[np.number]).columns:
            try:
                df[col] = df[col].astype('float64')
            except Exception as e:
                print(f"Warning: Could not convert {col} to float64: {e}")
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        print("Data cleaning completed!")
        return df
    
    def _handle_special_values(self, df):
        """
        Handle special values and inconsistencies in the dataset.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Cleaned dataframe
        """
        # Replace empty strings with NaN
        df = df.replace('', np.nan)
        
        # Handle specific problematic patterns
        # Some values have scientific notation or other formats
        for col in df.select_dtypes(include=[np.number]).columns:
            # Replace infinite values with NaN
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        
        return df
    
    def analyze_missing_values(self, df):
        """
        Analyze missing values in the dataset.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            dict: Missing value statistics
        """
        print("\n=== Missing Values Analysis ===")
        
        missing_stats = {}
        total_rows = len(df)
        
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            missing_percentage = (missing_count / total_rows) * 100
            
            missing_stats[col] = {
                'missing_count': missing_count,
                'missing_percentage': missing_percentage
            }
            
            if missing_count > 0:
                print(f"{col}: {missing_count} ({missing_percentage:.2f}%)")
        
        return missing_stats
    
    def feature_engineering(self, df):
        """
        Perform feature engineering on the dataset.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with engineered features
        """
        print("\n=== Feature Engineering ===")
        
        # Create a copy to avoid modifying original data
        df_engineered = df.copy()
        
        # 1. Create binary features for Status
        df_engineered['is_developed'] = (df_engineered['Status'] == 'Developed').astype(int)
        
        # 2. Create year-based features
        df_engineered['year_normalized'] = (df_engineered['Year'] - 2000) / 15  # Normalize years
        
        # 3. Create health-related composite features
        df_engineered['health_expenditure_per_capita'] = (
            df_engineered['percentage expenditure'] / df_engineered['Population']
        ).fillna(0)
        
        # 4. Create mortality rate features
        df_engineered['total_mortality_rate'] = (
            df_engineered['Adult Mortality'] + df_engineered['infant deaths'] + 
            df_engineered['under-five deaths']
        )
        
        # 5. Create vaccination coverage feature
        df_engineered['vaccination_coverage'] = (
            df_engineered['Hepatitis B'] + df_engineered['Polio'] + df_engineered['Diphtheria']
        ) / 3
        
        # 6. Create thinness composite feature
        df_engineered['thinness_composite'] = (
            df_engineered['thinness 1-19 years'] + df_engineered['thinness 5-9 years']
        ) / 2
        
        # 7. Create economic indicators
        df_engineered['gdp_per_capita'] = (
            df_engineered['GDP'] / df_engineered['Population']
        ).fillna(0)
        
        print("Feature engineering completed!")
        return df_engineered
    
    def encode_categorical_features(self, df):
        """
        Encode categorical features using label encoding.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with encoded categorical features
        """
        print("\n=== Categorical Feature Encoding ===")
        
        df_encoded = df.copy()
        
        # Encode Status
        if 'Status' in df_encoded.columns:
            df_encoded['Status_encoded'] = self.label_encoder_status.fit_transform(
                df_encoded['Status'].fillna('Unknown')
            )
        
        # Encode Region
        if 'Region' in df_encoded.columns:
            df_encoded['Region_encoded'] = self.label_encoder_region.fit_transform(
                df_encoded['Region'].fillna('Unknown')
            )
        
        print("Categorical encoding completed!")
        return df_encoded
    
    def handle_missing_values(self, df, strategy='median'):
        """
        Handle missing values using specified strategy.
        
        Args:
            df (pd.DataFrame): Input dataframe
            strategy (str): Imputation strategy ('mean', 'median', 'most_frequent')
            
        Returns:
            pd.DataFrame: Dataframe with imputed values
        """
        print(f"\n=== Handling Missing Values (Strategy: {strategy}) ===")
        
        df_imputed = df.copy()
        
        # Separate numeric and categorical columns
        numeric_columns = df_imputed.select_dtypes(include=[np.number]).columns
        categorical_columns = df_imputed.select_dtypes(include=['object']).columns
        
        # Impute numeric columns
        if len(numeric_columns) > 0:
            imputer_numeric = SimpleImputer(strategy=strategy)
            df_imputed[numeric_columns] = imputer_numeric.fit_transform(df_imputed[numeric_columns])
        
        # Impute categorical columns
        if len(categorical_columns) > 0:
            imputer_categorical = SimpleImputer(strategy='most_frequent')
            df_imputed[categorical_columns] = imputer_categorical.fit_transform(df_imputed[categorical_columns])
        
        print("Missing value imputation completed!")
        return df_imputed
    
    def scale_features(self, df, target_column='Life expectancy'):
        """
        Scale numerical features using StandardScaler.
        
        Args:
            df (pd.DataFrame): Input dataframe
            target_column (str): Name of the target column
            
        Returns:
            tuple: (scaled_features, target_values, feature_names)
        """
        print("\n=== Feature Scaling ===")
        
        # Separate features and target
        feature_columns = [col for col in df.columns if col != target_column and col not in 
                          ['Country', 'Year', 'Status', 'Region']]
        
        X = df[feature_columns].select_dtypes(include=[np.number])
        y = df[target_column]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        self.is_fitted = True
        
        print(f"Features scaled: {X.shape[1]} features")
        print("Feature scaling completed!")
        
        return X_scaled, y.values, X.columns.tolist()
    
    def get_feature_importance_dataframe(self, feature_names, coefficients):
        """
        Create a dataframe with feature importance information.
        
        Args:
            feature_names (list): List of feature names
            coefficients (array): Model coefficients
            
        Returns:
            pd.DataFrame: Feature importance dataframe
        """
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Coefficient': coefficients,
            'Abs_Coefficient': np.abs(coefficients)
        })
        
        importance_df = importance_df.sort_values('Abs_Coefficient', ascending=False)
        return importance_df
    
    def prepare_final_dataset(self, df, target_column='Life expectancy'):
        """
        Prepare the final dataset for modeling by combining all preprocessing steps.
        
        Args:
            df (pd.DataFrame): Input dataframe
            target_column (str): Name of the target column
            
        Returns:
            tuple: (X_scaled, y, feature_names, preprocessed_df)
        """
        print("\n=== Complete Data Preprocessing Pipeline ===")
        
        # Step 1: Feature engineering
        df_engineered = self.feature_engineering(df)
        
        # Step 2: Encode categorical features
        df_encoded = self.encode_categorical_features(df_engineered)
        
        # Step 3: Handle missing values
        df_imputed = self.handle_missing_values(df_encoded)
        
        # Step 4: Scale features
        X_scaled, y, feature_names = self.scale_features(df_imputed, target_column)
        
        print("Complete preprocessing pipeline finished!")
        
        return X_scaled, y, feature_names, df_imputed 