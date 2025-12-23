# src/feature_engineering.py

from datetime import datetime, timedelta
import pandas as pd

class FeatureEngineer:
    """Creates engineered features for fraud detection."""
    
    def __init__(self):
        self.features_created = []
    
    def create_time_features(self, df):
        """Create time-based features from timestamps."""
        print("\nCreating time-based features...")
        
        df_copy = df.copy()
        
        # 1. Time since signup (in hours)
        df_copy['time_since_signup'] = (
            df_copy['purchase_time'] - df_copy['signup_time']
        ).dt.total_seconds() / 3600  # Convert to hours
        
        # 2. Hour of day
        df_copy['hour_of_day'] = df_copy['purchase_time'].dt.hour
        
        # 3. Day of week
        df_copy['day_of_week'] = df_copy['purchase_time'].dt.dayofweek  # Monday=0, Sunday=6
        
        # 4. Is weekend?
        df_copy['is_weekend'] = df_copy['day_of_week'].isin([5, 6]).astype(int)
        
        # 5. Month and day
        df_copy['month'] = df_copy['purchase_time'].dt.month
        df_copy['day'] = df_copy['purchase_time'].dt.day
        
        self.features_created.extend(['time_since_signup', 'hour_of_day', 'day_of_week', 
                                      'is_weekend', 'month', 'day'])
        
        return df_copy
    
    def create_transaction_velocity(self, df, user_id_col='user_id', 
                                    time_col='purchase_time', window_hours=[1, 24, 168]):
        """
        Calculate transaction frequency per user in different time windows.
        
        Args:
            df: DataFrame with transactions
            user_id_col: Column name for user identifier
            time_col: Column name for transaction timestamp
            window_hours: List of time windows in hours [1h, 24h, 7 days]
        """
        print(f"\nCreating transaction velocity features...")
        
        df_copy = df.copy()
        df_copy = df_copy.sort_values([user_id_col, time_col])
        
        # For each time window
        for window in window_hours:
            feature_name = f'transactions_last_{window}h'
            
            # Calculate transactions in the last 'window' hours for each user
            df_copy[feature_name] = df_copy.groupby(user_id_col).apply(
                lambda x: x.set_index(time_col).rolling(f'{window}h').count()[user_id_col].values
            ).explode().reset_index(drop=True)
            
            self.features_created.append(feature_name)
            print(f"  Created: {feature_name}")
        
        return df_copy
    
    def create_purchase_patterns(self, df):
        """Create purchase-related features."""
        print("\nCreating purchase pattern features...")
        
        df_copy = df.copy()
        
        # 1. Purchase value categories
        df_copy['purchase_value_category'] = pd.cut(
            df_copy['purchase_value'],
            bins=[0, 50, 200, 500, float('inf')],
            labels=['low', 'medium', 'high', 'very_high']
        )
        
        # 2. Is high value transaction? (above 90th percentile)
        threshold = df_copy['purchase_value'].quantile(0.9)
        df_copy['is_high_value'] = (df_copy['purchase_value'] > threshold).astype(int)
        
        # 3. Purchase value to average ratio (per user)
        user_avg = df_copy.groupby('user_id')['purchase_value'].transform('mean')
        df_copy['purchase_to_user_avg_ratio'] = df_copy['purchase_value'] / (user_avg + 1e-6)
        
        self.features_created.extend(['purchase_value_category', 'is_high_value', 
                                      'purchase_to_user_avg_ratio'])
        
        return df_copy
    
    def create_device_features(self, df):
        """Create device-related features."""
        print("\nCreating device features...")
        
        df_copy = df.copy()
        
        # Number of users per device (potential device sharing)
        device_users = df_copy.groupby('device_id')['user_id'].nunique().reset_index()
        device_users.columns = ['device_id', 'users_per_device']
        df_copy = pd.merge(df_copy, device_users, on='device_id', how='left')
        
        # Transactions per device
        device_transactions = df_copy.groupby('device_id').size().reset_index()
        device_transactions.columns = ['device_id', 'transactions_per_device']
        df_copy = pd.merge(df_copy, device_transactions, on='device_id', how='left')
        
        self.features_created.extend(['users_per_device', 'transactions_per_device'])
        
        return df_copy