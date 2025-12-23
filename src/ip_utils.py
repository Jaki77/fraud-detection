import pandas as pd

def ip_to_int(ip_address):
    """Convert IP address to integer for range lookup."""
    try:
        if isinstance(ip_address, str) and ip_address != 'nan':
            parts = ip_address.split('.')
            if len(parts) == 4 and all(part.isdigit() for part in parts):
                return (int(parts[0]) << 24) + (int(parts[1]) << 16) + (int(parts[2]) << 8) + int(parts[3])
    except (ValueError, AttributeError):
        pass
    return None  # Return None instead of np.nan for better handling

def merge_with_ip_country(transactions_df, ip_country_df):
    """
    Merge transaction data with IP-country mapping using range-based lookup.
    """
    print("\nMerging with IP-country data...")
    
    # Convert IPs to integers
    transactions_df['ip_int'] = transactions_df['ip_address'].apply(ip_to_int)
    
    # Check for failed conversions
    failed_conversions = transactions_df['ip_int'].isnull().sum()
    if failed_conversions > 0:
        print(f"Warning: {failed_conversions} IP addresses could not be converted")
    
    # Remove rows where IP conversion failed
    transactions_df_clean = transactions_df.dropna(subset=['ip_int']).copy()
    transactions_df_clean['ip_int'] = transactions_df_clean['ip_int'].astype('int64')
    
    # Sort both DataFrames for merge_asof
    ip_country_df = ip_country_df.sort_values('lower_bound_ip_address')
    transactions_df_clean = transactions_df_clean.sort_values('ip_int')
    
    # Ensure the IP country data has integer types
    ip_country_df['lower_bound_ip_address'] = ip_country_df['lower_bound_ip_address'].astype('int64')
    ip_country_df['upper_bound_ip_address'] = ip_country_df['upper_bound_ip_address'].astype('int64')
    
    # Use merge_asof for range-based lookup
    try:
        merged_df = pd.merge_asof(
            transactions_df_clean,
            ip_country_df,
            left_on='ip_int',
            right_on='lower_bound_ip_address',
            direction='backward'
        )
        
        # Filter only valid merges (where IP falls within range)
        mask = (merged_df['ip_int'] >= merged_df['lower_bound_ip_address']) & \
               (merged_df['ip_int'] <= merged_df['upper_bound_ip_address'])
        merged_df = merged_df[mask].copy()
        
        # Drop helper columns
        columns_to_drop = ['ip_int', 'lower_bound_ip_address', 'upper_bound_ip_address']
        merged_df = merged_df.drop([col for col in columns_to_drop if col in merged_df.columns], axis=1)
        
        print(f"Original transactions: {len(transactions_df)}")
        print(f"Successfully merged: {len(merged_df)}")
        print(f"Rows with country match: {merged_df['country'].notnull().sum()} / {len(merged_df)}")
        
        # Check if we lost too many rows
        if len(merged_df) < len(transactions_df) * 0.9:  # Less than 90% success rate
            print("Warning: Low merge success rate. Consider alternative merging strategy.")
        
        return merged_df
        
    except Exception as e:
        print(f"Error during merge_asof: {e}")
        print("Trying alternative merge strategy...")
        
        # Alternative: Use interval index for merging
        return alternative_ip_merge(transactions_df_clean, ip_country_df)
    

def alternative_ip_merge(transactions_df, ip_country_df):
    """
    Alternative method using pandas IntervalIndex for IP range matching.
    """
    print("Using alternative interval-based merge...")
    
    # Create interval index from IP ranges
    intervals = pd.IntervalIndex.from_arrays(
        ip_country_df['lower_bound_ip_address'],
        ip_country_df['upper_bound_ip_address'],
        closed='both'
    )
    
    # Create a mapping Series
    ip_mapping = pd.Series(
        ip_country_df['country'].values,
        index=intervals
    )
    
    # Function to find country for each IP
    def find_country_for_ip(ip_int):
        try:
            # Find which interval contains the IP
            interval_idx = intervals.get_indexer([ip_int])[0]
            if interval_idx != -1:
                return ip_mapping.iloc[interval_idx]
        except:
            pass
        return None
    
    # Apply mapping
    transactions_df['country'] = transactions_df['ip_int'].apply(find_country_for_ip)
    
    print(f"Rows with country match: {transactions_df['country'].notnull().sum()} / {len(transactions_df)}")
    
    # Drop the helper column
    if 'ip_int' in transactions_df.columns:
        transactions_df = transactions_df.drop('ip_int', axis=1)
    
    return transactions_df

def validate_ip_addresses(df, ip_col='ip_address'):
    """Validate IP addresses in the dataset."""
    print("\nValidating IP addresses...")
    
    # Check for common issues
    invalid_ips = []
    valid_patterns = []
    
    for ip in df[ip_col].unique():
        if isinstance(ip, str):
            parts = ip.split('.')
            if len(parts) == 4:
                try:
                    # Check if all parts are numbers between 0-255
                    if all(0 <= int(p) <= 255 for p in parts if p.isdigit()):
                        valid_patterns.append(ip)
                        continue
                except ValueError:
                    pass
        invalid_ips.append(ip)
    
    print(f"Valid IP patterns: {len(valid_patterns)}")
    print(f"Invalid IP patterns: {len(invalid_ips)}")
    
    if invalid_ips:
        print("\nSample invalid IPs:")
        for ip in invalid_ips[:5]:
            print(f"  - {ip}")
    
    return valid_patterns, invalid_ips