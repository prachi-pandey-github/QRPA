import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import warnings
import os
from dotenv import load_dotenv
import urllib3

warnings.filterwarnings('ignore')
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Load environment variables from .env file
load_dotenv()

API_KEY = os.getenv('API_KEY')
BASE_URL = os.getenv('BASE_URL', 'https://financialmodelingprep.com/api/v3')

# Validate API key is loaded
if not API_KEY:
    raise ValueError("API_KEY not found. Please check your .env file contains API_KEY=your_api_key")

def generate_date_ranges(start_date, end_date, chunk_days=5):
    """Generate date ranges in chunks for optimal 1-minute data fetching"""
    date_ranges = []
    current_start = start_date
    
    while current_start < end_date:
        current_end = min(current_start + timedelta(days=chunk_days), end_date)
        date_ranges.append((current_start, current_end))
        current_start = current_end + timedelta(days=1)
        
    return date_ranges

def fetch_intraday_data(symbol, start_date, end_date, api_key):
    """Fetch 1-minute intraday data from FMP API"""
    url = f"{BASE_URL}/historical-chart/1min/{symbol}?from={start_date}&to={end_date}&apikey={api_key}"
    print(f"Fetching data from: {url}")
    response = requests.get(url, verify=False)
    if response.status_code == 200:
        data = response.json()
        if not data:
            print("No data found in response.")
            return pd.DataFrame()
        return pd.DataFrame(data)
    else:
        print(f"Error fetching data: {response.status_code}")
        return pd.DataFrame()

def fetch_full_intraday_data(symbol, start_date, end_date, api_key):
    """Fetch complete 1-minute data by merging batched requests"""
    date_ranges = generate_date_ranges(start_date, end_date)
    all_data = pd.DataFrame()

    for start, end in date_ranges:
        print(f"Fetching data for range: {start} to {end}")
        df = fetch_intraday_data(
            symbol, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), api_key
        )
        if not df.empty:
            all_data = pd.concat([all_data, df], ignore_index=True)

    if all_data.empty:
        print("No data returned for the specified date range.")
    else:
        all_data["date"] = pd.to_datetime(all_data["date"])
        all_data.sort_values(by="date", inplace=True)
        print("Fetched complete data successfully!")

    return all_data

class StockRecoveryAnalyzer:
    def __init__(self):
        self.data_source = 'Financial Modeling Prep API'
        
    def fetch_historical_data(self, symbol, period="5y", interval="1day"):
        """Fetch historical OHLCV data for the specified period and interval"""
        try:
            if not symbol.endswith('.NS') and symbol in ['ADANIENT', 'SHRIRAMFIN']:
                symbol = symbol + '.NS'
            
            print(f"Fetching {period} of {interval} data for {symbol}...")
            
            end_date = datetime.now()
            if period == "5y":
                start_date = end_date - timedelta(days=5*365)  # 5 years
            elif period == "60d":
                start_date = end_date - timedelta(days=60)
            elif period == "30d":
                start_date = end_date - timedelta(days=30)
            elif period == "7d":
                start_date = end_date - timedelta(days=7)
            else:
                start_date = end_date - timedelta(days=5*365)  # Default to 5 years
            
            if interval == "1day":
                # For daily data, use historical-price endpoint
                url = f"{BASE_URL}/historical-price-full/{symbol}?from={start_date.strftime('%Y-%m-%d')}&to={end_date.strftime('%Y-%m-%d')}&apikey={API_KEY}"
                print(f"Fetching daily data from: {url}")
                response = requests.get(url, verify=False)
                if response.status_code == 200:
                    response_data = response.json()
                    if 'historical' in response_data and response_data['historical']:
                        data = pd.DataFrame(response_data['historical'])
                        # Rename columns to match expected format
                        data = data.rename(columns={'date': 'Datetime'})
                        data['Datetime'] = pd.to_datetime(data['Datetime'])
                        # Capitalize column names
                        column_mapping = {
                            'open': 'Open',
                            'high': 'High', 
                            'low': 'Low',
                            'close': 'Close',
                            'volume': 'Volume'
                        }
                        data = data.rename(columns=column_mapping)
                    else:
                        print("No historical data found in response.")
                        data = pd.DataFrame()
                else:
                    print(f"Error fetching data: {response.status_code}")
                    data = pd.DataFrame()
            else:
                # For intraday data, use existing method
                data = fetch_full_intraday_data(symbol, start_date, end_date, API_KEY)
            
            if data.empty:
                print(f"No data returned for {symbol}")
                return None
            
            column_mapping = {
                'date': 'Datetime',
                'open': 'Open', 
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            }
            
            data = data.rename(columns=column_mapping)
            
            if 'Datetime' in data.columns:
                data['Datetime'] = pd.to_datetime(data['Datetime'])
            else:
                print("No datetime column found in API response")
                return None
            
            data = data.sort_values('Datetime').reset_index(drop=True)
            
            required_fields = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_fields = [field for field in required_fields if field not in data.columns]
            
            if missing_fields:
                print(f"Warning: Missing required fields: {missing_fields}")
                return None
            
            print(f"Successfully fetched {len(data)} records for {symbol}")
            return data
            
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return None

    
    def calculate_peak_price(self, data, lookback_period, price_type, current_idx):
        """Calculate peak price within lookback period"""
        start_idx = max(0, current_idx - lookback_period)
        if start_idx >= current_idx:
            return data.iloc[current_idx][['Open', 'High', 'Low', 'Close']].max()
            
        window_data = data.iloc[start_idx:current_idx]
        
        if price_type == "Close":
            return window_data['Close'].max()
        elif price_type == "High/Low":
            return window_data['High'].max()
        else:
            raise ValueError("Price_Type_For_Peak_Trough_Drop must be 'Close' or 'High/Low'")
    
    def detect_drop_events(self, data, lookback_period, drop_threshold, recovery_target, 
                          price_type, min_drop_duration):
        """Detect and validate drop events according to algorithm requirements"""
        print(f"  Analyzing drop threshold: {drop_threshold}%")
        
        drop_events = []
        current_event = None
        
        for i in range(lookback_period, len(data)):
            peak_price = self.calculate_peak_price(data, lookback_period, price_type, i)
            
            if price_type == "Close":
                current_price = data.iloc[i]['Close']
            else:
                current_price = data.iloc[i]['Low']
            
            drop_threshold_price = peak_price * (1 - drop_threshold / 100)
            
            if current_price <= drop_threshold_price:
                if current_event is None:
                    current_event = {
                        'peak_price': peak_price,
                        'drop_start_idx': i,
                        'trough_price': current_price,
                        'trough_idx': i,
                        'duration': 1
                    }
                else:
                    current_event['duration'] += 1
                    if current_price < current_event['trough_price']:
                        current_event['trough_price'] = current_price
                        current_event['trough_idx'] = i
            else:
                if current_event is not None:
                    if current_event['duration'] >= min_drop_duration:
                        total_price_drop = current_event['peak_price'] - current_event['trough_price']
                        current_event['total_price_drop'] = total_price_drop
                        current_event['recovery_target_price'] = current_event['trough_price'] + (total_price_drop * (recovery_target / 100))
                        drop_events.append(current_event.copy())
                    
                    current_event = None
        
        if current_event is not None and current_event['duration'] >= min_drop_duration:
            total_price_drop = current_event['peak_price'] - current_event['trough_price']
            current_event['total_price_drop'] = total_price_drop
            current_event['recovery_target_price'] = current_event['trough_price'] + (total_price_drop * (recovery_target / 100))
            drop_events.append(current_event.copy())
        
        print(f"    Found {len(drop_events)} valid drop events")
        return drop_events
    
    def check_recovery(self, data, drop_event, price_type, max_lookahead=252):
        """Check if price recovers to target within lookahead period (252 trading days = ~1 year)"""
        trough_idx = drop_event['trough_idx']
        target_recovery_price = drop_event['recovery_target_price']
        lookahead_limit = min(trough_idx + max_lookahead, len(data))
        
        for j in range(trough_idx + 1, lookahead_limit):
            if price_type == "Close":
                check_price = data.iloc[j]['Close']
            else:
                check_price = data.iloc[j]['High']
            
            if check_price >= target_recovery_price:
                recovery_time_days = j - trough_idx
                return True, recovery_time_days
        
        return False, None
    
    def analyze_stock_recovery(self, symbol, lookback_period, drop_thresholds, 
                             recovery_target, price_type, min_drop_duration):
        """Complete recovery analysis for a single stock"""
        print(f"\nAnalyzing: {symbol}")
        print("="*60)
        
        data = self.fetch_historical_data(symbol, period="5y", interval="1day")
        
        if data is None or len(data) == 0:
            print(f"No data available for {symbol}")
            return []
        
        print(f"Data validation - Records: {len(data)}, Range: {data['Datetime'].min()} to {data['Datetime'].max()}")
        
        results = []
        
        for drop_threshold in drop_thresholds:
            drop_events = self.detect_drop_events(
                data, lookback_period, drop_threshold, recovery_target,
                price_type, min_drop_duration
            )
            
            total_drop_events = len(drop_events)
            successful_recoveries = 0
            recovery_times = []
            
            for event_idx, event in enumerate(drop_events):
                recovered, recovery_time = self.check_recovery(data, event, price_type)
                if recovered:
                    successful_recoveries += 1
                    recovery_times.append(recovery_time)
            
            recovery_probability = (successful_recoveries / total_drop_events * 100) if total_drop_events > 0 else 0
            avg_recovery_time = np.mean(recovery_times) if recovery_times else None
            
            results.append({
                'Stock Symbol': symbol,
                'Drop Threshold (%)': drop_threshold,
                'Total Drop Events Observed': total_drop_events,
                'Successful Recovery Events': successful_recoveries,
                'Recovery Probability (%)': round(recovery_probability, 2),
                'Average Recovery Time (Days)': round(avg_recovery_time, 1) if avg_recovery_time else None,
                'Data Period': '5 Years (Daily)',
                'Analysis Date': datetime.now().strftime('%Y-%m-%d')
            })
        
        return results

def validate_configuration(config):
    """Validate configuration parameters"""
    if not isinstance(config['LOOKBACK_PERIOD_FOR_PEAK'], int) or config['LOOKBACK_PERIOD_FOR_PEAK'] <= 0:
        raise ValueError("Lookback_Period_For_Peak must be a positive integer")
    
    if not isinstance(config['DROP_THRESHOLDS_PERCENTAGE'], list):
        raise ValueError("Drop_Thresholds_Percentage must be a list of doubles")
    for threshold in config['DROP_THRESHOLDS_PERCENTAGE']:
        if not isinstance(threshold, (int, float)) or threshold <= 0 or threshold >= 100:
            raise ValueError("Each drop threshold must be between 0 and 100 percent")
    
    if not isinstance(config['RECOVERY_TARGET_PERCENTAGE'], (int, float)) or config['RECOVERY_TARGET_PERCENTAGE'] <= 0:
        raise ValueError("Recovery_Target_Percentage must be a positive number")
    
    valid_price_types = ["Close", "High/Low"]
    if config['PRICE_TYPE_FOR_PEAK_TROUGH_DROP'] not in valid_price_types:
        raise ValueError(f"Price_Type_For_Peak_Trough_Drop must be one of: {valid_price_types}")
    
    if not isinstance(config['MINIMUM_DROP_EVENT_DURATION'], int) or config['MINIMUM_DROP_EVENT_DURATION'] <= 0:
        raise ValueError("Minimum_Drop_Event_Duration must be a positive integer")
    
    return True

def main():
    """Main execution function"""
    CONFIG = {
        'LOOKBACK_PERIOD_FOR_PEAK': 252,  # 252 trading days = 1 year lookback
        'DROP_THRESHOLDS_PERCENTAGE': [10.0, 15.0, 20.0],
        'RECOVERY_TARGET_PERCENTAGE': 40.0,
        'PRICE_TYPE_FOR_PEAK_TROUGH_DROP': "High/Low",
        'MINIMUM_DROP_EVENT_DURATION': 2,  # 2 days minimum duration for daily data
        'STOCKS_TO_ANALYZE': ['ADANIENT.NS', 'SHRIRAMFIN.NS']
    }
    
    display_configuration(CONFIG)
    validate_configuration(CONFIG)
    
    analyzer = StockRecoveryAnalyzer()
    all_results = []
    
    for stock in CONFIG['STOCKS_TO_ANALYZE']:
        try:
            results = analyzer.analyze_stock_recovery(
                symbol=stock,
                lookback_period=CONFIG['LOOKBACK_PERIOD_FOR_PEAK'],
                drop_thresholds=CONFIG['DROP_THRESHOLDS_PERCENTAGE'],
                recovery_target=CONFIG['RECOVERY_TARGET_PERCENTAGE'],
                price_type=CONFIG['PRICE_TYPE_FOR_PEAK_TROUGH_DROP'],
                min_drop_duration=CONFIG['MINIMUM_DROP_EVENT_DURATION']
            )
            all_results.extend(results)
        except Exception as e:
            print(f"Error analyzing {stock}: {e}")
            continue
    
    if all_results:
        results_df = pd.DataFrame(all_results)
        
        print("\n" + "="*80)
        print("STOCK RECOVERY PROBABILITY ANALYSIS RESULTS")
        print("="*80)
        print(results_df.to_string(index=False))
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"stock_recovery_analysis_{timestamp}.csv"
        results_df.to_csv(filename, index=False)
        print(f"\nResults saved to: {filename}")
        
        print("\nSUMMARY:")
        print("="*40)
        for threshold in CONFIG['DROP_THRESHOLDS_PERCENTAGE']:
            threshold_data = results_df[results_df['Drop Threshold (%)'] == threshold]
            total_events = threshold_data['Total Drop Events Observed'].sum()
            successful_events = threshold_data['Successful Recovery Events'].sum()
            avg_prob = threshold_data['Recovery Probability (%)'].mean()
            
            print(f"{threshold}% drops: {successful_events}/{total_events} recovered (avg {avg_prob:.1f}%)")
        
        return results_df
    else:
        print("No results generated!")
        return None

def display_configuration(config):
    """Display current configuration parameters"""
    print("CONFIGURATION:")
    print("-" * 40)
    print(f"Data Period: 5 Years (Daily Data)")
    print(f"Lookback Period: {config['LOOKBACK_PERIOD_FOR_PEAK']} trading days")
    print(f"Drop Thresholds: {config['DROP_THRESHOLDS_PERCENTAGE']}%")
    print(f"Recovery Target: {config['RECOVERY_TARGET_PERCENTAGE']}%")
    print(f"Price Type: {config['PRICE_TYPE_FOR_PEAK_TROUGH_DROP']}")
    print(f"Min Duration: {config['MINIMUM_DROP_EVENT_DURATION']} days")
    print(f"Stocks: {config['STOCKS_TO_ANALYZE']}")
    print("-" * 40)

def create_custom_config(lookback_period=1440, drop_thresholds=[10.0, 15.0, 20.0], 
                        recovery_target=40.0, price_type="High/Low", min_duration=5, 
                        stocks=['ADANIENT', 'SHRIRAMFIN']):
    """Create custom configuration with different parameters"""
    return {
        'LOOKBACK_PERIOD_FOR_PEAK': lookback_period,
        'DROP_THRESHOLDS_PERCENTAGE': drop_thresholds,
        'RECOVERY_TARGET_PERCENTAGE': recovery_target,
        'PRICE_TYPE_FOR_PEAK_TROUGH_DROP': price_type,
        'MINIMUM_DROP_EVENT_DURATION': min_duration,
        'STOCKS_TO_ANALYZE': stocks,
        'USE_SAMPLE_DATA': False
    }

if __name__ == "__main__":
    print("STOCK RECOVERY ANALYZER - 5 YEAR ANALYSIS")
    print("="*50)
    print("Analyzing recovery probability following price drops")
    print("Using 5 years of daily data from Financial Modeling Prep API")
    print("="*50)
    
    results = main()
    
    if results is not None and not results.empty:
        print(f"\nAnalysis completed! Total events: {results['Total Drop Events Observed'].sum()}")
    else:
        print("\nNo results generated")