import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class StockValuationModel:
    def __init__(self):
        self.data = pd.DataFrame()
        self.scores = pd.DataFrame()
        
    def fetch_stock_data(self, symbols):
        """
        Fetch fundamental data for given stock symbols
        """
        stock_data = []
        
        for symbol in symbols:
            try:
                print(f"Fetching data for {symbol}...")
                ticker = yf.Ticker(symbol)
                info = ticker.info
                
                # Get financial ratios and metrics
                stock_metrics = {
                    'symbol': symbol,
                    'name': info.get('longName', 'N/A'),
                    'sector': info.get('sector', 'N/A'),
                    'industry': info.get('industry', 'N/A'),
                    'market_cap': info.get('marketCap', 0),
                    'price': info.get('currentPrice', 0),
                    'pe_ratio': info.get('trailingPE', np.nan),
                    'forward_pe': info.get('forwardPE', np.nan),
                    'pb_ratio': info.get('priceToBook', np.nan),
                    'ps_ratio': info.get('priceToSalesTrailing12Months', np.nan),
                    'peg_ratio': info.get('pegRatio', np.nan),
                    'roe': info.get('returnOnEquity', np.nan),
                    'roa': info.get('returnOnAssets', np.nan),
                    'debt_to_equity': info.get('debtToEquity', np.nan),
                    'current_ratio': info.get('currentRatio', np.nan),
                    'quick_ratio': info.get('quickRatio', np.nan),
                    'gross_margins': info.get('grossMargins', np.nan),
                    'operating_margins': info.get('operatingMargins', np.nan),
                    'profit_margins': info.get('profitMargins', np.nan),
                    'revenue_growth': info.get('revenueGrowth', np.nan),
                    'earnings_growth': info.get('earningsGrowth', np.nan),
                    'beta': info.get('beta', np.nan),
                    'dividend_yield': info.get('dividendYield', 0),
                    'payout_ratio': info.get('payoutRatio', np.nan),
                    'book_value': info.get('bookValue', np.nan),
                    'price_to_book': info.get('priceToBook', np.nan),
                    'enterprise_value': info.get('enterpriseValue', 0),
                    'ev_revenue': info.get('enterpriseToRevenue', np.nan),
                    'ev_ebitda': info.get('enterpriseToEbitda', np.nan)
                }
                
                stock_data.append(stock_metrics)
                
            except Exception as e:
                print(f"Error fetching data for {symbol}: {e}")
                continue
        
        self.data = pd.DataFrame(stock_data)
        return self.data
    
    def calculate_valuation_score(self, weights=None):
        """
        Calculate composite valuation score based on multiple metrics
        Lower scores indicate better value
        """
        if weights is None:
            weights = {
                'pe_score': 0.2,
                'pb_score': 0.15,
                'ps_score': 0.1,
                'peg_score': 0.15,
                'roe_score': 0.1,
                'debt_score': 0.1,
                'growth_score': 0.1,
                'margin_score': 0.1
            }
        
        df = self.data.copy()
        print(f"Starting with {len(df)} stocks")
        
        # Debug: Check data quality
        print("\nData quality check:")
        for col in ['pe_ratio', 'pb_ratio', 'roe', 'profit_margins', 'revenue_growth']:
            if col in df.columns:
                valid_count = df[col].notna().sum()
                print(f"{col}: {valid_count}/{len(df)} valid values")
        
        # More lenient cleaning - only remove stocks with ALL critical metrics missing
        critical_metrics = ['pe_ratio', 'pb_ratio']  # Reduced to just 2 most important
        initial_count = len(df)
        df = df.dropna(subset=critical_metrics, how='all')  # Only drop if ALL are missing
        print(f"After cleaning: {len(df)} stocks (removed {initial_count - len(df)})")
        
        if len(df) == 0:
            print("Warning: No stocks have sufficient data for scoring")
            return pd.DataFrame()
        
        # Fill remaining NaN values with more reasonable defaults
        # For ratios, use median of available data, for growth use 0
        numeric_columns = ['pe_ratio', 'pb_ratio', 'ps_ratio', 'peg_ratio', 'roe', 
                          'debt_to_equity', 'profit_margins']
        
        for col in numeric_columns:
            if col in df.columns:
                if df[col].notna().sum() > 0:  # If we have any valid values
                    median_val = df[col].median()
                    df[col] = df[col].fillna(median_val)
                else:
                    # If no valid values, use reasonable defaults
                    defaults = {
                        'pe_ratio': 20, 'pb_ratio': 2.5, 'ps_ratio': 3, 'peg_ratio': 1.5,
                        'roe': 0.12, 'debt_to_equity': 0.5, 'profit_margins': 0.08
                    }
                    df[col] = df[col].fillna(defaults.get(col, 0))
        
        # Revenue growth defaults to 0 if missing
        df['revenue_growth'] = df['revenue_growth'].fillna(0)
        
        scores_df = pd.DataFrame()
        scores_df['symbol'] = df['symbol']
        scores_df['name'] = df['name']
        scores_df['sector'] = df['sector']
        scores_df['price'] = df['price']
        scores_df['market_cap'] = df['market_cap']
        
        print(f"\nCalculating scores for {len(df)} stocks...")
        
        # P/E Score (lower is better)
        pe_percentile = df['pe_ratio'].rank(pct=True)
        scores_df['pe_score'] = pe_percentile * 100
        
        # P/B Score (lower is better)
        pb_percentile = df['pb_ratio'].rank(pct=True)
        scores_df['pb_score'] = pb_percentile * 100
        
        # P/S Score (lower is better)
        ps_percentile = df['ps_ratio'].rank(pct=True)
        scores_df['ps_score'] = ps_percentile * 100
        
        # PEG Score (lower is better)
        peg_percentile = df['peg_ratio'].rank(pct=True)
        scores_df['peg_score'] = peg_percentile * 100
        
        # ROE Score (higher is better, so invert)
        roe_percentile = df['roe'].rank(pct=True, ascending=False)
        scores_df['roe_score'] = roe_percentile * 100
        
        # Debt Score (lower debt-to-equity is better)
        debt_percentile = df['debt_to_equity'].rank(pct=True)
        scores_df['debt_score'] = debt_percentile * 100
        
        # Growth Score (higher revenue growth is better)
        growth_percentile = df['revenue_growth'].rank(pct=True, ascending=False)
        scores_df['growth_score'] = growth_percentile * 100
        
        # Margin Score (higher profit margins are better)
        margin_percentile = df['profit_margins'].rank(pct=True, ascending=False)
        scores_df['margin_score'] = margin_percentile * 100
        
        # Calculate composite score
        scores_df['composite_score'] = (
            scores_df['pe_score'] * weights['pe_score'] +
            scores_df['pb_score'] * weights['pb_score'] +
            scores_df['ps_score'] * weights['ps_score'] +
            scores_df['peg_score'] * weights['peg_score'] +
            scores_df['roe_score'] * weights['roe_score'] +
            scores_df['debt_score'] * weights['debt_score'] +
            scores_df['growth_score'] * weights['growth_score'] +
            scores_df['margin_score'] * weights['margin_score']
        )
        
        # Add raw metrics for reference
        scores_df['pe_ratio'] = df['pe_ratio']
        scores_df['pb_ratio'] = df['pb_ratio']
        scores_df['peg_ratio'] = df['peg_ratio']
        scores_df['roe'] = df['roe']
        scores_df['revenue_growth'] = df['revenue_growth']
        scores_df['profit_margins'] = df['profit_margins']
        scores_df['debt_to_equity'] = df['debt_to_equity']
        
        self.scores = scores_df.sort_values('composite_score').reset_index(drop=True)
        print(f"Scoring complete! Top stock: {self.scores.iloc[0]['symbol']} with score {self.scores.iloc[0]['composite_score']:.1f}")
        
        return self.scores
    
    def screen_stocks(self, criteria):
        """
        Screen stocks based on specific criteria
        """
        df = self.data.copy()
        
        # Apply filters
        for metric, (operator, value) in criteria.items():
            if metric in df.columns:
                if operator == '<=':
                    df = df[df[metric] <= value]
                elif operator == '>=':
                    df = df[df[metric] >= value]
                elif operator == '<':
                    df = df[df[metric] < value]
                elif operator == '>':
                    df = df[df[metric] > value]
        
        return df[['symbol', 'name', 'sector', 'pe_ratio', 'pb_ratio', 'peg_ratio', 
                  'roe', 'revenue_growth', 'profit_margins', 'debt_to_equity']]
    
    def plot_valuation_analysis(self):
        """
        Create visualizations for valuation analysis
        """
        if self.scores.empty:
            print("No scores calculated. Run calculate_valuation_score() first.")
            return
        
        print(f"Starting plot with {len(self.scores)} stocks")
        
        # Much more lenient data cleaning for plotting
        plot_data = self.scores.copy()
        
        # Only remove truly problematic values, keep most data
        plot_data = plot_data[plot_data['composite_score'].notna()]
        print(f"After removing NaN composite scores: {len(plot_data)} stocks")
        
        if len(plot_data) == 0:
            print("No data available for plotting after cleaning.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Top 10 undervalued stocks - this should always work
        top_10 = plot_data.head(10)
        axes[0, 0].barh(top_10['symbol'], top_10['composite_score'])
        axes[0, 0].set_title('Top 10 Undervalued Stocks (Lower Score = Better Value)')
        axes[0, 0].set_xlabel('Composite Valuation Score')
        
        # PE vs PB scatter - be more lenient with outliers
        pe_data = plot_data[plot_data['pe_ratio'].notna() & plot_data['pb_ratio'].notna()]
        pe_clean = pe_data[(pe_data['pe_ratio'] > 0) & (pe_data['pe_ratio'] < 200)]  # More lenient
        pb_clean = pe_clean[(pe_clean['pb_ratio'] > 0) & (pe_clean['pb_ratio'] < 50)]  # More lenient
        
        print(f"P/E vs P/B plot data: {len(pb_clean)} stocks")
        
        if len(pb_clean) > 1:
            scatter1 = axes[0, 1].scatter(pb_clean['pe_ratio'], pb_clean['pb_ratio'], 
                              c=pb_clean['composite_score'], cmap='RdYlGn_r', alpha=0.7)
            axes[0, 1].set_xlabel('P/E Ratio')
            axes[0, 1].set_ylabel('P/B Ratio')
            axes[0, 1].set_title('P/E vs P/B Ratio (Color = Valuation Score)')
            try:
                plt.colorbar(scatter1, ax=axes[0, 1])
            except:
                pass  # Skip colorbar if it fails
        else:
            axes[0, 1].text(0.5, 0.5, f'Insufficient clean data for P/E vs P/B plot\n({len(pb_clean)} stocks)', 
                           ha='center', va='center', transform=axes[0, 1].transAxes)
        
        # ROE vs Revenue Growth - be more lenient
        roe_data = plot_data[plot_data['roe'].notna() & plot_data['revenue_growth'].notna()]
        roe_clean = roe_data[(roe_data['roe'] >= -2) & (roe_data['roe'] <= 5)]  # Very lenient
        growth_clean = roe_clean[(roe_clean['revenue_growth'] >= -1) & (roe_clean['revenue_growth'] <= 2)]
        
        print(f"ROE vs Growth plot data: {len(growth_clean)} stocks")
        
        if len(growth_clean) > 1:
            scatter2 = axes[1, 0].scatter(growth_clean['roe'], growth_clean['revenue_growth'], 
                              c=growth_clean['composite_score'], cmap='RdYlGn_r', alpha=0.7)
            axes[1, 0].set_xlabel('Return on Equity')
            axes[1, 0].set_ylabel('Revenue Growth')
            axes[1, 0].set_title('ROE vs Revenue Growth')
            try:
                plt.colorbar(scatter2, ax=axes[1, 0])
            except:
                pass  # Skip colorbar if it fails
        else:
            axes[1, 0].text(0.5, 0.5, f'Insufficient clean data for ROE vs Growth plot\n({len(growth_clean)} stocks)', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
        
        # Score distribution by sector
        sector_data = plot_data[plot_data['sector'].notna()]
        sector_scores = sector_data.groupby('sector')['composite_score'].mean().sort_values()
        
        print(f"Sector analysis: {len(sector_scores)} sectors")
        
        if len(sector_scores) > 0:
            axes[1, 1].bar(range(len(sector_scores)), sector_scores.values)
            axes[1, 1].set_xticks(range(len(sector_scores)))
            axes[1, 1].set_xticklabels(sector_scores.index, rotation=45, ha='right')
            axes[1, 1].set_title('Average Valuation Score by Sector')
            axes[1, 1].set_ylabel('Average Score')
        else:
            axes[1, 1].text(0.5, 0.5, 'No sector data available', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
        
        plt.tight_layout()
        plt.show()
        print("Visualization complete!")
    
    def generate_report(self, top_n=10):
        """
        Generate a detailed report of top undervalued stocks
        """
        if self.scores.empty:
            print("No scores calculated. Run calculate_valuation_score() first.")
            return
        
        print("=" * 80)
        print("STOCK VALUATION ANALYSIS REPORT")
        print("=" * 80)
        print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Stocks Analyzed: {len(self.data)}")
        print(f"Stocks with Valid Scores: {len(self.scores)}")
        print("\n")
        
        top_stocks = self.scores.head(top_n)
        
        print(f"TOP {top_n} UNDERVALUED STOCKS:")
        print("-" * 40)
        
        for idx, stock in top_stocks.iterrows():
            print(f"\n{stock['symbol']} - {stock['name']}")
            print(f"Sector: {stock['sector']}")
            print(f"Current Price: ${stock['price']:.2f}")
            print(f"Market Cap: ${stock['market_cap']:,.0f}")
            print(f"Composite Score: {stock['composite_score']:.1f}")
            
            # Handle NaN values in display
            pe_ratio = stock['pe_ratio'] if pd.notna(stock['pe_ratio']) else 'N/A'
            pb_ratio = stock['pb_ratio'] if pd.notna(stock['pb_ratio']) else 'N/A'
            peg_ratio = stock['peg_ratio'] if pd.notna(stock['peg_ratio']) else 'N/A'
            roe = stock['roe'] if pd.notna(stock['roe']) else 'N/A'
            rev_growth = stock['revenue_growth'] if pd.notna(stock['revenue_growth']) else 'N/A'
            profit_margin = stock['profit_margins'] if pd.notna(stock['profit_margins']) else 'N/A'
            
            print(f"P/E Ratio: {pe_ratio if pe_ratio == 'N/A' else f'{pe_ratio:.1f}'}")
            print(f"P/B Ratio: {pb_ratio if pb_ratio == 'N/A' else f'{pb_ratio:.1f}'}")
            print(f"PEG Ratio: {peg_ratio if peg_ratio == 'N/A' else f'{peg_ratio:.1f}'}")
            print(f"ROE: {roe if roe == 'N/A' else f'{roe:.1%}'}")
            print(f"Revenue Growth: {rev_growth if rev_growth == 'N/A' else f'{rev_growth:.1%}'}")
            print(f"Profit Margin: {profit_margin if profit_margin == 'N/A' else f'{profit_margin:.1%}'}")
            print("-" * 40)

# Example usage and demo
if __name__ == "__main__":
    # Initialize the model
    model = StockValuationModel()
    
    # Example stock symbols (you can modify this list)
    symbols = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA',
        'JNJ', 'JPM', 'V', 'PG', 'UNH',
        'HD', 'MA', 'DIS', 'NVDA', 'PYPL',
        'BAC', 'XOM', 'WMT', 'KO', 'PFE'
    ]
    
    print("Fetching stock data...")
    data = model.fetch_stock_data(symbols)
    print(f"Successfully fetched data for {len(data)} stocks")
    
    print("\nCalculating valuation scores...")
    scores = model.calculate_valuation_score()
    
    print("\nGenerating analysis report...")
    model.generate_report()
    
    # Example screening criteria
    print("\n" + "=" * 80)
    print("CUSTOM STOCK SCREENING")
    print("=" * 80)
    
    # Define screening criteria (modify as needed)
    screening_criteria = {
        'pe_ratio': ('<=', 20),           # P/E ratio <= 20
        'pb_ratio': ('<=', 3),            # P/B ratio <= 3
        'peg_ratio': ('<=', 1.5),         # PEG ratio <= 1.5
        'roe': ('>=', 0.15),              # ROE >= 15%
        'debt_to_equity': ('<=', 1.0),    # Debt-to-equity <= 1.0
        'revenue_growth': ('>=', 0.05)    # Revenue growth >= 5%
    }
    
    screened_stocks = model.screen_stocks(screening_criteria)
    print("Stocks meeting screening criteria:")
    print(screened_stocks.to_string(index=False))
    
    # Create visualizations
    print("\nGenerating visualizations...")
    model.plot_valuation_analysis()