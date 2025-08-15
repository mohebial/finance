import pandas as pd
import yfinance as yf
from datetime import datetime
import requests

# ---------- CONFIG ----------
NOTION_TOKEN = "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"
DATABASE_ID = "24d7c4ea218780e38f35d1282a850ed5"

headers = {
    "Authorization": f"Bearer {NOTION_TOKEN}",
    "Content-Type": "application/json",
    "Notion-Version": "2022-06-28"
}

# ---------- GET LAST QUARTER ----------
def get_last_quarter():
    """Get the start and end dates of the last complete quarter"""
    today = datetime.today()
    current_quarter = (today.month - 1) // 3 + 1
    
    # If we're in the first month of a quarter, use the previous quarter
    # Otherwise use the current quarter (assuming it's complete)
    if today.day < 15 and today.month in [1, 4, 7, 10]:  # First month of quarter
        quarter_to_use = current_quarter - 1 if current_quarter > 1 else 4
        year_to_use = today.year if current_quarter > 1 else today.year - 1
    else:
        # Use previous quarter since current might not be complete
        quarter_to_use = current_quarter - 1 if current_quarter > 1 else 4
        year_to_use = today.year if current_quarter > 1 else today.year - 1
    
    # Calculate quarter dates
    start_month = (quarter_to_use - 1) * 3 + 1
    start_date = datetime(year_to_use, start_month, 1)
    
    # End date is last day of the quarter
    end_month = start_month + 2
    if end_month == 12:
        end_date = datetime(year_to_use, 12, 31)
    else:
        next_month_start = datetime(year_to_use, end_month + 1, 1)
        end_date = datetime(year_to_use, end_month, (next_month_start - datetime(year_to_use, end_month, 1)).days)
    
    quarter_name = f"Q{quarter_to_use} {year_to_use}"
    
    return start_date, end_date, quarter_name

# ---------- GET S&P 500 ----------
def get_sp500_tickers():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    table = pd.read_html(url, header=0)[0]
    return table['Symbol'].tolist()

# ---------- ANALYZE ----------
def analyze_tickers(tickers, start_date, end_date, quarter_name):
    results = []
    for sym in tickers:
        try:
            stock = yf.Ticker(sym)
            hist = stock.history(start=start_date, end=end_date)
            if hist.empty:
                continue

            start_price = hist['Close'].iloc[0]
            end_price = hist['Close'].iloc[-1]
            growth = (end_price - start_price) / start_price * 100
            
            # Calculate quarterly return instead of annualized metrics
            quarterly_return = growth

            fin = stock.financials
            if 'Total Revenue' in fin.index and 'Net Income' in fin.index:
                revenue = fin.loc['Total Revenue'].iloc[0]
                net_income = fin.loc['Net Income'].iloc[0]
                profit_margin = (net_income / revenue) * 100
            else:
                profit_margin = None

            index_val = (quarterly_return + profit_margin) if profit_margin is not None else None

            results.append({
                'Quarter': quarter_name,
                'Ticker': sym,
                'Quarterly Return (%)': round(quarterly_return, 2),
                'Net Profit Margin (%)': round(profit_margin, 2) if profit_margin else None,
                'Custom Index': round(index_val, 2) if index_val else None
            })
        except Exception as e:
            print(f"Error processing {sym}: {e}")
    return pd.DataFrame(results)

# ---------- PUSH TO NOTION ----------
def send_to_notion(df):
    for _, row in df.iterrows():
        data = {
            "parent": {"database_id": DATABASE_ID},
            "properties": {
                "Quarter": {"rich_text": [{"text": {"content": row['Quarter']}}]},
                "Ticker": {"title": [{"text": {"content": row['Ticker']}}]},
                "Quarterly Return (%)": {"number": row['Quarterly Return (%)']},
                "Net Profit Margin (%)": {"number": row['Net Profit Margin (%)']},
                "Custom Index": {"number": row['Custom Index']}
            }
        }
        response = requests.post("https://api.notion.com/v1/pages", headers=headers, json=data)
        if response.status_code != 200:
            print(f"Error adding {row['Ticker']} for {row['Quarter']}: {response.text}")

# ---------- MAIN ----------
if __name__ == "__main__":
    print("📊 Starting last quarter S&P 500 analysis...")
    
    # Get last quarter dates
    start_date, end_date, quarter_name = get_last_quarter()
    print(f"Analyzing {quarter_name}: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    # Get S&P 500 tickers
    tickers = get_sp500_tickers()
    print(f"Found {len(tickers)} S&P 500 companies")
    
    # Analyze the last quarter
    df = analyze_tickers(
        tickers,
        start_date.strftime("%Y-%m-%d"),
        end_date.strftime("%Y-%m-%d"),
        quarter_name
    )
    
    # Send to Notion and display results
    if not df.empty:
        print(f"Sending {len(df)} records to Notion...")
        send_to_notion(df)
        print("✅ Data sent to Notion database.")
        
        # Display summary
        print(f"\n📈 Summary for {quarter_name}:")
        print(f"Companies analyzed: {len(df)}")
        print(f"Average quarterly return: {df['Quarterly Return (%)'].mean():.2f}%")
        print(f"Average custom index: {df['Custom Index'].mean():.2f}")
        print(f"Best performer: {df.loc[df['Quarterly Return (%)'].idxmax(), 'Ticker']} ({df['Quarterly Return (%)'].max():.2f}%)")
        print(f"Worst performer: {df.loc[df['Quarterly Return (%)'].idxmin(), 'Ticker']} ({df['Quarterly Return (%)'].min():.2f}%)")
    else:
        print("❌ No data to send to Notion.")
