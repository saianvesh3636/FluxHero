"""
Script to download historical data for multiple symbols (AAPL, MSFT) from Yahoo Finance.

This script downloads 1 year of daily OHLCV data for AAPL and MSFT
and saves them to backend/test_data/ directory for development and testing.

Usage:
    python scripts/download_multi_symbol_data.py
"""

from datetime import datetime, timedelta
from pathlib import Path

import yfinance as yf


def download_symbol_data(symbol: str, output_dir: Path):
    """
    Download 1 year of daily data for a symbol.

    Args:
        symbol: Stock symbol (e.g., 'AAPL', 'MSFT')
        output_dir: Directory to save CSV file
    """
    print(f"Downloading {symbol} data...")

    # Calculate date range (1 year back from today)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)

    # Download data from Yahoo Finance
    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start_date, end=end_date, interval='1d')

    if df.empty:
        print(f"ERROR: No data retrieved for {symbol}")
        return

    # Rename columns to match SPY format
    df = df.rename(columns={
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume'
    })

    # Keep only OHLCV columns
    df = df[['open', 'high', 'low', 'close', 'volume']]

    # Reset index to make Date a column
    df.reset_index(inplace=True)
    df = df.rename(columns={'Date': 'Price'})

    # Add Ticker column header (matching SPY format)
    df.insert(0, 'Ticker', symbol)

    # Save to CSV
    output_file = output_dir / f"{symbol.lower()}_daily.csv"
    df.to_csv(output_file, index=False)

    print(f"✓ Saved {len(df)} rows to {output_file}")

def main():
    """Download data for AAPL and MSFT"""
    # Get output directory
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    output_dir = project_dir / "backend" / "test_data"

    # Create directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir}")
    print("=" * 60)

    # Download data for each symbol
    symbols = ['AAPL', 'MSFT']
    for symbol in symbols:
        try:
            download_symbol_data(symbol, output_dir)
        except Exception as e:
            print(f"ERROR downloading {symbol}: {e}")

    print("=" * 60)
    print("Download complete!")

if __name__ == "__main__":
    main()
