import yfinance

class FetchData:
    def __init__(self, ticker, start_date, end_date):
        """
        Initializes the FetchData class with ticker symbol and date range.
        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL' for Apple Inc.)
            start_date: Start date for fetching data (format: 'YYYY-MM-DD')
            end_date: End date for fetching data (format: 'YYYY-MM-DD')
        """
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date

    def get_data(self):
        """
        Fetches historical stock data for the specified ticker symbol
        Returns:
            data: DataFrame containing stock data with date as index
        """
        data = yfinance.download(self.ticker, start=self.start_date, end=self.end_date, auto_adjust=True)
        if data.empty:
            raise ValueError(f"No data found for ticker {self.ticker} between {self.start_date} and {self.end_date}.")
        
        return data

