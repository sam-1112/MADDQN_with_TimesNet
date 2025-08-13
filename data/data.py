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
    
    def get_multi_data(self, ticker_list):
        """
        Fetches multiple stock datas for the specified ticker list.
        Args:
            ticker_list: List of stock ticker symbols (e.g., ['AAPL', 'MSFT'])
        Returns:
            data: DataFrame containing stock data for all tickers with date as index
        """
        data = {}
        for ticker in ticker_list:
            ticker_data = yfinance.download(ticker, start=self.start_date, end=self.end_date, auto_adjust=True)
            if ticker_data.empty:
                raise ValueError(f"No data found for ticker {ticker} between {self.start_date} and {self.end_date}.")
            data[ticker] = ticker_data
        return data
