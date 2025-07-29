import numpy as np
class PreprocessData:
    def __init__(self, data, window_size=10, dates=None):
        """
        Initializes the PreprocessData class with the data and window size.
        Args:
            data: DataFrame containing stock data
            window_size: Size of the rolling window for preprocessing
        """
        self.data = data
        self.window_size = window_size
        self.dates = dates

    def subAgentData(self, current_step, batch_size):
        original_data = self.data.iloc[current_step:current_step + self.window_size]
        if original_data.empty:
            raise ValueError("No data available for the given step and window size.")
        # 確保數據是 2D 的
        prerprocessedData = np.zeros((batch_size, self.window_size, original_data.shape[1]))
        for step in range(self.window_size):
            if step < len(original_data):
                row = original_data.iloc[step].values
                if len(row.shape) == 1:
                    row = row.reshape(1, -1)
                prerprocessedData[:, step, :] = row
        return prerprocessedData

    def normalizeData(self):
        """
        Normalizes the data using Min-Max scaling for each column independently.
        Returns:
            normalized_data: Normalized data as a 2D numpy array
        """
        if self.data.empty:
            raise ValueError("Data is empty, cannot normalize.")

        normalized_data = self.data.copy()
        for col in normalized_data.columns:
            min_val = normalized_data[col].min()
            max_val = normalized_data[col].max()
            if max_val - min_val == 0:
                normalized_data[col] = 0.0  # 或 np.nan，視需求而定
            else:
                normalized_data[col] = (normalized_data[col] - min_val) / (max_val - min_val)

        if normalized_data.isnull().values.any():
            raise ValueError("Normalization resulted in NaN values.")

        return normalized_data.values
    
    # def splitData(self, train_ratio=0.7) -> list:
    #     """
    #     Splits the data into training, validation, and test sets.
    #     Args:
    #         train_ratio: Proportion of data to use for training

    #     :return: 
    #         list: Tuple containing training, and test sets as numpy arrays
    #     """
    #     if self.data.empty:
    #         raise ValueError("Data is empty, cannot split.")

    #     normData = self.normalizeData()
    #     total_samples = len(normData)
        
    #     train_size = int(total_samples * train_ratio)
    #     train_data = normData[:train_size]
    #     test_data = normData[train_size:]

    #     return [train_data, test_data]
    
    def splitData(self):
        """
        Splits the data into training and test sets based on provided date ranges.

        :return: List containing training and test data as numpy arrays
        """
        train_start_date, train_end_date, test_start_date, test_end_date = self.dates
        if self.data.empty:
            raise ValueError("Data is empty, cannot split.")
        normData = self.normalizeData()
        train_data = normData[(self.data.index >= train_start_date) & (self.data.index <= train_end_date)]
        test_data = normData[(self.data.index >= test_start_date) & (self.data.index <= test_end_date)]
        
        return [train_data, test_data]
    
    def timeSeriesData(self):
        """
        Preprocess the data into 3-dimensional arrays for model input.
        
        :return: Preprocessed data as a 3D numpy array and labels as a 2D numpy array
        """
        
        normDataList = self.splitData()
        processedDataList = []
        for normdata in normDataList:
            X = np.zeros((len(normdata) - self.window_size, self.window_size, normdata.shape[1]))
            # y = np.zeros((len(normdata) - self.window_size, normdata.shape[1]))
            for i in range(len(normdata) - self.window_size):
                X[i] = normdata[i:i + self.window_size]
                # y[i] = normdata[i + self.window_size]
            # processedDataList.append((X, y))
            processedDataList.append(X)
    
        return processedDataList