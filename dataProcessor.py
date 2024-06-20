import pandas as pd


class DataProcessor:
    def __init__(self, df):
        self.df = df

    def prepareData(self):
        self.df['Date'] = pd.to_datetime(self.df['Date'], format='%Y-%m')
        self.df = self.df.set_index('Date')
        self.df = self.df.asfreq('MS')

    def splitTrainTest(self, data, steps):
        # self.prepareData()
        data_train = data[:-steps]
        data_test = data[-steps:]
        return data_train, data_test

    def getPreparedData(self):
        self.prepareData()
        return self.df
