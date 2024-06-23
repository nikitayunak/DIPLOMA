from modelProcessor import *
from dataProcessor import *


class Forecaster:
    def __init__(self, df):
        dp = dataProcessor.DataProcessor(df)
        df = dp.getPreparedData()
        self.mp = ModelProcessor(df)
        self.model = self.mp.getModel()

    def makePrediction(self, steps):
        predictions = self.model.predict(steps=steps)
        return predictions
