from sklearn.metrics import r2_score
import skforecast
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from skforecast.ForecasterAutoreg import ForecasterAutoreg
import dataProcessor
import dataVisualizer


class ModelProcessor:
    def __init__(self, df):
        self.df = df
        self.dp = dataProcessor.DataProcessor(df)

        self.forecaster = None

    def createModel(self):
        self.forecaster = ForecasterAutoreg(
                 regressor = Ridge(alpha=0.1, random_state=765),
                 lags      = 15
             )

    def trainModel(self, data):
        self.forecaster.fit(y=data['BEV Sales'])

    def getTrainedModel(self):
        self.df = self.dp.getPreparedData()
        self.data_train, self.data_test = self.dp.splitTrainTest(self.df, 12)
        self.createModel()
        self.trainModel(self.data_train)
        dv = dataVisualizer.DataVisualizer(self.df)
        predictions = self.forecaster.predict_interval(
            steps=12,
            interval=[1, 99],
            n_boot=500
        )
        r2 = r2_score(y_true=self.data_test['BEV Sales'], y_pred=predictions['pred'])
        print(f"R^2 score: {r2}")
        dv.showTrainingResults(self.data_test, predictions)
        return self.forecaster

    def getModel(self):
        self.createModel()
        self.trainModel(self.df)
        return self.forecaster
