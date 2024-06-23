import matplotlib.pyplot as plt
import dataProcessor

plt.style.use('fivethirtyeight')
plt.rcParams['lines.linewidth'] = 1.5
plt.rcParams['font.size'] = 10


class DataVisualizer:
    def __init__(self, df):
        self.df = df

    def visualizeData(self):
        plt.plot(self.df)
        plt.show()

    def showTrainingResults(self, test_data, predictions):
        fig, ax = plt.subplots(figsize=(6, 2.5))
        test_data['BEV Sales'].plot(ax=ax, label='test')
        predictions['pred'].plot(ax=ax, label='prediction')
        ax.fill_between(
            predictions.index,
            predictions['lower_bound'],
            predictions['upper_bound'],
            color='red',
            alpha=0.2
        )
        ax.legend(loc='upper left')
        plt.show()

    def showForecast(self, predictions):
        dp = dataProcessor.DataProcessor(self.df)
        self.df = dp.getPreparedData()
        fig, ax = plt.subplots(dpi=80, facecolor='#000000')
        ax.set_facecolor('#000000')
        line1 = ax.plot(self.df.index, self.df["BEV Sales"], 'green', marker='o', linewidth=2, label='Actual Sales')
        line2 = ax.plot(predictions.index, predictions["pred"], 'yellow', marker='o', linewidth=2,
                        label='Forecasted Sales')
        ax.grid(alpha=0.2)
        ax.set_xlabel("Date", color='white', family='Cambria', size=15)
        ax.set_ylabel("Electric Vehicles Sales", color='white', family='Cambria', size=15)
        ax.tick_params(color='white', labelcolor='white')
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.legend()
        return fig, ax
