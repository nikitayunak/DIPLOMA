import pandas as pd


class FileProcessor:
    def __init__(self):
        self.df = pd.DataFrame()

    def load_file(self, filepath):
        self.df = pd.read_csv(filepath)
        return self.df
