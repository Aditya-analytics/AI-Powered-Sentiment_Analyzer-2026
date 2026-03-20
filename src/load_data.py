import os
import pandas as pd

class LoadDataset:
    def __init__(self, path):
        self.path = path

    def validate(self):
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"File not found: {self.path}")  # ✅ FIX
        return self.path

    def load_dataset(self):
        path = self.validate()
        df = pd.read_csv(path)
        return df
       
# data = LoadDataset(r"data\sample.csv")
# data = data.load_dataset()
# print(data)

            