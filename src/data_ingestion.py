import os
import yaml
import pandas as pd


class Ingestion:
    def __init__(self, is_raw_data=True):
        with open('config.yml', 'r', encoding='utf-8') as file:
            data = yaml.safe_load(file)
        
        if is_raw_data:
            self.train_path = data['data']['train_path']
            self.test_path = data['data']['test_path']
        else:
            self.train_path = data['data']['processed_train_path']
            self.test_path = data['data']['processed_test_path']

        train_exists = os.path.exists(self.train_path)
        test_exists = os.path.exists(self.test_path)

        if not (train_exists and test_exists):
            raise FileNotFoundError(f'You need to load {self.train_path} and {self.test_path}')

    
    def load_data(self):
        train_df = pd.read_csv(self.train_path)
        test_df = pd.read_csv(self.test_path)

        return train_df, test_df
