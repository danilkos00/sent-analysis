import os
import joblib
import yaml
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.feature_extraction.text import TfidfVectorizer


class Trainer:
    def __init__(self):
        with open('config.yml', 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)

        self.model_name = config['model']['name']
        self.model_params = config['model']['params']
        self.store_path = config['model']['store_path']

        self.tfidf_params = config['processing']

        self.model = self._make_pipe()

    
    def train_model(self, train_data):
        X_train, y_train = train_data[['text']], train_data[['label']]
        # X_train, y_train = train_data['text'].values, train_data['label'].values

        # smote = SMOTE(
        #     sampling_strategy=1.0,
        #     random_state=self.model_params['random_state']
        # )
        rus = RandomOverSampler(random_state=42)

        X_train_res, y_train_res = rus.fit_resample(X_train, y_train)

        self.model.fit(X_train_res['text'].values, y_train_res['label'].values)
        # self.model.fit(X_train, y_train)


    def save_model(self):
        os.makedirs(self.store_path, exist_ok=True)
        model_path = os.path.join(self.store_path, 'model.pkl')
        joblib.dump(self.model, model_path)
        
    
    def _make_pipe(self):
        model_map = {
            'LogisticRegression': LogisticRegression,
            'LGBMClassifier': LGBMClassifier
        }

        model_class = model_map[self.model_name]

        model = model_class(**self.model_params)

        tfidf = TfidfVectorizer(**self.tfidf_params, ngram_range=(1, 3))

        pipe = Pipeline([
            ('tfidf', tfidf),
            ('model', model),
        ])

        return pipe
    
