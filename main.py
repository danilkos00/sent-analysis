import logging
import mlflow
import mlflow.sklearn
import yaml
from src.data_ingestion import Ingestion
from src.data_processing import process_data
from src.train import Trainer
from src.predict import Predictor


logging.basicConfig(level=logging.INFO,format='%(asctime)s:%(levelname)s:%(message)s')


def main():
    with open('config.yml', 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)

    mlflow.set_experiment('Sentiment model training experiment')

    with mlflow.start_run() as run:
        ingestor = Ingestion()
        train_data, test_data = ingestor.load_data()

        logging.info("Data ingestion completed successfully")

        processed_train = process_data(train_data)
        processed_test = process_data(test_data)

        logging.info('Data processing completed successfully')
        trainer = Trainer()
        trainer.train_model(processed_train)
        trainer.save_model()

        logging.info('Model training completed successfully')

        predictor = Predictor()
        precision, recall, f1, class_report = predictor.evaluate_model(processed_test)

        mlflow.set_tag('Model developer', 'danilkos')
        mlflow.set_tag('preprocessing', 'TfidfVectorizer')

        mlflow.log_params(config['model']['params'])
        mlflow.log_params({f"tfidf__{k}": v for k, v in config['processing'].items()})

        mlflow.log_metric('precision', precision)
        mlflow.log_metric('recall', recall)
        mlflow.log_metric('macro_f1_score', f1)
        mlflow.sklearn.log_model(predictor.model, 'model')

        model_name = "sentiment_model" 
        model_uri = f"runs:/{run.info.run_id}/model"
        mlflow.register_model(model_uri, model_name)

        print("\n============= Model Evaluation Results ==============")
        print(f"Model: {trainer.model_name}")
        print(f"\n{class_report}")
        print("=====================================================\n")


if __name__ == '__main__':
    main()
