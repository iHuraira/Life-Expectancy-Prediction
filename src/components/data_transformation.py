import os
import sys
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer

from src.components.data_ingestion import DataIngestion
from src.exceptions import CustomException
from src.utils import save_object

from src.logger import logging

class DataTransformation:
    def __init__(self, artifact_dir="artifacts"):
        self.artifact_dir = artifact_dir
        logging.info("Initialized DataTransformation with artifact_dir: %s", self.artifact_dir)

    def get_data_transformer_object(self):
        try:
            logging.info("Creating data transformer object.")

            numerical_columns = ['Hepatitis B', 'Polio', 'Diphtheria ',
                'Adult Mortality', 'infant deaths', 'Measles ', 'under-five deaths ',
                'Total expenditure', ' HIV/AIDS', 'GDP', 'Population',
                ' thinness  1-19 years', ' thinness 5-9 years', 'Alcohol', ' BMI ',
                'Income composition of resources', 'Schooling', 'percentage expenditure']

            categorical_columns = ['Status']


            num_pipeline = Pipeline(
                steps=[
                    ("scaler", StandardScaler())
                ]
            )
            logging.info("Numerical pipeline created.")

            cat_pipeline = Pipeline(
                steps=[
                    ("Onehotencoder", OneHotEncoder())
                ]
            )
            logging.info("Categorical pipeline created.")

            preprocessor = ColumnTransformer(
                transformers=[
                    ("numeric_pipeline", num_pipeline, numerical_columns),
                    ("categoric_pipeline", cat_pipeline, categorical_columns),
                ]
            )
            logging.info("Preprocessor object created.")

            return preprocessor

        except Exception as e:
            logging.error("Error in creating data transformer object: %s", str(e))
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_data, test_data):
        try:

            train_data = train_data.drop(['Country', 'Year'], axis = 1)
            test_data = test_data.drop(['Country', 'Year'], axis = 1)

            logging.info("Initiating data transformation process.")

            target_column = 'Life expectancy '

            X_train = train_data.drop(columns=[target_column], axis=1)
            y_train = train_data[target_column]
            logging.info("Split train data into features and target.")

            X_test = test_data.drop(columns=[target_column], axis=1)
            y_test = test_data[target_column]
            logging.info("Split test data into features and target.")

            preprocessor = self.get_data_transformer_object()

            X_train_transformed = preprocessor.fit_transform(X_train)
            logging.info("Transformed training features.")

            X_test_transformed = preprocessor.transform(X_test)
            logging.info("Transformed testing features.")

            train_combined = np.c_[X_train_transformed, y_train.to_numpy()]
            test_combined = np.c_[X_test_transformed, y_test.to_numpy()]
            logging.info("Combined transformed features with target for train and test datasets.")

            preprocessor_path = os.path.join(self.artifact_dir, "preprocessor.pkl")

            save_object(preprocessor, preprocessor_path)
            logging.info("Saved preprocessor object to %s", preprocessor_path)

            return train_combined, test_combined

        except Exception as e:
            logging.error("Error in data transformation process: %s", str(e))
            raise CustomException(e, sys)
