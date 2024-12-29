import os
import sys

from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Ridge, LinearRegression, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.exceptions import CustomException
from src.utils import find_best_model, save_object
from src.logger import logging

from xgboost import XGBRegressor

class ModelTrainer:
    def __init__(self, artifacts_dir = "artifacts"):
        
        self.artifacts_dir = artifacts_dir
        
    def model_train(self, train_data, test_data):
       
        try:
            
            logging.info("Model Trainer initiated.")
            
            models = {
                "Linear" : LinearRegression(),
                "Lasso": Lasso(),
                "Ridge": Ridge(),
                "ElasticNet": ElasticNet(),
                "SVR" : SVR(),
                "KNeighborsRegressor" : KNeighborsRegressor(),
                "DecisionTreeRegressor" : DecisionTreeRegressor(),
                "ExtraTreeRegressor" : ExtraTreeRegressor(),
                "RandomForestRegressor" : RandomForestRegressor(),
                "AdaBoostRegressor" : AdaBoostRegressor(),
                "GradientBoostingRegressor" : GradientBoostingRegressor(),
                "XGBRegressor" : XGBRegressor()
            }
            
            param_grid = {
                    "Linear" : {},
                    "Lasso": {
                        "alpha": [ 0.001, 0.01, 0.1, 1, 10]  # 'alpha' corresponds to Lasso's regularization parameter
                    },
                    "Ridge": {
                        "alpha": [ 0.001, 0.01, 0.1, 1, 10]  # 'alpha' corresponds to Ridge's regularization parameter
                    },
                    "ElasticNet": {
                        "alpha": [ 0.001, 0.01, 0.1, 1, 10],  # Regularization strength
                        "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9]  # Mix of L1 and L2 penalties
                    },
                    "SVR" : {
                        "C" : [0.1, 1, 10, 100],
                        "epsilon" : [0.01, 0.1, 1],
                        "kernel" : ["linear", 'rbf'],
                        "gamma" : [0.001, 0.01, 0.1, 1]
                    },
                    "KNeighborsRegressor" : {
                        "n_neighbors" : [4, 6, 8, 10],
                        "weights" : ['uniform', 'distance'],
                        "p" : [1, 2]
                    },
                    "DecisionTreeRegressor" : {
                        "max_depth" : [5, 10, 15, 20],
                        "min_samples_split" : [0, 2, 4, 6, 8, 10],
                        "min_samples_leaf" : [0, 2, 4, 6, 8, 10]
                    },
                    "ExtraTreeRegressor" : {
                        "max_depth" : [5, 10, 15, 20],
                        "min_samples_split" : [0, 2, 4, 6, 8, 10],
                        "min_samples_leaf" : [0, 2, 4, 6, 8, 10],
                        "min_weight_fraction_leaf" : [0, 0.3, 0.6, 0.9]
                    },
                    "RandomForestRegressor" : {
                        "max_depth" : [15, 20],
                        "min_samples_split" : [4, 6],
                        "min_samples_leaf" : [4, 6],
                        "min_weight_fraction_leaf" : [0, 0.3],
                        "n_estimators" : [100, 150]
                    },
                    "AdaBoostRegressor" : {
                        "n_estimators" : [50, 100, 150, 200],
                        "learning_rate" : [0, 0.2, 0.4, 0.6, 0.8],
                        "loss" : ['linear', 'square', 'exponential'] 
                    },
                    "GradientBoostingRegressor" : {
                        "learning_rate" : [0.01, 0.05],
                        "n_estimators" : [200],
                        "subsample" : [0.6],
                        "min_samples_split" : [3],
                        "min_samples_leaf" : [6, 9],
                        "min_weight_fraction_leaf" : [0.0, 0.1],
                        "max_depth" : [20]
                    },
                    "XGBRegressor" : {
                        "n_estimators" : [50, 100, 150, 200],
                        "max_depth" :  [5, 10, 15, 20],
                        "max_leaves" : [3, 5, 9],
                        "learning_rate" : [0.01, 0.05, 0.1]
                    }
                }
            
            logging.info("Data Splitting started.")
            
            X_train = train_data[:, :-1]
            Y_train = train_data[:, -1]
            
            X_test = test_data[:, :-1]
            Y_test = test_data[:, -1]
            
            logging.info("Data Splitting Completed.")
            
            logging.info("Model Training Started.")
            
            best_model = find_best_model(X_train, Y_train, X_test, Y_test, models, param_grid)
            
            logging.info("Model Training Completed.")
            
            logging.info(f"Best model found {best_model['best_model_name']}")
            
            best_model_instance = best_model["best_model_instance"]

            if best_model_instance is not None:
                save_object(best_model_instance, os.path.join(self.artifacts_dir, "model.pkl"))
                logging.info("Best Model Saved as Pickle file.")
            else:
                logging.error("No model instance found to save.")
                        
        except Exception as e:
            raise CustomException(e, sys)
        
        
if __name__ == "__main__":
    ingestion = DataIngestion(artifact_dir="artifacts")
    train_data, test_data = ingestion.split_and_save()

    data_transformation = DataTransformation()
    train_array, test_array = data_transformation.initiate_data_transformation(train_data, test_data)
    
    model_trainer = ModelTrainer()
    model_trainer.model_train(train_array, test_array)
    