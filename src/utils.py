import os
import sys
from src.logger import logging

import pickle
from src.exceptions import CustomException

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
import time

def save_object(obj, file_path):
        """Utility function to save an object as a pickle file."""
        try:
            with open(file_path, "wb") as file:
                pickle.dump(obj, file)
        except Exception as e:
            raise CustomException(e, sys)
        
def load_object(file_path):
    try:
        with open(file_path, 'rb') as file:
            obj = pickle.load(file)
        return obj
    except Exception as e:
        logging.error(f"Error loading object from {file_path}")
        raise CustomException(e)
        

def find_best_model(X_train, y_train, X_test, y_test, models, param_grid):
    best_model_name = None
    best_params = None
    best_r2_score = -float('inf')
    best_time = None
    best_model_instance = None  # Track the actual best model instance
    
    for model_name, model in models.items():
        try:
            grid = param_grid.get(model_name, {})
            grid_search = GridSearchCV(estimator=model,
                                       param_grid=grid,
                                       scoring="r2",
                                       cv=3,
                                       verbose=0,  # Reduce verbosity
                                       n_jobs=-1)

            start_time = time.time()
            grid_search.fit(X_train, y_train)
            end_time = time.time()

            elapsed_time = end_time - start_time
            best_model = grid_search.best_estimator_

            # Evaluate on test data
            y_pred = best_model.predict(X_test)
            test_r2 = r2_score(y_test, y_pred)

            if test_r2 > best_r2_score:
                best_r2_score = test_r2
                best_model_name = model_name
                best_params = grid_search.best_params_
                best_time = elapsed_time
                best_model_instance = best_model  # Save the actual best model

            logging.info(f"Model: {model_name}, Test R²: {test_r2:.4f}, Time: {elapsed_time:.2f}s")

        except Exception as e:
            logging.error(f"Error while processing model {model_name}: {e}")

    return {
        "best_model_name": best_model_name,
        "best_params": best_params,
        "best_r2_score": best_r2_score,
        "best_time": best_time,
        "best_model_instance": best_model_instance  # Include the model instance
    }
