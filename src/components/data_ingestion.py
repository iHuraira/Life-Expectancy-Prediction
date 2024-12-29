import os
import logging
import pandas as pd
from sklearn.model_selection import train_test_split

from src.exceptions import CustomException
import sys

class DataIngestion:
    def __init__(self, artifact_dir="artifacts"):
        """
        Initialize the DataIngestion object.

        Parameters:
            artifact_dir (str): Directory where artifacts (data files) will be saved.
        """
        
        self.artifact_dir = artifact_dir

        # Create the artifact directory if it doesn't exist
        os.makedirs(self.artifact_dir, exist_ok=True)
        logging.info(f"Artifact directory set to: {self.artifact_dir}")

    def split_and_save(self, test_size=0.2, random_state=42):
        """
        Split the data into train and test sets and save them as CSV files.

        Parameters:
            data (pd.DataFrame): Data to split.
            test_size (float): Proportion of the data to include in the test split.
            random_state (int): Random state for reproducibility.
        """
        try:
            raw_data =  pd.read_csv("notebook\data_cleaned.csv")
            
            logging.info("Splitting data into train and test sets.")
            
            train_data, test_data = train_test_split(raw_data, test_size=test_size, random_state=random_state)

            train_data.to_csv(os.path.join("artifacts", "train_data.csv"), index = False)
            
            logging.info("Train data saved.")
            
            test_data.to_csv(os.path.join("artifacts", "test_data.csv"), index = False)
            
            logging.info("Test data saved.")
            
            return train_data, test_data
        
        except Exception as e:
            CustomException(e, sys)

