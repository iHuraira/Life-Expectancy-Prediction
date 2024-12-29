import sys
import os

import numpy as np
from src.exceptions import CustomException
from src.logger import logging
import pandas as pd

from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass
 
    def predict(self, features):
        try:
            model_path = "artifacts/model.pkl"
            preprocessor_path = "artifacts/preprocessor.pkl"       

            model = load_object(model_path)
            preprocessor = load_object(preprocessor_path)

            desired_columns = ['Hepatitis B', 'Polio', 'Diphtheria ',
                            'Adult Mortality', 'infant deaths', 'Measles ', 'under-five deaths ',
                            'Total expenditure', ' HIV/AIDS', 'GDP', 'Population',
                            ' thinness  1-19 years', ' thinness 5-9 years', 'Alcohol', ' BMI ',
                            'Income composition of resources', 'Schooling',
                            'Status', 'percentage expenditure']

            # Ensure the number of columns match
            if len(features.columns) == len(desired_columns):
                features.columns = desired_columns
            else:
                raise ValueError("The number of columns in the DataFrame does not match the desired column names.")

            data_scaled = preprocessor.transform(features)
            
            data_scaled = np.nan_to_num(data_scaled, nan=0)
            
            preds = model.predict(data_scaled)
            
            return preds
        
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self, 
                 hepatitis_b: float, 
                 polio: float, 
                 diphtheria: float, 
                 adult_mortality: float, 
                 infant_deaths: float, 
                 measles: float, 
                 under_five_deaths: float, 
                 total_expenditure: float, 
                 hiv_aids: float, 
                 gdp: float, 
                 population: float, 
                 thinness_1_19_years: float, 
                 thinness_5_9_years: float, 
                 alcohol: float, 
                 bmi: float, 
                 income_composition_of_resources: float, 
                 schooling: float, 
                 status: str, 
                 percentage_expenditure: float):
        
        self.hepatitis_b = hepatitis_b
        self.polio = polio
        self.diphtheria = diphtheria
        self.adult_mortality = adult_mortality
        self.infant_deaths = infant_deaths
        self.measles = measles
        self.under_five_deaths = under_five_deaths
        self.total_expenditure = total_expenditure
        self.hiv_aids = hiv_aids
        self.gdp = gdp
        self.population = population
        self.thinness_1_19_years = thinness_1_19_years
        self.thinness_5_9_years = thinness_5_9_years
        self.alcohol = alcohol
        self.bmi = bmi
        self.income_composition_of_resources = income_composition_of_resources
        self.schooling = schooling
        self.status = status
        self.percentage_expenditure = percentage_expenditure

    def get_data_as_data_frame(self):
        try:
            data = {
                "Hepatitis B": [self.hepatitis_b],
                "Polio": [self.polio],
                "Diphtheria": [self.diphtheria],
                "Adult Mortality": [self.adult_mortality],
                "Infant Deaths": [self.infant_deaths],
                "Measles": [self.measles],
                "Under-Five Deaths": [self.under_five_deaths],
                "Total Expenditure": [self.total_expenditure],
                "HIV/AIDS": [self.hiv_aids],
                "GDP": [self.gdp],
                "Population": [self.population],
                "Thinness 1-19 Years": [self.thinness_1_19_years],
                "Thinness 5-9 Years": [self.thinness_5_9_years],
                "Alcohol": [self.alcohol],
                "BMI": [self.bmi],
                "Income Composition of Resources": [self.income_composition_of_resources],
                "Schooling": [self.schooling],
                "Status": [self.status],
                "Percentage Expenditure": [self.percentage_expenditure]
            }

            return pd.DataFrame(data)
        except Exception as e:
            raise CustomException(e, sys)
        