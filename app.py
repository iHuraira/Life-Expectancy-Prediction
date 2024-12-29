from flask import Flask, render_template, request
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

from src.pipelines.predict_pipeline import CustomData, PredictPipeline


application = Flask(__name__)

app = application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods = ['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
                hepatitis_b=request.form.get("hepatitis_b"),
                polio=request.form.get("polio"),
                diphtheria=request.form.get("diphtheria"),
                adult_mortality=request.form.get("adult_mortality"),
                infant_deaths=request.form.get("infant_deaths"),
                measles=request.form.get("measles"),
                under_five_deaths=request.form.get("under_five_deaths"),
                total_expenditure=request.form.get("total_expenditure"),
                hiv_aids=request.form.get("hiv_aids"),
                gdp=request.form.get("gdp"),
                population=request.form.get("population"),
                thinness_1_19_years=request.form.get("thinness_1_19_years"),
                thinness_5_9_years=request.form.get("thinness_5_9_years"),
                alcohol=request.form.get("alcohol"),
                bmi=request.form.get("bmi"),
                income_composition_of_resources=request.form.get("income_composition_of_resources"),
                schooling=request.form.get("schooling"),
                status=request.form.get("status", ""),
                percentage_expenditure=request.form.get("percentage_expenditure")
            )
    
        pred_data = data.get_data_as_data_frame()
        
        predict_pipeline = PredictPipeline()
        
        results = predict_pipeline.predict(pred_data)
        
        return render_template('home.html', results = results[0])
    
if __name__ == "__main__":
    app.run(debug=True)
