from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)

app = application

## Route for home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            Private_Attorney = int(request.form.get('Private_Attorney')),	
            Marital_Status = int(request.form.get('Marital_Status')),
            Specialty = request.form.get('Specialty'),
            Insurance = request.form.get('Insurance'),
            Gender = request.form.get('Gender'),
            Age_Group = request.form.get('Age_Group'),
            Severity = int(request.form.get('Severity')),
            Attorney_Severity = int(request.form.get('Attorney_Severity'))
        )
        pred_df = data.get_data_as_data_frame()
        print(pred_df)

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        return render_template('home.html', results=results[0])

if __name__ == "__main__":
    app.run(host = "0.0.0.0")
