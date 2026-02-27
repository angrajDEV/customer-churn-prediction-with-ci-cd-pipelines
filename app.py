from flask import Flask, request, render_template
import pandas as pd 
import numpy as np

from src.pipelines.prediction import CustomData, PredictPipeline

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predict', methods=['GET','POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('predict.html')
    else:
        data = CustomData(
            SeniorCitizen=float(request.form.get('SeniorCitizen')),
            Partner=request.form.get('Partner'),
            Dependents=request.form.get('Dependents'),
            tenure=float(request.form.get('tenure')),
            PhoneService=request.form.get('PhoneService'),
            MultipleLines=request.form.get('MultipleLines'),
            InternetService=request.form.get('InternetService'),
            OnlineSecurity=request.form.get('OnlineSecurity'),
            OnlineBackup=request.form.get('OnlineBackup'),
            DeviceProtection=request.form.get('DeviceProtection'),
            TechSupport=request.form.get('TechSupport'),
            StreamingTV=request.form.get('StreamingTV'),
            StreamingMovies=request.form.get('StreamingMovies'),
            Contract=request.form.get('Contract'),
            PaperlessBilling=request.form.get('PaperlessBilling'),
            PaymentMethod=request.form.get('PaymentMethod'),
            MonthlyCharges=float(request.form.get('MonthlyCharges')),
            TotalCharges=float(request.form.get('TotalCharges')),
        )
        pred_df = data.get_data_as_data_frame()
        predict_pipeline = PredictPipeline()
        preds_class, preds_proba = predict_pipeline.predict(pred_df)
        churn = "Yes - Churn Hoga" if preds_class[0] == 1 else "No - Churn Nahi Hoga"
        churn_prob = round(preds_proba[0][1] * 100, 2)
        return render_template('predict.html', results=churn, probability=churn_prob)

if __name__ == '__main__':
    app.run(debug=True)