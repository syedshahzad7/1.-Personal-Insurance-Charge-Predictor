from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline


application = Flask(__name__)
app = application

@app.route('/')             #Route for home page
def index():
    return render_template('index.html')

@app.route('/predictdata',methods = ['GET', 'POST'])
def predict_datapoint():
    #in this function, we'll get our data from the form, apply the preprocessing object saved in the pkl file and then make predictions
    
    
    if request.method == 'GET':
        return render_template('home.html')             #home.html will have our form. 
    else:                                               #request.method == 'POST'. Here is where you have to get the data, standardize it and then make predictions.
        data = CustomData(                              #This is the class defined in the prediction pipeline that maps the data we received from user to the backend.
            sex = request.form.get('sex'),
            smokes = request.form.get('smokes'),
            region = request.form.get('region'),
            age = request.form.get('age'),
            bmi = request.form.get('bmi'),
            children = request.form.get('children')                            
        )
        pred_df = data.get_data_as_data_frame() 
        print(pred_df)

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)                 #results is a list because check the comment written at predict function of PredictPipe
        return render_template('home.html', results = results[0])
    

if __name__ == '__main__':
    app.run(host= "0.0.0.0", debug = True)
    
