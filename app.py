from flask import Flask, request, app,render_template
from flask import Response
import pickle
import numpy as np
import pandas as pd


app = Flask(__name__)


scaler=pickle.load(open("/config/workspace/model/standardScaler.pkl", "rb"))
model = pickle.load(open("/config/workspace/model/ModelForPrediction.pkl", "rb"))

## Route for homepage

@app.route('/')
def index():
    return render_template('index.html')

## Route for Single data point prediction
@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    result=""

    if request.method=='POST':

        ten_Year_Treasury_Yield=float(request.form.get("10 Year Treasury Yield"))
        three_Month_Treasury_Yield = float(request.form.get('3 Month Treasury Yield'))
        three_Month_Treasury_Yield_Bond_Equivalent_Basis = float(request.form.get('3 Month Treasury Yield (Bond Equivalent Basis)'))
        Spread = float(request.form.get('Spread'))
        Rec_prob = float(request.form.get('Rec_prob'))
        year = int(request.form.get('year'))
        month = int(request.form.get('month'))
        day = int(request.form.get('day'))
        

        new_data=scaler.transform([[ten_Year_Treasury_Yield,three_Month_Treasury_Yield,three_Month_Treasury_Yield_Bond_Equivalent_Basis,Spread,Rec_prob,year,month,day]])
        predict=model.predict(new_data)
       
        if predict[0] ==1 :
            result = 'Occured'
        else:
            result ='Not occured'
            
        return render_template('single_prediction.html',result=result)

    else:
        return render_template('home.html')


if __name__=="__main__":
    app.run(host="0.0.0.0")