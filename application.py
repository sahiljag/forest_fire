from flask import Flask,request,jsonify,render_template
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# import ridge regressor and standard scaler

ridge_model = pickle.load(open("models/ridge.pkl",'rb'))
scaler_model = pickle.load(open("models/scaler.pkl",'rb'))

application = Flask(__name__)

@application.route("/")
def index():
    return render_template('index.html')


@application.route("/predict_data",methods=['GET','POST'])
def prediction():
    if request.method == 'POST':
        Temperature=float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))

        new_data_scaled = scaler_model.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])
        result = ridge_model.predict(new_data_scaled)
        return render_template('home.html',results = result[0])
    else:
        return render_template('home.html')






if __name__ ==  "__main__":
    application.run(host='0.0.0.0',debug=True)
