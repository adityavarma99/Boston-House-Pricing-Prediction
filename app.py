import json
import pickle

from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd


# __name__ means starting point of the application from where it will run
app=Flask(__name__)

#loading the model
regmodel=pickle.load(open('regmodel.pkl','rb'))
scaler=pickle.load(open('scaling.pkl','rb'))

# app.route('/) means go to the homepage
@app.route('/')
def home():
    return render_template('home.html')

#creating predict_api, it is a post request,we are collecting the data using postman based on that we are getting the predicted output after passing the data trough scaler transformation(standardization) and regmodel.
@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data']
    print(data)
    ### getting the data from json will be key value pairs, to convert into dictionary values we use data.values and convert ino list
    ### after converting into list, reshape the values like np.array(  ).reshape(1,-1)
    print(np.array(list(data.values())[0]).reshape(1,-1))
    ## when we hit predict_api, input that we pass will be in json format and it will capured inside the 'data' and stored inside data
    new_data=scaler.transform(np.array(list(data.values())[0]).reshape(1,-1))
    output=regmodel.predict(new_data)
    print(output[0])
    return jsonify(output[0])
    ## as it is a 2d array, we give array[0]


# creating front end application to interact with predict_api
# new method='/predict'
@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    final_input=scaler.transform(np.array(data).reshape(1,-1))
    print(final_input)
    output=regmodel.predict(final_input)[0]
    return render_template("home.html",prediction_text="The House price prediction is {}".format(output))

if __name__=="__main__":
    app.run(debug=True)