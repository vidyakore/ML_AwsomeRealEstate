import numpy as np
import pandas as pd
from flask import Flask, request, render_template

from joblib import dump, load
import requests


def lala():
    return 'hello this is lala'



model = load('VC_Awsome.joblib')
df = pd.DataFrame()

app = Flask(__name__,template_folder='templates') 

@app.route('/') 
 
def hello_world(): 
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        a= [x for x in request.form.values()]
        

        if len(a) >  0 and  len(a) == 13:
            data=list((request.form['feature']).split())
            array_data =np.array(a).reshape(1,13)
            results = model.predict(array_data)
            results = str(results)
            return render_template('index.html',prediction_text= results)

        else:  
            
            return render_template('index.html', error_text='Please enter valid features! Input Invalid !!!')
        

    # return 'hello'

if __name__ == '__main__': 
    app.run() 
