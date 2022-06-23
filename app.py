# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 11:56:58 2022

@author: arunk
"""

# flask app
# importing libraries
import numpy as np
from collections.abc import Mapping
from flask import Flask, request, jsonify, render_template
import pickle

# flask app
app = Flask(__name__)
# loading model
model = pickle.load(open('sample_f.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')
@app.route('/prediction' ,methods = ['POST'])
def prediction():
    final_features = [float(x) for x in request.form.values()]
    final_features = [np.array(final_features)]
    prediction = model.predict(final_features)
    
    return render_template('index.html', output='Scheduler time for sample collection is {}'.format(prediction))


if __name__=="__main__":
    app.run(debug=True)