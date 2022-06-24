import numpy as np
from flask import Flask, request, jsonify, render_template    
import numpy as np    
import os
import base64
import io
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model
from keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array

app = Flask(__name__) 

model = load_model('prediction_model.h5')
print(" * Model loaded!")



@app.route('/')
def home():
    return render_template('index.html')



@app.route('/predict',methods=['POST'])
def predict():
    '''
    for rendering results in HTML GUI
    '''
    
    int_features = [int(x) for x in request.form.values()]
    print(int_features)
    final_features = np.array(int_features)
    print(final_features)
    print(final_features.shape)
    final_features=final_features.reshape(-1, 10)
    print(final_features)
    print(final_features.shape)
    
    prediction = model.predict(final_features)
    print(prediction)
    output = prediction
    print(output)
    print(type(output))
    
    return render_template('index.html',prediction_text = 'Driving Range Is  {}'.format(output))


if __name__ == "__main__":
    app.run(port= 8000 ,debug = True)