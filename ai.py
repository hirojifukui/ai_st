import streamlit as st
import os
from datetime import datetime
import numpy as np
from keras.utils import load_img, img_to_array
from keras.models import load_model

model_s = load_model("static/model/mnist_202311.h5")
path = ""; secure_files = []

def evaluate_img(path):
    img = load_img(path, color_mode ='grayscale', target_size=(28, 28))
    x = img_to_array(img)
    if np.average(x)-128 > 0:
       x = 255 - x
    x /= 255
    x = np.expand_dims(x, axis=0)
    y_proba = model_s.predict(x)
    result = y_proba.tolist()
    return (result)

def upload_file(file):
        new_filename = str(datetime.timestamp(datetime.now())) + file.name
        path = os.path.join(os.getcwd(), 
            'static/img/upload/', new_filename)
        with open(path,"wb") as f: 
            f.write(file.getbuffer())
        result = evaluate_img(path)
        pred = int(np.argmax(result[0]))
        #pred = int(np.argmax(result, axis=-1))
        # print(result[0])
        return path, result[0], pred

st.title("OCR Application")
img_file = st.file_uploader("Please select an image file (png, jpg, jpeg only)", type =['png', 'jpg','jpeg'])
if img_file is not None:
    path, result, pred = upload_file(img_file)
    st.image(img_file)
    st.write("Analysis result: ", pred)
    st.write("Probability: ", result)