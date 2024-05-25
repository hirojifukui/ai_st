import streamlit as st
import os
from datetime import datetime
import numpy as np
from keras.utils import load_img, img_to_array
from keras.models import load_model
import cv2
import pandas as pd

model_s = load_model("static/model/mnist_202311.h5")
path = ""; secure_files = []

def evaluate_img(path):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Convert a color image to a grayscale image 
    ret, thresh = cv2.threshold(gray, 0, 255,cv2.THRESH_OTSU|cv2.THRESH_BINARY_INV)
    # Convert the grayscale image to blak and white image
    # cv2.imshow('threh',thresh)
    # cv2.waitKey(0)
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    # Define size of kernel
    dilation = cv2.dilate(thresh, rect_kernel, iterations =8)
    #cv2.dilate makes the font thicker. 
    # cv2.imshow('dilation',dilation)
    # cv2.waitKey(0)
    contors, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.findCOntours find groups of continuous color/brightness changes
    sorted_ctrs = sorted(contors, key=lambda contors: cv2.boundingRect(contors)[0])
    # sort the contors by upper left coordinate
    #print("Contors: ", sorted_ctrs)
    im2 = dilation.copy()
    results=[]
    num_img = 0
    try:
        x=os.mkdir(path[:-4])
    except:
        print("Mkdir error", x)
    if len(sorted_ctrs) != 1:
        for i in sorted_ctrs:
            x,y,w,h = cv2.boundingRect(sorted_ctrs[num_img])
            # cv2.boundingRect returns the upper left cordinates, width, and height that 
            # can encloses a contour
            print("print x, y, w, h:", x, y, w, h)
            cropped = im2[y:y+h, x:x+w]
            # Extract an image from the larger image
            bg = np.zeros((28,28), np.uint8)
            if w>=h:
                resized = cv2.resize(cropped, (26, int(round(26*h/w))), interpolation=cv2.INTER_AREA)
                # cv2.resize resies the given image with specified interporation method
                rh, rw = resized.shape
                print("Shape:", rh, rw)
                bg[round((28-rh)/2):round((28-rh)/2)+rh, 1:27]=resized
            else:
                resized = cv2.resize(cropped, (int(round(26*w/h)), 26), interpolation=cv2.INTER_AREA)
                rh, rw = resized.shape
                print("Shape:", rh, rw)
                bg[1:27, round((28-rw)/2):round((28-rw)/2)+rw]=resized
            # cv2.imshow('resized_centered', bg)
            # cv2.waitKey(0)
            char_path = os.path.join(path[:-4], str(num_img)+".png")
            cv2.imwrite(char_path,bg)
            #img_path = char_path.split('streamlit_11\\')[1]
            x = img_to_array(bg)    
            #print ("Ave - 128: ", np.average(x)-128)
            if np.average(x)-128 > 0:
                x = 255 - x
                #print ("Inversed Ave: ", np.average(x))
            x /= 255
            x = np.expand_dims(x, axis=0)
            y_proba = model_s.predict(x)
            result = y_proba.tolist()
            pred= int(np.argmax(result[0], axis=-1))
            results.append([result[0], pred, char_path])
            num_img+=1
        return (results)
    else:
        img = load_img(path, color_mode ='grayscale', target_size=(28, 28))
        char_path = os.path.join(path[:-4], "0.png")
        x = img_to_array(img)
        if np.average(x)-128 > 0:
            x = 255 - x
        cv2.imwrite(char_path,x)
        x /= 255
        x = np.expand_dims(x, axis=0)
        y_proba = model_s.predict(x)
        result = y_proba.tolist()
        #print(result)
        pred= int(np.argmax(result[0], axis=-1))
        return ([[result[0], pred, char_path]])

def cimg(results):
    bimg = np.zeros((28,len(results)*28), np.uint8)
    for idx, x in enumerate(results):

        img = cv2.imread(x[2])
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #bimg[0:28, idx*28:idx*28+28] = np.zeros((28,28), np.uint8)
        bimg[0:28, idx*28:idx*28+28] = gray
    return(bimg)

def conv_df(results):
    pred = []; p=[]
    for e in range(10):
        p.append([])
    for r in results:
        pred.append(r[1])
        for idx, pv in enumerate(r[0]):
            p[idx].append(str(round(pv*100))+"%")

    df = pd.DataFrame({"Prediction": pred, "0":p[0], "1":p[1], "2":p[2], "3":p[3], "4":p[4], "5":p[5], "6":p[6], "7":p[7], "8":p[8], "9":p[9] })
    img = cimg(results)
    return(df, img)

def upload_file(file):
        new_filename = str(datetime.timestamp(datetime.now())) + file.name
        path = os.path.join(os.getcwd(), 
            'static/img/upload/', new_filename)
        with open(path,"wb") as f: 
            f.write(file.getbuffer())
        results = evaluate_img(path)
        df, img = conv_df(results)
        path_t = os.path.join(path, 'thumbnail')
        cv2.imwrite(path_t,img)
        #pred = int(np.argmax(result))
        #pred = int(np.argmax(result, axis=-1))
        return df, img#path, result[0], pred

st.title("OCR Application")
img_file = st.file_uploader("Please select an image file (png, jpg, jpeg only)", type =['png', 'jpg','jpeg'])
if img_file is not None:
    df, img = upload_file(img_file)
    #path, result, pred = upload_file(img_file)
    st.image(img_file)
    st.image(img)
    st.write("Analysis result: ", df)
