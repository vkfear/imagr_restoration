# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 20:37:29 2020

@author: wreckk
"""

import streamlit as st
from PIL import Image
import cv2 
import numpy as np




def main():

    selected_box = st.sidebar.selectbox(
    'Choose one of the following',
    ('Welcome','Image Colorization', 'Team')
    )
    
    if selected_box == 'Welcome':
        welcome() 
    if selected_box == 'Image Colorization':
        photo()
    if selected_box == "Team":
        team()
 
def load_image(filename):
    image = cv2.imread(filename)
    return image
    
def welcome():
    
    st.title('Ancient Art and Heritage Restoration')
    
    st.subheader('DIGITAL RECREATORS')
    
    st.image('logo.jpeg',use_column_width=True)



 
def photo():
    
    #img_data = st.file_uploader(label='Drag and Drop Image',type =['png','jpg'])
    
    #if img_data is not None :
        
        #display image
        #uploaded_image = Image.open(img_data)
        #st.image(uploaded_image)

        
        

    net = cv2.dnn.readNetFromCaffe('./model/colorization_deploy_v2.prototxt','./model/colorization_release_v2.caffemodel')
    pts = np.load('./model/pts_in_hull.npy')



    class8 = net.getLayerId("class8_ab")
    conv8 = net.getLayerId("conv8_313_rh")
    pts = pts.transpose().reshape(2,313,1,1)

    net.getLayer(class8).blobs = [pts.astype("float32")]
    net.getLayer(conv8).blobs = [np.full([1,313],2.606,dtype='float32')]

    image1 = cv2.imread('./images/ansel_adams3.jpg')
    st.image(image1)
        
    scaled = image1.astype("float32")/255.0
    lab = cv2.cvtColor(scaled,cv2.COLOR_BGR2LAB)

    resized = cv2.resize(lab,(224,224))
    L = cv2.split(resized)[0]
    L -= 50

    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1,2,0))

    ab = cv2.resize(ab, (image1.shape[1],image1.shape[0]))

    L = cv2.split(lab)[0]
    colorized = np.concatenate((L[:,:,np.newaxis], ab), axis=2)

    colorized = cv2.cvtColor(colorized,cv2.COLOR_LAB2BGR)
    colorized = np.clip(colorized,0,1)

    colorized = (255 * colorized).astype("uint8")

    #cv2.imshow("Original",image1)
    #cv2.imshow("Colorized",colorized)
    st.image(colorized)

    cv2.waitKey(0)




def team():
    st.title('team page')
    
    st.subheader('DIGITAL RECREATORS')
       
if __name__ == "__main__":
    main()
