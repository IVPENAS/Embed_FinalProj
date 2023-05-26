import streamlit as st
import tensorflow as tf
import cv2
from PIL import Image,ImageOps
import numpy as np

@st.cache_resource
def load_model():
  model=tf.keras.models.load_model('weights-improvement-16-0.89.hdf5')
  return model
model=load_model()
st.title(" Cat and Dog Classifier 🐱🐶")
st.write("Final Exam: Model Deployment in the Cloud")
st.image('https://github.com/IVPENAS/Embed_FinalProj/assets/111822151/07779232-ea8c-4e2f-8a9f-09dba009c803')
file=st.file_uploader("Choose a photo from your computer",type=["jpg","png"])

def import_and_predict(image_data,model):
    size=(128,128)
    image = ImageOps.fit(image_data,size, Image.LANCZOS)
    image = np.asarray(image)
    image = image / 255.0
    img_reshape = np.reshape(image, (1, 128, 128, 3))
    prediction = model.predict(img_reshape)
    return prediction
if file is None:
    st.text("Please upload an image file")
else:
    image=Image.open(file)
    st.image(image,use_column_width=True)
    prediction=import_and_predict(image,model)
    class_names=['Apple', 'Tomato']
    string="OUTPUT : "+class_names[np.argmax(prediction)]
    st.success(string)
    
st.sidebar.success("Details")
st.sidebar.text("wow")
    


#Final Exam: Model Deployment in the Cloud

#![n-cats-a-20181226](https://github.com/IVPENAS/Embed_FinalProj/assets/111822151/07779232-ea8c-4e2f-8a9f-09dba009c803)

#Created By:
#- Peñas, Issa Victoria H.
#- Villamor, Earl Kristian G.

#<p align = "justify"> Due to our interest about animals specifically Cats and Dogs, allowed us to create a custom dataset to our Classifier utilized on Convolutional Neural Network learning model which is designed to distinguish using [number of ssize] pictures between the mentioned animals. The Classfier starts by inputting a image into the network which it'll be processed through multiple layers of convolutional filters allowing to distinguish patterns and features present in the inputted image, in which the network will gradually determine the different characteristics of a cat and dog including [1]<b><i> ears</i></b>, [2]]<b><i> tail</i></b>, [3]<b><i> snout</i></b>, or [4]<b><i> eyes</i></b> which by then inputed to the fully conncted layers consolidating the informmation and making predictions about the image's class. By modifying the weights and biases of its network based on the discrepancies between predicted and real labels throughout the training phase will result an improvement upon the classifier. The model can iteratively adjust its parameters through a method called backpropagation to increase its accuracy over time. Once the model was trained and completed the classifier can be inputted images from the user and output the preditions whether the images contains a cat or a dog.
#</p>
