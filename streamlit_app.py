import tensorflow as tf
import numpy as np
import streamlit as st
from PIL import Image
import requests
from io import BytesIO

st.set_option('deprecation.showfileUploaderEncoding', False)
st.title("Pneumonie Detection")
st.text("Inserrer un URL d'une image de Poumou")

@st.cache(allow_output_mutation=True)
def load_model():
  model = tf.keras.models.load_model('/app/models/')
  return model

with st.spinner('Loading Model Into Memory....'):
  model = load_model()

classes=['NORMAL','PNEUMONIA']

def decode_img(image):
  img = tf.image.decode_jpeg(image, channels=3)  
  img = tf.image.resize(img,[224,224])
  return np.expand_dims(img, axis=0)

path = st.text_input("Veuillez entrer un URL d'image de poumon",'https://www.imbm-radiologie.com/wp-content/uploads/2020/03/Capture-d%E2%80%99e%CC%81cran-2020-03-19-a%CC%80-19.18.43.png')
if path is not None:
    content = requests.get(path).content

    st.write("Prediction ... :")
    with st.spinner('chargment.....'):
      label =np.argmax(model.predict(decode_img(content)),axis=1)
      st.write(classes[label[0]])    
    st.write("")
    image = Image.open(BytesIO(content))
    st.image(image, caption='Classification de Poumon', use_column_width=True)