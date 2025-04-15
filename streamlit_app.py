pip install streamlit -q
wget -q -O - ipv4.icanhazip.com
npm install -g localtunnel
%%writefile app.py
import streamlit as st
import tensorflow as tf
st.write("""
         # Welcome to PneumoScan AI
         Your AI-Assisted X-ray Review for Paediatric Pneumonia
         """
         )
@st.cache_resource
def load_model():
    model=tf.keras.models.load_model('/kaggle/input/mymodel/my_model2.hdf5')
    return model
with st.spinner('Model is being loaded..'):
    model=load_model()
file = st.file_uploader("Please upload a chest x-ray file", type=["jpeg", "jpg", "png"])

from PIL import Image, ImageOps
import numpy as np

def import_and_predict(image_data, model):

    size = (224, 224)

    image = ImageOps.fit(image_data, size, Image.LANCZOS)
    image = np.asarray(image) / 255.0  # Convert to array and normalize

    image_reshape = np.expand_dims(image, axis=0)  


    prediction = model.predict(image_reshape)

    return prediction


if file is None:

    st.text("Please upload an x-ray image file")

else:

    image = Image.open(file).convert('RGB')  

    st.image(image, use_container_width=True)

    predictions = import_and_predict(image, model)


    class_names = ['Pneumonia', 'Normal']
    st.write(predictions)  
  

    st.write(

        "This image most likely represents a **{}** chest x-ray".format(

            class_names[np.argmax(predictions)]

        )
    )
! streamlit run app.py & npx localtunnel --port 8501
