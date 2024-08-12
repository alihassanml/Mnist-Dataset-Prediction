import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.preprocessing import image as keras_image
import matplotlib.pyplot as plt

model = tf.keras.models.load_model('model.h5')

st.title("MNIST Digit Predictor")

uploaded_file = st.file_uploader("Choose an image...", type=['png','jpg','jpeg','webp'])

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    img = Image.open(uploaded_file).convert('L')  # Convert to grayscale
    img = img.resize((28, 28))  # Resize to 28x28 pixels
    img_array = keras_image.img_to_array(img)
    img_array = img_array.reshape(1, 28, 28, 1)
    img_array = img_array / 255.0  # Normalize the image

    prediction = model.predict(img_array)
    predicted_digit = np.argmax(prediction)

    st.write(f"The model predicts this digit as: {predicted_digit}")

    plt.figure(figsize=(10, 4))
    plt.bar(range(10), prediction[0])
    plt.xlabel('Digit')
    plt.ylabel('Probability')
    plt.title('Prediction Probabilities')
    st.pyplot(plt)
