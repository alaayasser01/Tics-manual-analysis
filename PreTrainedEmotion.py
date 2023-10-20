import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
from deepface import DeepFace
import cv2
import numpy as np

def main():
    st.title("Emotion Analyzer App")
    
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    
    if uploaded_image is not None:
        # Display the uploaded image
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Convert the PIL image to an OpenCV image (numpy array)
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Analyze emotion using DeepFace
        result = DeepFace.analyze(image_cv, "emotion")
        emotion = result[0]['dominant_emotion']
        
        # Display the emotion result
        st.write(f"Detected Emotion: {emotion}")
        
        # You can also display other analysis results if needed
        # st.write(result)
        
if __name__ == "__main__":
    main()
