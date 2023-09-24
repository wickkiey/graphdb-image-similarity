import streamlit as st
import numpy as np
from PIL import Image
import io
from neo4j_retrieve_utils import *


# Streamlit UI
st.title("Embeddings and Neo4j for Efficient Image Retrieval")

# Upload an image
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Display the uploaded image
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

    # save image to 'temp_dir' 
    temp_dir = './temp_dir/'
    image_path = temp_dir + uploaded_image.name
    with open(image_path, 'wb') as f:
        f.write(uploaded_image.getbuffer())
        
        
    
    # Find similar images and plot them
    image_paths = find_and_plot_similar_images(image_path, top_k=6)[1:]

    # Display the similar images of 6
    print(image_paths)
    st.header("Similar Images")
    
    for i, image_path in enumerate(image_paths):
        image = Image.open(image_path)
        st.image(image, caption=f"Image {i+1}", use_column_width=True)