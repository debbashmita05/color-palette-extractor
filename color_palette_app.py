
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

st.set_page_config(page_title="Color Palette Extractor", page_icon="🎨")

st.title("🎨 Color Palette Extractor using K-Means")
st.write("Upload an image and choose the number of colors to extract a beautiful palette.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])


k = st.slider("Number of colors in the palette", 2, 10, 5)

if uploaded_file is not None:
    
    image = Image.open(uploaded_file)
    image = np.array(image)

   
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

   
    st.image(image, caption="Uploaded Image", use_container_width=True)

    pixels = image.reshape(-1, 3)

    
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(pixels)
    colors = kmeans.cluster_centers_.astype(int)

    
    palette = np.zeros((50, 300, 3), dtype=np.uint8)
    steps = 300 // k
    for idx, color in enumerate(colors):
        palette[:, idx*steps:(idx+1)*steps, :] = color

    
    st.image(palette, caption="Extracted Color Palette", use_container_width=False)

    st.subheader("RGB Values of Palette Colors:")
    for idx, color in enumerate(colors, start=1):
        st.write(f"Color {idx}: {tuple(color)}")
