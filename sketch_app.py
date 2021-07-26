import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt

uploaded_img = st.file_uploader("Upload Image", ['png','jpeg','jpg'])
image_view_scale = 100
ORIGINAL_IMG_W = None
ORIGINAL_IMG_H = None
EQULIZE_HISTOGRAM = False

with st.sidebar:
	st.header('Side Bar Menu')
	image_view_scale = st.slider("Image Viewport Scaling %", 10, 100, 100)

	brightness_slider = st.slider("Brightnes Slider", 0, 100, 0)
	contrast_slider = st.slider("Contrast Slider",-3.0, 3.0, 1.0)

	show_histogram = st.checkbox("Show Histogram")
	histogram_container = st.beta_container()

	show_sketch = st.checkbox("Show Sketch?")
	if show_sketch:
		st.subheader("GaussianBlur Controls")
		gaussian_kernel_control = st.slider("Kernel Size (n x n)", 3, 61, 5, 2)
		gaussian_sigma_control_X = st.slider("Sigma X", 0, 20, 0, 1)
		gaussian_sigma_control_Y = st.slider("Sigma Y", 0, 20, 0, 1)




if uploaded_img is not None:
	file_bytes = np.asarray(bytearray(uploaded_img.read()), dtype=np.uint8)
	img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
	display_img = img.copy()

	display_img = cv2.convertScaleAbs(display_img, alpha=contrast_slider, beta=brightness_slider)

	ORIGINAL_IMG_H, ORIGINAL_IMG_W, _ = img.shape

	view_w = int(ORIGINAL_IMG_W * (image_view_scale / 100))
	view_h = int(ORIGINAL_IMG_H * (image_view_scale / 100))


	grayscale = cv2.cvtColor(display_img, cv2.COLOR_BGR2GRAY)
	if EQULIZE_HISTOGRAM:
		st.write("Equalize")
		grayscale = cv2.equalizeHist(grayscale)

	invert = cv2.bitwise_not(grayscale)

	#if image_view_scale != 100:
	#	display_img = cv2.resize(img, (view_w, view_h))

	if show_histogram:
		with histogram_container:
			hist_graph_container = st.beta_container()
			if st.checkbox("Equalize Histogram"):
				EQULIZE_HISTOGRAM = True
				grayscale = cv2.equalizeHist(grayscale)

		with hist_graph_container:
			fig, ax = plt.subplots()
			ax.hist(grayscale.flatten(), bins=256, range=[0, 256])
			st.pyplot(fig)


	if show_sketch:
		gaussian_blur = cv2.GaussianBlur(invert, (gaussian_kernel_control , gaussian_kernel_control), gaussian_sigma_control_X, gaussian_sigma_control_Y)
		dodge_burn = cv2.divide(grayscale, 255-gaussian_blur, scale=256)
		st.image(cv2.cvtColor(dodge_burn, cv2.COLOR_BGR2RGB), width=view_w)
	else:
		st.image(cv2.cvtColor(grayscale, cv2.COLOR_BGR2RGB), width=view_w)

