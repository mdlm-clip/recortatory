import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
st.set_page_config(page_title='Recortador SOMOS CLIP', page_icon='📸')
st.title('📸 Recortador Automático SOMOS CLIP')
st.write('Sube tus fotos y la IA las encuadrará en formato pasaporte (4:5).')
detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
fotos = st.file_uploader('Sube tus imágenes...', type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)
if fotos:
    for foto in fotos:
        bytes_data = np.asarray(bytearray(foto.read()), dtype=np.uint8)
        img = cv2.imdecode(bytes_data, 1)
        if img is None: continue
        gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rostros = detector.detectMultiScale(gris, 1.1, 5, minSize=(30, 30))
        if len(rostros) > 0:
            x, y, w, h = rostros[0]
            y_in = max(0, int(y - h * 0.5))
            y_fi = min(img.shape[0], int(y + h * 2.0))
            an_re = int((y_fi - y_in) * 4 / 5)
            x_in = max(0, (x + w//2) - an_re//2)
            crop = img[y_in:y_fi, x_in:x_in+an_re]
            final = cv2.resize(crop, (540, 675))
            final_rgb = cv2.cvtColor(final, cv2.COLOR_BGR2RGB)
            st.image(final_rgb, caption=f'Listo: {foto.name}')
            _, buffer = cv2.imencode(os.path.splitext(foto.name)[1], final)
            st.download_button(label='⬇️ Descargar', data=buffer.tobytes(), file_name=f'recut_{foto.name}')
        else:
            st.warning(f'❌ No se detectó rostro en: {foto.name}')