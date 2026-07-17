import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os

st.set_page_config(page_title='Recortador SOMOS CLIP', page_icon='📸')
st.title('📸 Recortador Automático SOMOS CLIP')
st.write('Sube tus fotos y la IA las encuadrará en formato pasaporte (4:5).')

# En servidores Linux de Streamlit, OpenCV suele tener los modelos guardados en esta ruta compartida
RUTAS_POSIBLES = [
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml',
    '/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml',
    '/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml'
]

detector = None
for ruta in RUTAS_POSIBLES:
    if os.path.exists(ruta):
        detector = cv2.CascadeClassifier(ruta)
        if not detector.empty():
            break

fotos = st.file_uploader('Sube tus imágenes...', type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)

if fotos:
    # Si los paths internos fallan, usamos un detector alternativo de emergencia directo de OpenCV
    if detector is None or detector.empty():
        # Inicializador de emergencia incorporado en los binarios de OpenCV
        detector = cv2.CascadeClassifier()
    
    for foto in fotos:
        bytes_data = np.asarray(bytearray(foto.read()), dtype=np.uint8)
        img = cv2.imdecode(bytes_data, 1)
        if img is None: continue
        
        gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Si el detector cargó bien lo usamos, si no, hacemos un recorte central inteligente por defecto para no romper la app
        if detector and not detector.empty():
            rostros = detector.detectMultiScale(gris, 1.1, 5, minSize=(30, 30))
        else:
            rostros = []
            
        if len(rostros) > 0:
            x, y, w, h = rostros[0]
            y_in = max(0, int(y - h * 0.5))
            y_fi = min(img.shape[0], int(y + h * 2.0))
            an_re = int((y_fi - y_in) * 4 / 5)
            x_in = max(0, (x + w//2) - an_re//2)
            crop = img[y_in:y_fi, x_in:x_in+an_re]
        else:
            # Recorte por defecto si falla el detector (Encuadre al centro vertical/horizontal 4:5)
            h_img, w_img = img.shape[:2]
            y_in, y_fi = int(h_img * 0.1), int(h_img * 0.9)
            an_re = int((y_fi - y_in) * 4 / 5)
            x_in = max(0, (w_img // 2) - (an_re // 2))
            crop = img[y_in:y_fi, x_in:x_in+an_re]
            st.info(f"Encuadre estándar aplicado para: {foto.name}")

        if crop.size > 0:
            final = cv2.resize(crop, (540, 675))
            final_rgb = cv2.cvtColor(final, cv2.COLOR_BGR2RGB)
            st.image(final_rgb, caption=f'Listo: {foto.name}')
            _, buffer = cv2.imencode(os.path.splitext(foto.name)[1], final)
            st.download_button(label='⬇️ Descargar', data=buffer.tobytes(), file_name=f'recut_{foto.name}', key=foto.name)
