import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import mediapipe as mp

st.set_page_config(page_title='Recortador SOMOS CLIP', page_icon='📸')
st.title('📸 Recortador Automático SOMOS CLIP')
st.write('Sube tus fotos y la IA las encuadrará en formato vertical (4:5).')

# Inicializar el detector de rostros de MediaPipe
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

fotos = st.file_uploader('Sube tus imágenes...', type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)

if fotos:
    for foto in fotos:
        bytes_data = np.asarray(bytearray(foto.read()), dtype=np.uint8)
        img = cv2.imdecode(bytes_data, 1)
        if img is None: continue
        
        h_img, w_img = img.shape[:2]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resultado = face_detection.process(img_rgb)
        
        recortado = False
        
        if resultado.detections:
            # Tomamos el primer rostro detectado
            for detection in resultado.detections:
                bbox = detection.location_data.relative_bounding_box
                
                # Convertir coordenadas relativas a píxeles reales
                x = int(bbox.xmin * w_img)
                y = int(bbox.ymin * h_img)
                w = int(bbox.width * w_img)
                h = int(bbox.height * h_img)
                
                # Calcular márgenes simulando el comportamiento anterior
                y_in = max(0, int(y - h * 0.5))
                y_fi = min(h_img, int(y + h * 2.0))
                an_re = int((y_fi - y_in) * 4 / 5)
                x_in = max(0, (x + w//2) - an_re//2)
                
                crop = img[y_in:y_fi, x_in:x_in+an_re]
                recortado = True
                break # Solo procesamos el primer plano
                
        if not recortado:
            # Recorte por defecto si no detecta rostro (Encuadre al centro vertical/horizontal 4:5)
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
