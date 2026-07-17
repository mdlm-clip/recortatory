import streamlit as st
from PIL import Image
import numpy as np
import os
import mediapipe as mp
import io

st.set_page_config(page_title='Recortador SOMOS CLIP', page_icon='📸')
st.title('📸 Recortador Inteligente SOMOS CLIP')
st.write('Sube tus fotos. La IA detectará los rostros y los encuadrará en formato vertical (4:5).')

# Inicializar el detector de rostros de MediaPipe
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

fotos = st.file_uploader('Sube tus imágenes...', type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)

if fotos:
    for foto in fotos:
        try:
            pil_img = Image.open(foto).convert("RGB")
        except:
            st.error(f"No se pudo leer la imagen: {foto.name}")
            continue
            
        w_img, h_img = pil_img.size
        img_np = np.array(pil_img)
        
        # Procesar con la IA de MediaPipe
        resultado = face_detection.process(img_np)
        crop_pil = None
        
        if resultado.detections:
            # Tomamos el primer rostro detectado con precisión
            for detection in resultado.detections:
                bbox = detection.location_data.relative_bounding_box
                
                # Convertir coordenadas relativas a píxeles reales
                x = int(bbox.xmin * w_img)
                y = int(bbox.ymin * h_img)
                w = int(bbox.width * w_img)
                h = int(bbox.height * h_img)
                
                # Calcular márgenes verticales y horizontales para el plano 4:5
                y_in = max(0, int(y - h * 0.6))  # Margen superior para la cabeza
                y_fi = min(h_img, int(y + h * 2.2))  # Margen inferior
                an_re = int((y_fi - y_in) * 4 / 5)
                x_in = max(0, (x + w//2) - an_re//2)
                x_fi = min(w_img, x_in + an_re)
                
                # Ajuste por si el ancho calculado se desfasa
                if (x_fi - x_in) < an_re and x_in == 0:
                    x_fi = min(w_img, x_in + an_re)
                
                crop_pil = pil_img.crop((x_in, y_in, x_fi, y_fi))
                break
        
        if crop_pil:
            # Redimensionar al tamaño final requerido
            final_img = crop_pil.resize((540, 675), Image.Resampling.LANCZOS)
            st.image(final_img, caption=f'🎯 Rostro detectado: {foto.name}')
            
            # Preparar descarga
            ext = os.path.splitext(foto.name)[1].lower()
            formato = "JPEG" if ext in [".jpg", ".jpeg"] else "PNG"
            buffer = io.BytesIO()
            final_img.save(buffer, format=formato, quality=95)
            
            st.download_button(
                label=f'⬇️ Descargar Recorte', 
                data=buffer.getvalue(), 
                file_name=f'recut_{foto.name}', 
                key=foto.name
            )
        else:
            # Si la IA NO encuentra un rostro, te muestra la original y te avisa
            st.warning(f"⚠️ No se pudo detectar ningún rostro de forma automática en: {foto.name}")
            st.image(pil_img, caption=f'(Original sin recortar) - {foto.name}', width=300)
