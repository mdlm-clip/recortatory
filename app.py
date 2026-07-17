import streamlit as st
from PIL import Image
import numpy as np
import os
import mediapipe as mp
import io

st.set_page_config(page_title='Recortador SOMOS CLIP', page_icon='📸')
st.title('📸 Recortador Automático SOMOS CLIP')
st.write('Sube tus fotos y la IA las encuadrará en formato vertical (4:5).')

# Inicializar el detector de rostros de MediaPipe
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

fotos = st.file_uploader('Sube tus imágenes...', type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)

if fotos:
    for foto in fotos:
        # Leer imagen usando PIL en lugar de OpenCV
        try:
            pil_img = Image.open(foto).convert("RGB")
        except:
            continue
            
        w_img, h_img = pil_img.size
        img_np = np.array(pil_img)
        
        # Procesar con MediaPipe
        resultado = face_detection.process(img_np)
        recortado = False
        crop_pil = None
        
        if resultado.detections:
            for detection in resultado.detections:
                bbox = detection.location_data.relative_bounding_box
                
                # Convertir coordenadas relativas a píxeles
                x = int(bbox.xmin * w_img)
                y = int(bbox.ymin * h_img)
                w = int(bbox.width * w_img)
                h = int(bbox.height * h_img)
                
                # Calcular márgenes del encuadre
                y_in = max(0, int(y - h * 0.5))
                y_fi = min(h_img, int(y + h * 2.0))
                an_re = int((y_fi - y_in) * 4 / 5)
                x_in = max(0, (x + w//2) - an_re//2)
                
                # Asegurar que el ancho no desborde la imagen original
                x_fi = min(w_img, x_in + an_re)
                
                # Recortar usando PIL
                crop_pil = pil_img.crop((x_in, y_in, x_fi, y_fi))
                recortado = True
                break
                
        if not recortado or crop_pil is None:
            # Recorte por defecto si no detecta rostro (Centro 4:5)
            y_in, y_fi = int(h_img * 0.1), int(h_img * 0.9)
            an_re = int((y_fi - y_in) * 4 / 5)
            x_in = max(0, (w_img // 2) - (an_re // 2))
            x_fi = min(w_img, x_in + an_re)
            crop_pil = pil_img.crop((x_in, y_in, x_fi, y_fi))
            st.info(f"Encuadre estándar aplicado para: {foto.name}")

        if crop_pil:
            # Redimensionar al tamaño final requerido
            final_img = crop_pil.resize((540, 675), Image.Resampling.LANCZOS)
            
            # Mostrar previsualización en la web
            st.image(final_img, caption=f'Listo: {foto.name}')
            
            # Preparar buffer para descarga sin guardar en disco
            ext = os.path.splitext(foto.name)[1].lower()
            formato = "JPEG" if ext in [".jpg", ".jpeg"] else "PNG"
            
            buffer = io.BytesIO()
            final_img.save(buffer, format=formato)
            bytes_data = buffer.getvalue()
            
            st.download_button(
                label='⬇️ Descargar', 
                data=bytes_data, 
                file_name=f'recut_{foto.name}', 
                key=foto.name
            )
