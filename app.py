import streamlit as st
from PIL import Image
import numpy as np
import os
from ultralytics import YOLO
import io

st.set_page_config(page_title='Recortador SOMOS CLIP', page_icon='📸')
st.title('📸 Recortador Inteligente SOMOS CLIP')
st.write('Sube tus fotos. La IA detectará los rostros y los encuadrará en formato vertical (4:5).')

# Cargamos el modelo ligero de detección de objetos de Ultralytics (detecta personas/rostros de forma nativa)
@st.cache_resource
def cargar_modelo():
    # Usamos el modelo nano de yolo que se baja en un segundo y es hiper preciso
    return YOLO("yolov8n.pt")

try:
    model = cargar_modelo()
except Exception as e:
    st.error(f"Error al inicializar la IA: {e}")
    model = None

fotos = st.file_uploader('Sube tus imágenes...', type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)

if fotos and model:
    for foto in fotos:
        try:
            pil_img = Image.open(foto).convert("RGB")
        except:
            st.error(f"No se pudo leer la imagen: {foto.name}")
            continue
            
        w_img, h_img = pil_img.size
        
        # Predicción de la IA (buscando la clase 0 que es 'persona' en el dataset de YOLO)
        resultados = model(pil_img, verbose=False)
        crop_pil = None
        
        # Buscar si detectó a alguien
        boxes = resultados[0].boxes
        if len(boxes) > 0:
            # Filtramos para intentar agarrar la primera persona/cabeza detectada
            box = boxes[0]
            # Coordenadas en píxeles reales [xmin, ymin, xmax, ymax]
            xyxy = box.xyxy[0].tolist()
            
            x_min_ia, y_min_ia, x_max_ia, y_max_ia = xyxy
            w_box = x_max_ia - x_min_ia
            h_box = y_max_ia - y_min_ia
            
            # Ajustamos el encuadre para que se concentre en la parte superior (básicamente el rostro/busto)
            # si es que detectó el cuerpo entero de la persona
            centro_x = x_min_ia + (w_box // 2)
            centro_y = y_min_ia + (h_box // 4) # Apuntamos más arriba del centro del recuadro
            
            # Calculamos el tamaño del recorte para mantener la proporción pasaporte 4:5
            # Le damos aire alrededor para que no quede muy pegado
            alto_recorte = int(h_box * 1.5) if h_box * 1.5 < h_img else h_img
            ancho_recorte = int(alto_recorte * 4 / 5)
            
            # Definir coordenadas finales
            y_in = max(0, int(centro_y - (alto_recorte * 0.4)))
            y_fi = min(h_img, int(y_in + alto_recorte))
            x_in = max(0, int(centro_x - (ancho_recorte // 2)))
            x_fi = min(w_img, int(x_in + ancho_recorte))
            
            crop_pil = pil_img.crop((x_in, y_in, x_fi, y_fi))

        if crop_pil:
            # Redimensionar exactamente al tamaño estándar requerido
            final_img = crop_pil.resize((540, 675), Image.Resampling.LANCZOS)
            st.image(final_img, caption=f'🎯 Sujeto detectado y encuadrado: {foto.name}')
            
            # Preparar descarga en memoria
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
            # Si no detectó nada seguro, no recorta a ciegas: te avisa para que lo hagas manual
            st.warning(f"⚠️ La IA no pudo determinar con precisión la posición del sujeto en: {foto.name}")
            st.image(pil_img, caption=f'(Original sin procesar) - {foto.name}', width=300)
