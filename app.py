import streamlit as st
from PIL import Image
import os
import io

st.set_page_config(page_title='Recortador SOMOS CLIP', page_icon='📸', layout="wide")
st.title('📸 Recortador Preciso SOMOS CLIP')
st.write('Sube tus fotos. El sistema generará un encuadre vertical (4:5) que puedes ajustar manualmente para no cortar cabezas.')

fotos = st.file_uploader('Sube tus imágenes...', type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)

if fotos:
    for foto in fotos:
        st.markdown(f"### 🖼️ Procesando: {foto.name}")
        try:
            pil_img = Image.open(foto).convert("RGB")
        except:
            st.error(f"No se pudo leer la imagen: {foto.name}")
            continue
            
        w_img, h_img = pil_img.size
        
        # Crear dos columnas: controles a la izquierda, resultado a la derecha
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.write("🔧 **Ajuste de Encuadre**")
            
            # Si la imagen es más ancha que 4:5, controlamos el desplazamiento horizontal
            if w_img / h_img > 0.8:
                an_ideal = int(h_img * 0.8)
                max_x = w_img - an_ideal
                
                # Desplazamiento horizontal manual (por defecto al centro)
                centro_sugerido = max_x // 2
                offset_x = st.slider(f"Mover Izquierda/Derecha ({foto.name})", 0, max_x, centro_sugerido, key=f"x_{foto.name}")
                
                x_in = offset_x
                x_fi = x_in + an_ideal
                y_in = 0
                y_fi = h_img
            else:
                # Si la imagen es más alta que 4:5, controlamos el desplazamiento vertical (clave para no cortar cabezas)
                al_ideal = int(w_img / 0.8)
                max_y = h_img - al_ideal
                
                # Desplazamiento vertical manual (por defecto un poco más arriba del centro para capturar rostros)
                superior_sugerido = int(max_y * 0.25) if max_y > 0 else 0
                offset_y = st.slider(f"Mover Arriba/Abajo ({foto.name})", 0, max_y, superior_sugerido, key=f"y_{foto.name}")
                
                x_in = 0
                x_fi = w_img
                y_in = offset_y
                y_fi = y_in + al_ideal

            # Asegurar límites válidos
            x_in, y_in = max(0, x_in), max(0, y_in)
            x_fi, y_fi = min(w_img, x_fi), min(h_img, y_fi)

        with col2:
            # Realizar el recorte nativo con Pillow (0% OpenCV)
            crop_pil = pil_img.crop((x_in, y_in, x_fi, y_fi))
            
            if crop_pil:
                # Redimensionar exactamente al tamaño estándar requerido
                final_img = crop_pil.resize((540, 675), Image.Resampling.LANCZOS)
                
                # Mostrar el resultado final corregido
                st.image(final_img, caption=f"Resultado final 4:5 (540x675)")
                
                # Preparar el archivo de descarga en memoria
                ext = os.path.splitext(foto.name)[1].lower()
                formato = "JPEG" if ext in [".jpg", ".jpeg"] else "PNG"
                
                buffer = io.BytesIO()
                final_img.save(buffer, format=formato, quality=95)
                bytes_data = buffer.getvalue()
                
                st.download_button(
                    label=f'⬇️ Descargar {foto.name}', 
                    data=bytes_data, 
                    file_name=f'recut_{foto.name}', 
                    key=f"dl_{foto.name}"
                )
        st.markdown("---")
