import os
import io
import numpy as np
import cv2
from PIL import Image
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import tensorflow as tf
from rembg import remove
from openai import OpenAI

app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)

# Inicializar OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "insert_you_api_key"))

CLASES_MODELOS = {
    'sexo': ['Hombre', 'Mujer'],
    'gafas': ['Con Gafas', 'Sin Gafas'],
    'profesiones': ['Bombero', 'Camarero', 'Medico', 'Policia'],
    'emociones': ['Enfadado', 'Feliz', 'Neutral', 'Triste'],
    'pelo': ['Pelo Corto', 'Pelo Largo'],
    'edad': ['Adulto', 'Anciano', 'Joven', 'Niño']
}

# Cargar modelos
MODELOS = {}
for nombre in CLASES_MODELOS:
    try:
        MODELOS[nombre] = tf.keras.models.load_model(f'programa/modelos/modelo_{nombre}.h5')
    except Exception as e:
        print(f'[ADVERTENCIA] No se pudo cargar el modelo {nombre}: {e}')

# Modelo que no usa detección facial
MODELOS_SIN_ROSTRO = {'profesiones'}

# Cascade para rostro
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
IMG_SIZE = (224, 224)

def allowed_file(fn):
    return '.' in fn and fn.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'avif', 'webp'}

def resize_and_pad(img, size, pad_color=(255, 255, 255)):
    h, w = img.shape[:2]
    sh, sw = size
    scale = min(sw / w, sh / h)
    nw, nh = int(w * scale), int(h * scale)
    r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    top = (sh - nh) // 2
    bottom = sh - nh - top
    left = (sw - nw) // 2
    right = sw - nw - left
    return cv2.copyMakeBorder(r, top, bottom, left, right, cv2.BORDER_CONSTANT, value=pad_color)

def detectar_rostro(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    if not len(faces):
        return None
    x, y, w, h = faces[0]
    pad = int(0.4 * h)
    x1, y1 = max(x - pad, 0), max(y - pad, 0)
    x2 = min(x + w + pad, img.shape[1])
    y2 = min(y + h + pad, img.shape[0])
    return img[y1:y2, x1:x2]

def recortar_por_contorno(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contornos, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contornos:
        return None
    c = max(contornos, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    return img[y:y+h, x:x+w]

def preprocesar_por_modelo(file_bytes, modelo):
    try:
        no_bg = remove(file_bytes)
        img = Image.open(io.BytesIO(no_bg)).convert("RGB")
        cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        if modelo in MODELOS_SIN_ROSTRO:
            crop = recortar_por_contorno(cv_img)
            if crop is None:
                return None, "No se pudo detectar el contorno"
        else:
            crop = detectar_rostro(cv_img)
            if crop is None:
                return None, "No se detectó ningún rostro"

        resized = resize_and_pad(crop, IMG_SIZE)
        arr = np.array(resized) / 255.0
        return arr.reshape(1, 224, 224, 3), None

    except Exception as e:
        return None, f"Error procesando imagen: {e}"

def generar_historia(resultados):
    partes = []
    for categoria, info in resultados.items():
        top_pct = info['probabilidad']
        if top_pct >= 60:
            partes.append(f"{info['prediccion']} al {int(top_pct)}%")
        else:
            sorted_p = sorted(info['porcentajes'].items(), key=lambda x: x[1], reverse=True)[:3]
            desc = " / ".join(f"{lbl} {int(p)}%" for lbl, p in sorted_p)
            partes.append(f"entre {desc}")

    prompt = f"""
Eres un narrador creativo con mucho sentido del humor.

Tu tarea es escribir una historia divertida y breve (máximo 45 palabras) basada en características predecidas por una IA de visión artificial. Las predicciones incluyen atributos como edad, género, emociones, profesión, entre otros, junto con su porcentaje de certeza.

Si una etiqueta supera el 60% de probabilidad, úsala directamente en la historia, que se mencione claramente, 
solo en el caso de que upere el 60%.
Si ninguna etiqueta supera el 60% en una categoría, usa una descripción ambigua o mezcla de posibilidades 
No hables nunca de dos personas cuando haya lo comentado antes de etiquetas ambiguas. Siempre centrate en una sola y le haces la descripcion amigua. 
(por ejemplo, "alguien que podría ser camarero o médico").
La historia debe ser coherente, creativa, graciosa y visualmente sugerente.
No menciones porcentajes ni el hecho de que son predicciones.
No expliques lo que estás haciendo, solo escribe la historia.


Etiquetas con porcentajes:
{chr(10).join(partes)}
"""

    try:
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Eres un narrador creativo y divertido."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=100,
            temperature=0.7
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"Error al generar historia: {e}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predecir():
    if 'imagen' not in request.files:
        return jsonify({'error': 'No se envió imagen'}), 400

    file = request.files['imagen']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Archivo no válido'}), 400

    # ✅ Leer una vez los bytes de la imagen
    file_bytes = file.read()

    resultados = {}
    for nombre, model in MODELOS.items():
        # 👇 Usamos los mismos bytes para cada modelo
        img_proc, err = preprocesar_por_modelo(file_bytes, nombre)
        if err:
            return jsonify({'error': f"[{nombre}] {err}"}), 400

        preds = model.predict(img_proc)[0]
        idx = int(np.argmax(preds))
        etiq = CLASES_MODELOS[nombre][idx]
        resultados[nombre] = {
            'prediccion': etiq,
            'probabilidad': round(float(preds[idx]) * 100, 2),
            'porcentajes': {
                CLASES_MODELOS[nombre][i]: round(float(p) * 100, 2)
                for i, p in enumerate(preds)
            }
        }

    historia = generar_historia(resultados)
    return jsonify({'resultados': resultados, 'historia': historia})


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=5000)
