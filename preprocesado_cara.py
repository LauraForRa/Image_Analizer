import os
import cv2
import numpy as np
from rembg import remove
from PIL import Image
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import io
import multiprocessing

input_dir = "C:/Users/fanat/OneDrive/Escritorio/data"
output_dir = "C:/Users/fanat/OneDrive/Escritorio/data_procesada"
img_size = (224, 224)
force_preprocess = False

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def preprocess_image(img_path):
    try:
        with open(img_path, 'rb') as f:
            input_image = f.read()

        # Eliminar fondo
        removed_bg = remove(input_image)

        # Cargar imagen sin fondo
        img = Image.open(io.BytesIO(removed_bg)).convert("RGB")
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        # Detectar rostro
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        if len(faces) == 0:
            print(f"⚠️ No se detectó rostro en {img_path}")
            return None

        # Tomar la primera cara detectada
        x, y, w, h = faces[0]

        # Expandir un poco el recorte
        padding = int(0.4 * h)
        x1 = max(x - padding, 0)
        y1 = max(y - padding, 0)
        x2 = min(x + w + padding, img_cv.shape[1])
        y2 = min(y + h + padding, img_cv.shape[0])
        face_crop = img_cv[y1:y2, x1:x2]

        # Redimensionar y centrar en fondo blanco
        face_resized = resize_and_pad(face_crop, img_size)

        return face_resized

    except Exception as e:
        print(f"Error procesando {img_path}: {e}")
        return None

def resize_and_pad(image, size, pad_color=(255, 255, 255)):
    h, w = image.shape[:2]
    sh, sw = size

    scale = min(sw / w, sh / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    pad_top = (sh - new_h) // 2
    pad_bottom = sh - new_h - pad_top
    pad_left = (sw - new_w) // 2
    pad_right = sw - new_w - pad_left

    padded = cv2.copyMakeBorder(resized, pad_top, pad_bottom, pad_left, pad_right,
                                borderType=cv2.BORDER_CONSTANT, value=pad_color)
    return padded

def process_and_save_image(file_info):
    in_path, out_path = file_info
    if os.path.exists(out_path) and not force_preprocess:
        return
    processed = preprocess_image(in_path)
    if processed is not None:
        cv2.imwrite(out_path, processed)

def process_dataset_parallel(input_folder, output_folder, force=False):
    if not os.path.isdir(input_folder):
        raise FileNotFoundError(f"El directorio de entrada no existe: {input_folder}")
    os.makedirs(output_folder, exist_ok=True)

    tasks = []

    for label in os.listdir(input_folder):
        label_folder = os.path.join(input_folder, label)
        output_label_folder = os.path.join(output_folder, label)
        os.makedirs(output_label_folder, exist_ok=True)

        for file in os.listdir(label_folder):
            in_path = os.path.join(label_folder, file)
            out_path = os.path.join(output_label_folder, file)
            if force or not os.path.exists(out_path):
                tasks.append((in_path, out_path))

    print(f"Total imágenes a procesar: {len(tasks)}")
    with ProcessPoolExecutor() as executor:
        list(tqdm(executor.map(process_and_save_image, tasks), total=len(tasks)))

if __name__ == "__main__":
    multiprocessing.freeze_support()
    process_dataset_parallel(input_dir, output_dir, force=force_preprocess)
