import os
import time
import hashlib
import requests
from PIL import Image
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

# Ruta de destino
carpeta_destino = os.path.join(os.path.expanduser("~"), "Desktop", "policia")
os.makedirs(carpeta_destino, exist_ok=True)

# Configuración Selenium
options = webdriver.ChromeOptions()
# options.add_argument("--headless")  # Descomenta si quieres modo invisible
options.add_argument("--window-size=1920x1080")
options.add_argument("--log-level=3")

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
wait = WebDriverWait(driver, 10)

# URL base
url_base = "https://www.istockphoto.com/en/search/2/image?excludenudity=true&alloweduse=availableforalluses&phrase=police%20uniform%20close%20up&servicecontext=srp-related&page="

def scroll_pagina(driver, veces=10, espera=1):
    """Hace scroll hacia abajo para cargar más imágenes."""
    for _ in range(veces):
        driver.execute_script("window.scrollBy(0, window.innerHeight);")
        time.sleep(espera)

def descargar_imagenes(url_base, carpeta_destino, cantidad=3000, max_paginas=100):
    descargadas = 0
    descargadas_hash = set()
    pagina_actual = 1
    
    while descargadas < cantidad and pagina_actual <= max_paginas:
        url = f"{url_base}{pagina_actual}"
        driver.get(url)
        print(f"\nDescargando imágenes de la página {pagina_actual}...")

        try:
            wait.until(EC.presence_of_all_elements_located((By.TAG_NAME, "img")))
        except:
            print("No se encontraron imágenes en la página.")
            break

        # Hacer scroll para cargar todas las imágenes
        scroll_pagina(driver, veces=20, espera=0.5)

        imagenes = driver.find_elements(By.TAG_NAME, "img")

        for img in imagenes:
            try:
                img_url = img.get_attribute("src")
                
                if not img_url or not img_url.startswith("http") or "data:image" in img_url:
                    continue

                if hashlib.md5(img_url.encode()).hexdigest() in descargadas_hash:
                    continue

                response = requests.get(img_url, timeout=10)

                if response.status_code == 200:
                    img_data = response.content
                    img_obj = Image.open(requests.get(img_url, stream=True).raw)
                    ancho, alto = img_obj.size
                    if ancho < 224 or alto < 200:
                        print(f"Imagen demasiado pequeña ({ancho}x{alto}), omitida.")
                        continue

                    nombre_archivo = os.path.join(carpeta_destino, f"policia_{descargadas:05d}.jpg")
                    with open(nombre_archivo, "wb") as f:
                        f.write(img_data)

                    descargadas_hash.add(hashlib.md5(img_url.encode()).hexdigest())
                    descargadas += 1

                    print(f"Imagen descargada: {nombre_archivo}")

                else:
                    print(f"Error al descargar imagen: {response.status_code}")

                if descargadas >= cantidad:
                    break

            except Exception as e:
                print(f"Error al procesar imagen: {e}")
                continue

        pagina_actual += 1

    print(f"\n¡Descarga completada! {descargadas} imágenes descargadas.")

# Ejecutar
descargar_imagenes(url_base, carpeta_destino, cantidad=3000, max_paginas=100)

driver.quit()
