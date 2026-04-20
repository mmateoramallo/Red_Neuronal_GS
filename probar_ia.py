import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

# --- Configuración ---
MODEL_PATH = 'modelo_animales_cifar10.h5'
# Las 10 clases de CIFAR-10 en orden
CLASSES = ['avión', 'automóvil', 'pájaro', 'gato', 'ciervo', 
           'perro', 'rana', 'caballo', 'barco', 'camión']

# --- Cargar el modelo entrenado ---
print(f"[SISTEMA] Cargando modelo desde {MODEL_PATH}...")
model = tf.keras.models.load_model(MODEL_PATH)

# --- Función de Predicción ---
def predecir_imagen(file_path):
    # 1. Cargar y preprocesar la imagen para que coincida con el entrenamiento (32x32)
    img = image.load_img(file_path, target_size=(32, 32))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # Normalizar (igual que en el entrenamiento)
    img_array = np.expand_dims(img_array, axis=0)  # Crear el batch (1, 32, 32, 3)

    # 2. Hacer la predicción
    prediction = model.predict(img_array)
    score = tf.nn.softmax(prediction[0]) # Convertir a probabilidades
    
    # 3. Obtener la clase con mayor probabilidad
    class_idx = np.argmax(score)
    category = CLASSES[class_idx]
    confidence = 100 * np.max(score)
    
    return category, confidence

# --- Interfaz Gráfica (Tkinter) ---
def open_file():
    file_path = filedialog.askopenfilename()
    if not file_path:
        return
    
    # Mostrar la imagen en la interfaz
    img = Image.open(file_path)
    img = img.resize((250, 250), Image.Resampling.LANCZOS)
    img_tk = ImageTk.PhotoImage(img)
    label_image.config(image=img_tk)
    label_image.image = img_tk
    
    # Hacer la predicción
    category, confidence = predecir_imagen(file_path)
    label_result.config(text=f"Predicción: {category.upper()}\nConfianza: {confidence:.2f}%", fg="blue")

# Configuración de la ventana principal
root = tk.Tk()
root.title("Clasificador de Animales (TP Conciencia)")
root.geometry("400x500")

# Elementos de la interfaz
btn_open = tk.Button(root, text="Cargar Imagen de Animal", command=open_file, font=("Arial", 12))
btn_open.pack(pady=20)

label_image = tk.Label(root) # Espacio para la imagen
label_image.pack()

label_result = tk.Label(root, text="Esperando imagen...", font=("Arial", 14, "bold"))
label_result.pack(pady=20)

print("[SISTEMA] Interfaz lista. ¡Cargá una foto!")
root.mainloop()