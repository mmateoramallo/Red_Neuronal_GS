import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from codecarbon import EmissionsTracker
import time

# --- PASO 1: Método de trackeo y cálculo de compensación ---
def ejecutar_entrenamiento_con_conciencia():
    tracker = EmissionsTracker()
    tracker.start()
    
    print("\n[SISTEMA] Iniciando trackeo de potencia y CO2...")
    
    # --- PASO 2: Carga de Dataset (Animales: pájaros, gatos, perros, ciervos, etc.) ---
    # CIFAR-10 ya viene integrado en TensorFlow
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
    
    # Normalizamos los píxeles (0 a 1) para que la CPU trabaje mejor
    train_images, test_images = train_images / 255.0, test_images / 255.0

    # --- PASO 3: Diseño de la Red Neuronal (CNN) ---
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10) # 10 categorías de animales/objetos
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # --- PASO 4: Entrenamiento ---
    print("[IA] Entrenando modelo de clasificación de animales...")
    # Usamos 5 épocas para que el consumo sea medible
    model.fit(train_images, train_labels, epochs=5, validation_data=(test_images, test_labels))
    
    model.save('modelo_animales_cifar10.h5')

    # Finalizamos trackeo
    emisiones_kg = tracker.stop()
    return emisiones_kg

def calcular_compensacion_arbol(emisiones_kg):
    # Un árbol consume 30kg de CO2 al año
    # 30kg / (365 días * 24 horas) = CO2 por hora
    co2_por_hora_arbol = 30 / (365 * 24)
    
    horas_necesarias = emisiones_kg / co2_por_hora_arbol
    
    print("-" * 50)
    print(f"RESULTADOS DEL TRABAJO PRÁCTICO")
    print("-" * 50)
    print(f"CO2 emitido por el entrenamiento: {emisiones_kg:.10f} kg")
    print(f"Para compensar este código, un árbol de 30kg/año")
    print(f"debe procesar durante: {horas_necesarias:.6f} horas.")
    print("-" * 50)

if __name__ == "__main__":
    emisiones = ejecutar_entrenamiento_con_conciencia()
    calcular_compensacion_arbol(emisiones)