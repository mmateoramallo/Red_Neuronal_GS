import tensorflow as tf

print("Versión de TensorFlow:", tf.__version__)

# Listar dispositivos físicos detectados
devices = tf.config.list_physical_devices('GPU')

if len(devices) > 0:
    print(f"✅ ¡ÉXITO! Se detectaron {len(devices)} GPU(s):")
    for gpu in devices:
        print(f"  - {gpu}")
else:
    print("❌ NO se detectó GPU. TensorFlow usará solo la CPU (Ryzen 5 5600).")

# Prueba de ejecución rápida en GPU
if tf.test.is_built_with_cuda():
    print("🚀 TensorFlow fue compilado con soporte para CUDA.")
else:
    print("⚠️ Esta versión de TensorFlow no soporta CUDA (GPU).")   