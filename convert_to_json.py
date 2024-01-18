from tensorflow.keras.models import load_model
import tensorflowjs as tfjs

saved_model_path = './sign_language_model.h5'
model = load_model(saved_model_path)
save_to_path = "./"
tfjs.converters.save_keras_model(model, save_to_path)