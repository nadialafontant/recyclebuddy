import tensorflow as tf
import tensorflowjs as tfjs 
import os
import sys

# path to the trained 2-class model created by our training notebook
# this goes up one folder (from scripts/ → project root), then into models/
SAVED_MODEL_PATH = "../models/recyclebuddy_binary_model.h5"

# where to save the converted tf.js model files
# these files will be dropped into a folder in the same directory as this script
# (we'll later move this folder next to our index.html on our web server)
MODEL_SAVE_PATH = "tfjs_model_artifacts"


def export_model_to_tfjs():
    """
    loads your trained keras model and converts it into the tf.js format 
    so it can run directly in a browser.
    """

    # check that the model exists before converting
    if not os.path.exists(SAVED_MODEL_PATH):
        print(f"🚨 error: trained model not found at {SAVED_MODEL_PATH}")
        print("make sure your training notebook saved the model correctly.")
        print("if you're running this script from a different folder, update SAVED_MODEL_PATH.")
        sys.exit(1)

    print(f"loading trained model from: {SAVED_MODEL_PATH}")
    try:
        model_to_convert = tf.keras.models.load_model(SAVED_MODEL_PATH)
        print("model loaded successfully.")
    except Exception as e:
        print(f"error loading model: {e}")
        sys.exit(1)

    # make sure the output folder exists
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

    print(f"\nconverting keras model to tf.js format into '{MODEL_SAVE_PATH}/'...")
    try:
        # convert and save the model in tf.js layers format
        tfjs.converters.save_keras_model(model_to_convert, MODEL_SAVE_PATH)

        print("\n✅ conversion successful!")
        print(f"the tf.js model files are now in '{MODEL_SAVE_PATH}'")
        
    except Exception as e:
        print(f"error during tf.js conversion: {e}")
        print("make sure tensorflowjs is installed: pip install tensorflowjs")
        sys.exit(1)


if __name__ == "__main__":
    export_model_to_tfjs()