import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'kisanai_model.h5')
IMG_SIZE = 224

def get_gradcam(model, img_array, last_conv_layer_name='top_conv'):
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def save_gradcam(image_path, output_path='gradcam_output.jpg'):
    model = tf.keras.models.load_model(MODEL_PATH)

    img = Image.open(image_path).resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img)
    img_processed = preprocess_input(img_array.copy())
    img_processed = np.expand_dims(img_processed, axis=0)

    heatmap = get_gradcam(model, img_processed)

    heatmap = np.uint8(255 * heatmap)
    jet = cm.get_cmap('jet')
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    jet_heatmap = tf.keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img_array.shape[1], img_array.shape[0]))
    jet_heatmap = tf.keras.utils.img_to_array(jet_heatmap)

    superimposed = jet_heatmap * 0.4 + img_array
    superimposed = tf.keras.utils.array_to_img(superimposed)
    superimposed.save(output_path)
    print(f"Grad-CAM saved to {output_path}")
    return output_path

if __name__ == '__main__':
    import sys
    image_path = sys.argv[1]
    save_gradcam(image_path, 'gradcam_output.jpg')