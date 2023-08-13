import os
import numpy as np

import glob
import tensorflow as tf
import tensorflow_addons as tfa

from skimage import io, exposure, transform

def get_image_paths():
    image_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "image")
    return glob.glob(os.path.join(image_dir, "*.*"))

def build():
    inputs=tf.keras.Input(shape=(224, 224, 3))
    x=inputs
    d=tf.keras.applications.DenseNet121(weights="imagenet", input_shape=(224, 224, 3), pooling="avg", include_top=False)
    x=d(x)
    x=tf.keras.layers.Dense(128, activation=tfa.activations.mish, activity_regularizer=tf.keras.regularizers.L1L2(1e-3, 1e-3))(x)
    x=tf.keras.layers.Dropout(0.5)(x)
    x=tf.keras.layers.Dense(128, activation=tfa.activations.mish, activity_regularizer=tf.keras.regularizers.L1L2(1e-3, 1e-3))(x)
    x=tf.keras.layers.Dropout(0.5)(x)
    class Sampling(tf.keras.layers.Layer):
        def call(self, inputs):
            z_mean, z_log_var = inputs
            batch =tf.shape(z_mean)[0]
            dim   =tf.shape(z_mean)[1]
            epsilon = tf.random.normal(shape=(batch, dim))
            return z_mean +tf.exp(0.5 *z_log_var) *epsilon
    x=[tf.keras.layers.Dense(128, activity_regularizer=tf.keras.regularizers.L1L2(1e-3, 1e-3))(x), tf.keras.layers.Dense(128, activity_regularizer=tf.keras.regularizers.L1L2(1e-3, 1e-3))(x)]
    x=Sampling()(x)
    x=tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1), name="latent")(x)
    latent=x
    x=tf.keras.layers.Dense(1, activation="sigmoid", name="binary")(x)
    binary=x
    x=tf.keras.Model(inputs, [latent, binary])
    weights_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model", "DenseNet121-Triplet-ImageNet.h5")
    x.load_weights(weights_path)
    x.trainable=False
    # done!
    return x
    
def read_preprocess(x, augment=15):
    x=io.imread(x)
    x=exposure.adjust_log(x)
    d=np.abs(x.shape[0] - x.shape[1]) // 2
    if x.shape[0] > x.shape[1]: x=np.pad(x, ((0, 0), (d, d), (0, 0)), mode="constant")
    if x.shape[0] < x.shape[1]: x=np.pad(x, ((d, d), (0, 0), (0, 0)), mode="constant")
    x=transform.resize(x, (224, 224), anti_aliasing=True)
    if augment == 0:
        x=[x]
    else:
        x=[transform.rotate(x, np.random.uniform(-360, 360)) for _ in range(augment)]
    x=np.stack(x)
    return x
    
def make_prediction(x, model, threshold):
    x=model.predict(x, verbose=0)
    x=x[1]
    x=x.reshape(np.prod(x.shape))
    x=(x >=threshold).astype(int)
    if len(set(x)) == 1:
        Prediction=["Robust", "Susceptible"][x[0]]
        Confidence=100
        return {"Prediction" : Prediction, "Confidence" : Confidence}
    else:
        d=np.bincount(x)
        Prediction=["Robust", "Susceptible"][np.argmax(d)]
        Confidence=d[np.argmax(d)] / np.sum(d) * 100
        return {"Prediction" : Prediction, "Confidence" : Confidence}
        
