import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

generator = tf.keras.models.load_model("face_generator.keras")

noise = tf.random.normal([16, 100])
faces = generator(noise, training=False)
faces = (faces + 1) / 2.0

fig = plt.figure(figsize=(6, 6))
for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.imshow(faces[i])
    plt.axis('off')
plt.show()