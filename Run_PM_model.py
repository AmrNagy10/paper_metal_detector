import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

model = tf.keras.models.load_model('/content/best_model.keras')

img_path = '/content/image'

img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = img_array / 255.0  
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)
class_names = ['metal', 'paper']
result = class_names[int(prediction[0] > 0.5)]
confidence = prediction[0][0] if result == 'paper' else 1 - prediction[0][0]

plt.imshow(img)
plt.title(f'Its type is {result} by {confidence*100:.2f}%')
plt.axis('off')
plt.show()
