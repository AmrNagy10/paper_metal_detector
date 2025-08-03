from google.colab import drive
from sklearn.utils import class_weight
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

train_dataset = tf.keras.utils.image_dataset_from_directory(
    '/content/DataSets/train',
    label_mode='binary',
    shuffle=True,        
    batch_size=16,
    image_size=(224, 224)
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    '/content/DataSets/validation',
    label_mode='binary',
    shuffle=False,       
    batch_size=16,
    image_size=(224, 224)
)



data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2),
])

AUTOTUNE = tf.data.AUTOTUNE


def prepare(ds, augment=False):
    if augment:
        ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)
    return ds.cache().prefetch(buffer_size=AUTOTUNE)

train_ds = prepare(train_dataset, augment=True)
val_ds = prepare(validation_dataset)


labels = np.concatenate([y for x, y in train_ds], axis=0)
labels = labels.flatten()

weights = class_weight.compute_class_weight(
    'balanced',
    classes=np.unique(labels),
    y=labels
)

class_weights = {0: weights[0], 1: weights[1]}

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False
inputs = tf.keras.Input(shape=(224, 224, 3))
x = tf.keras.layers.Rescaling(1./255)(inputs)
x = base_model(x, training=False)

x = tf.keras.layers.GlobalAveragePooling2D()(x) 
x = tf.keras.layers.Dense(64, activation='relu')(x) 
x = tf.keras.layers.Dropout(0.2)(x)
output = tf.keras.layers.Dense(1, activation='sigmoid')(x)


model = tf.keras.Model(inputs, output)

checkpoint_cb = tf.keras.callbacks.ModelCheckpoint("best_model.keras", save_best_only=True, monitor="val_accuracy")
early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor="val_accuracy")
reduce_lr_cb = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, verbose=1)
callbacks_list = [checkpoint_cb, early_stopping_cb, reduce_lr_cb]


print("بدء مرحلة التدريب الأولية (Feature Extraction)...")

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(
    train_ds,  
    epochs=20,  
    validation_data=val_ds,
    callbacks=callbacks_list,
    class_weight=class_weights 
)


print("\nبدء مرحلة الضبط الدقيق (Fine-Tuning)...")
base_model.trainable = True
for layer in base_model.layers[:100]:
    layer.trainable = False


model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='binary_crossentropy',
    metrics=['accuracy']
)


fine_tune_epochs = 10
total_epochs = len(history.history['accuracy']) + fine_tune_epochs

history_fine = model.fit(
    train_ds,
    epochs=total_epochs,
    initial_epoch=len(history.history['accuracy']),
    validation_data=val_ds,
    callbacks=callbacks_list,
    class_weight=class_weights
)

acc = history.history['accuracy'] + history_fine.history['accuracy']
val_acc = history.history['val_accuracy'] + history_fine.history['val_accuracy']

loss = history.history['loss'] + history_fine.history['loss']
val_loss = history.history['val_loss'] + history_fine.history['val_loss']


plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Verification accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()
model.save('/content/model.keras')
print("Model saved Successfully. . . . .")
