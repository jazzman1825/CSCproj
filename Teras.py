import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow import keras

# Define directories for images and masks
images_dir = "TrainImages"
masks_dir = "TrainMasks"

# List files in both directories
image_files = [file for file in os.listdir(images_dir) if file.endswith('.jpg')]
mask_files = [file for file in os.listdir(masks_dir) if file.endswith('.jpg')]

# Ensure the lists are sorted for proper matching
image_files.sort()
mask_files.sort()

# Function to load a batch of data
def load_data(image_files, mask_files, batch_size):
    num_samples = len(image_files)
    while True:
        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            batch_image_files = image_files[start_idx:end_idx]
            batch_mask_files = mask_files[start_idx:end_idx]
            
            # Initialize lists to store images and masks
            images = []
            masks = []
            
            # Load images and masks for the current batch
            for image_file, mask_file in zip(batch_image_files, batch_mask_files):
                image_path = os.path.join(images_dir, image_file)
                mask_path = os.path.join(masks_dir, mask_file)

                # Load images
                image = plt.imread(image_path)
                images.append(image)

                # Load masks
                mask = plt.imread(mask_path)
                masks.append(mask)

            # Convert lists to numpy arrays
            images = np.array(images)
            masks = np.array(masks)

            # Scaling
            images = images / 255 
            masks = masks / 255

            yield images, masks

batch_size = 32
train_data_generator = load_data(image_files, mask_files, batch_size)

#CONFIG
image_shape = plt.imread(os.path.join(images_dir, image_files[0])).shape
IMG_HEIGHT = image_shape[0]
IMG_WIDTH = image_shape[1]
IMG_CHANNELS = image_shape[2]

train_ds = []
val_ds = []
for i in range(len(image_files)):
    if np.random.rand() < 0.2:
        train_ds.append(i)
    else:
        val_ds.append(i)

traind_ds = np.array(train_ds)
val_ds = np.array(val_ds)

# Model def here 
def make_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)

    # Contracting Path
    x = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1_residue = x
    x = keras.layers.MaxPooling2D((2, 2))(x)
    
    x = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    c2_residue = x
    x = keras.layers.MaxPooling2D((2, 2))(x)

    x = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    c3_residue = x
    x = keras.layers.MaxPooling2D((2, 2))(x)

    x = keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    c4_residue = x
    x = keras.layers.MaxPooling2D((2, 2))(x)

    x = keras.layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(x)
    
    # Expansive Path    
    x = keras.layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(x)
    x = keras.layers.Concatenate()([x, c4_residue])
    x = keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    
    x = keras.layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(x)
    x = keras.layers.Concatenate()([x, c3_residue])
    x = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
   
    x = keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(x)
    x = keras.layers.Concatenate()([x, c2_residue])
    x = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
   
    x = keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(x)
    x = keras.layers.Concatenate()([x, c1_residue])
    x = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)


    outputs = keras.layers.Conv2D(num_classes, (1, 1), activation="sigmoid")(x)
    return keras.Model(inputs, outputs)

model = make_model(input_shape=image_shape, num_classes=1)

#Model Config
epochs = 50
#Saving as *.weights.keras according to Keras3 recommendation
#https://github.com/keras-team/keras-io/issues/1568 - outdated?
callbacks = [
    keras.callbacks.ModelCheckpoint("teras.weights.keras", monitor='val_loss', verbose = 1, save_best_only = True, mode = 'min'),
    keras.callbacks.EarlyStopping(patience = 30, verbose = 1),
    ]
model.compile(
    optimizer=keras.optimizers.Adam(1e-4),
    loss = "binary_crossentropy",
    metrics = ["accuracy"],
)
history = model.fit(
    train_data_generator,
    steps_per_epoch=len(image_files) // batch_size,
    epochs=epochs,
    callbacks=callbacks,
    validation_data=train_data_generator,  # Adjusted to use the same generator for validation
    validation_steps=len(val_ds) // batch_size,
)

history_dict = history.history
loss_values = history_dict["loss"]
val_loss_values = history_dict["val_loss"]
epochs = range(1, len(loss_values) + 1)
plt.plot(epochs, loss_values, "bo", label="Training loss")
plt.plot(epochs, val_loss_values, "b", label="Validation loss")
plt.title("Training and validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig("training_validation_loss.png")  # Save figure as image

plt.clf() #matplotlib figure clearing
acc = history_dict["accuracy"]
val_acc = history_dict["val_accuracy"]
plt.plot(epochs, acc, "bo", label = "Training acc")
plt.plot(epochs, val_acc, "b", label = "Validation acc")
plt.title("Training and validation accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("training_validation_accuracy.png")  # Save figure as image