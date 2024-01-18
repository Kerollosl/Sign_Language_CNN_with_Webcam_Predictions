import string
import matplotlib.pyplot as plt
import random
import csv
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense


def parse_data_from_input(filename, random_samples=None):

  with open(filename) as file:

    csv_reader = csv.reader(file, delimiter=',')
    first_line = True
    images = []
    labels = []

    for row in csv_reader:
        if first_line:
           first_line = False
        else:
          labels.append(row[0])
          image_row = row[1:]
          image_data_as_array = np.array_split(image_row, 28)
          images.append(image_data_as_array)
    images = np.array(images).astype('float64')
    labels = np.array(labels).astype('float64')

    # Shuffle data
    combined_data = list(zip(images, labels))
    random.shuffle(combined_data)

    # Random subset of n elements
    if random_samples:
        random_subset = combined_data[:random_samples]
    else:
        random_subset = combined_data

    # Unzip shuffled subset
    shuffled_images, shuffled_labels = zip(*random_subset)

  return np.array(shuffled_images), np.array(shuffled_labels)

def train_val_generators(training_images, training_labels, validation_images, validation_labels):

  training_images = np.expand_dims(training_images, axis=3)
  validation_images = np.expand_dims(validation_images, axis=3)

  train_datagen = ImageDataGenerator(rescale=1./255,
                                     rotation_range=40,
                                     width_shift_range=0.2,
                                     height_shift_range=0.2,
                                     shear_range=0.2,
                                     zoom_range=0.2,
                                     horizontal_flip=True,
                                     fill_mode='nearest')


  train_generator = train_datagen.flow(x=training_images,
                                       y=training_labels,
                                       batch_size=32)


  validation_datagen = ImageDataGenerator(rescale=1./255,
                                     rotation_range=40,
                                     width_shift_range=0.2,
                                     height_shift_range=0.2,
                                     shear_range=0.2,
                                     zoom_range=0.2,
                                     horizontal_flip=True,
                                     fill_mode='nearest')

  validation_generator = validation_datagen.flow(x=validation_images,
                                                 y=validation_labels,
                                                 batch_size=32)

  return train_generator, validation_generator

def plot_categories(training_images, training_labels):
  fig, axes = plt.subplots(1, 10, figsize=(16, 15))
  axes = axes.flatten()
  letters = list(string.ascii_lowercase)

  for k in range(10):
    img = training_images[k]
    img = np.expand_dims(img, axis=-1)
    img = array_to_img(img)
    ax = axes[k]
    ax.imshow(img, cmap="Greys_r")
    ax.set_title(f"{letters[int(training_labels[k])]}")
    ax.set_axis_off()

  plt.tight_layout()
  plt.show()

def create_model(target_size):
  model = Sequential([
                      Conv2D(32, (3, 3), activation='relu', input_shape=target_size + (1,)),
                      MaxPooling2D(2, 2),

                      Conv2D(64, (3, 3), activation='relu'),
                      MaxPooling2D(2, 2),

                      Flatten(),
                      Dense(512, activation='relu'),
                      Dropout(0.25),
                      Dense(26, activation='softmax')
                      ])

  model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])


  return model

def plot_losses_and_accuracies(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))

    plt.plot(epochs, acc, 'r', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()

    plt.plot(epochs, loss, 'r', label='Training Loss')
    plt.plot(epochs, val_loss, 'b', label='Validation Loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()