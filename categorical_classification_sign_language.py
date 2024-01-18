import cv2
from functions import *

TRAINING_FILE = './sign_mnist_train.csv'
VALIDATION_FILE = './sign_mnist_test.csv'

# Show what the CSV setup looks like
with open(TRAINING_FILE) as training_file:
    line = training_file.readline()
    print(f"First line (header) looks like this:\n{line}")
    line = training_file.readline()
    print(f"Each subsequent line (data points) look like this:\n{line}")

# Generate random sample of training image data and all of validation data
training_images, training_labels = parse_data_from_input(TRAINING_FILE, 30000)
validation_images, validation_labels = parse_data_from_input(VALIDATION_FILE)

# Resize images for more clear HTML webcam purposes
target_size = (100, 100)
training_images = np.array([cv2.resize(image, target_size) for image in training_images])
validation_images = np.array([cv2.resize(image, target_size) for image in validation_images])
print(f"Training images has shape: {training_images.shape} and dtype: {training_images.dtype}")
print(f"Training labels has shape: {training_labels.shape} and dtype: {training_labels.dtype}")
print(f"Validation images has shape: {validation_images.shape} and dtype: {validation_images.dtype}")
print(f"Validation labels has shape: {validation_labels.shape} and dtype: {validation_labels.dtype}")

# Plot 10 example sign language letters
plot_categories(training_images, training_labels)

# Set up images and labels for NN training
train_generator, validation_generator = train_val_generators(training_images, training_labels, validation_images, validation_labels)

print(f"Images of training generator have shape: {train_generator.x.shape}")
print(f"Labels of training generator have shape: {train_generator.y.shape}")
print(f"Images of validation generator have shape: {validation_generator.x.shape}")
print(f"Labels of validation generator have shape: {validation_generator.y.shape}")

# Create model
model = create_model(target_size)
model.summary()

# Train Model
history = model.fit(train_generator,
                    epochs=15,
                    validation_data=validation_generator)

# Plot performance metrics
plot_losses_and_accuracies(history)

model.save('./sign_language_model.h5')
