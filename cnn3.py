# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from sklearn.model_selection import KFold

# Initialising the CNN
model = Sequential()

# Step 1 - Convolution
# number of filters
# shape each filter 3x3
# input shape and type of image (64x64 res & 3=RGB)
# relu = rectifier function
model.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling
model.add(MaxPooling2D(pool_size = (2, 2)))
# Adding a second convolutional layer
model.add(Conv2D(32, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
# Adding a third convolutional layer
model.add(Conv2D(32, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
# Adding a fourth convolutional layer
model.add(Conv2D(32, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
model.add(Flatten())

# Step 4 - Full connection
# hidden layer
# units = # of nodes in hidden layer
# relu = rectifier function
model.add(Dense(units = 128, activation = 'relu'))
# binary output of x or y == sigmoid
model.add(Dense(units = 1, activation = 'sigmoid'))

# compile our cnn
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images
# train data
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
shear_range = 0.2,
zoom_range = 0.2,
horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)
training_set = train_datagen.flow_from_directory('CNN_Data/training_set',
target_size = (64, 64),
batch_size = 32,
class_mode = 'binary')
test_set = test_datagen.flow_from_directory('CNN_Data/test_set',
target_size = (64, 64),
batch_size = 32,
class_mode = 'binary')

# fit data into model
# steps_per_epoch = # of training images
# epochs = single step in training a neural network
history = model.fit_generator(training_set,
    steps_per_epoch = 1000,
    epochs = 10,
    validation_data = training_set,
    validation_steps = 250)

# give a summary of model
model.summary()

import matplotlib.pyplot as plt

history_dict = history.history
history_dict.keys()

print(history_dict)
acc = history_dict['acc']
val_acc = history_dict['val_acc']
loss = history_dict['loss']
val_loss = history_dict['val_loss']
epochs = range(1, len(acc) + 1)

plt.clf()
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

# Save entire model to a HDF5 file
model.save('model3.h5')
