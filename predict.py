import numpy as np
from keras.models import load_model
from keras.preprocessing import image

# Recreate the exact same model, including weights and optimizer.
model = load_model('model1.h5')
model.summary()

test_image = image.load_img('Predictions/myDog3.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = model.predict(test_image)

if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'
print(prediction)