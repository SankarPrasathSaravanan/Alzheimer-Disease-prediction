wimport numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from keras.models import load_model

# Load the model from the file



model = load_model('Alzheimer_Trained_Model2.h5')  # Make sure to use .h5 for Keras models

# Define the model architecture (if not loading from file)
# model = Sequential()
# model.add(Conv2D(32, kernel_size=(2, 2), input_shape=(128, 128, 3), padding='Same'))
# model.add(Conv2D(32, kernel_size=(2, 2), activation='relu', padding='Same'))
# model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
# model.add(Conv2D(64, kernel_size=(2, 2), activation='relu', padding='Same'))
# model.add(Conv2D(64, kernel_size=(2, 2), activation='relu', padding='Same'))
# model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
# model.add(Dropout(0.25))
# model.add(Flatten())
# model.add(Dense(512, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(4, activation='softmax'))
# model.compile(loss='categorical_crossentropy', optimizer='Adamax', metrics=['accuracy'])

# print(model.summary())

# Plotting training history (if available)
# history = joblib.load('Alzheimer_Trained_Model_History.pkl')  # If you saved the history separately
# plt.plot(history['loss'])
# plt.plot(history['val_loss'])
# plt.title('Model Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend(['Training Loss', 'Validation Loss'], loc='upper right')
# plt.show()

def names(number):
    if number == 0:
        return 'Non Demented'
    elif number == 1:
        return 'Mild Dementia'
    elif number == 2:
        return 'Moderate Dementia'
    elif number == 3:
        return 'Very Mild Dementia'
    else:
        return 'Error in Prediction'

# Prediction
def predict_image(image_path):
    img = Image.open(image_path)
    x = np.array(img.resize((128, 128)))
    x = x.reshape(1, 128, 128, 3)
    res = model.predict_on_batch(x)
    classification = np.argmax(res, axis=1)[0]
    print(f'{res[0][classification] * 100:.2f}% Confidence This Is {names(classification)}')
    plt.imshow(img)
    plt.show()

# Test with different images
predict_image('./Data/Moderate Dementia/OAS1_0308_MR1_mpr-1_108.jpg')
# predict_image('./Data/Very mild Dementia/OAS1_0003_MR1_mpr-1_117.jpg')
# predict_image('./Data/Mild Dementia/OAS1_0028_MR1_mpr-1_145.jpg')
# predict_image('./test1.jpg')
predict_image('./test3.jpeg')
# predict_image('./test4.jpg')
predict_image('./test5.jpg')




# To activate the virtual environment
# .\venv\Scripts\activate

# to deactivate the virtual environment
# deactivate
