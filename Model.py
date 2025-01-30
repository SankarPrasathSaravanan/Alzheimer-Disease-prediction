import numpy as np 
import os
import keras
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from keras.models import Sequential
from PIL import Image
from keras.layers import Conv2D,Flatten,Dense,Dropout,BatchNormalization,MaxPooling2D
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from joblib import Parallel, delayed 
import joblib 



path1 = []
path2 = []
path3 = []
path4 = []
for dirname, _, filenames in os.walk('./Data/Non Demented'):
    for filename in filenames:
        path1.append(os.path.join(dirname, filename))
        
for dirname, _, filenames in os.walk('./Data/Mild Dementia'):
    for filename in filenames:
        path2.append(os.path.join(dirname, filename))
        
for dirname, _, filenames in os.walk('./Data/Moderate Dementia'):
    for filename in filenames:
        path3.append(os.path.join(dirname, filename))
        
for dirname, _, filenames in os.walk('./Data/Very mild Dementia'):
    for filename in filenames:
        path4.append(os.path.join(dirname, filename)) 



path1 = path1[0:100]
path2 = path2[0:100]
path3 = path3[0:100]
path4 = path4[0:100]

encoder = OneHotEncoder()
encoder.fit([[0],[1],[2],[3]])

data = []
result = []
for path in path1:
    img = Image.open(path)
    img = img.resize((128,128))
    img = np.array(img)
    if(img.shape == (128,128,3)):
        data.append(np.array(img))
        result.append(encoder.transform([[0]]).toarray())
        
for path in path2:
    img = Image.open(path)
    img = img.resize((128,128))
    img = np.array(img)
    if(img.shape == (128,128,3)):
        data.append(np.array(img))
        result.append(encoder.transform([[1]]).toarray()) 
        
for path in path3:
    img = Image.open(path)
    img = img.resize((128,128))
    img = np.array(img)
    if(img.shape == (128,128,3)):
        data.append(np.array(img))
        result.append(encoder.transform([[2]]).toarray())
        
for path in path4:
    img = Image.open(path)
    img = img.resize((128,128))
    img = np.array(img)
    if(img.shape == (128,128,3)):
        data.append(np.array(img))
        result.append(encoder.transform([[3]]).toarray())

data = np.array(data)
data.shape

result = np.array(result)
result = result.reshape((400,4))
print(result.shape)


# splitting The Data


x_train,x_test,y_train,y_test = train_test_split(data,result,test_size=30,shuffle=True,random_state=82)

model = Sequential()

model.add(Conv2D(32,kernel_size =(2,2),input_shape = (128,128,3),padding = 'Same'))
model.add(Conv2D(32,kernel_size =(2,2),activation='relu',padding = 'Same'))

model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64,kernel_size =(2,2),activation='relu',padding = 'Same'))
model.add(Conv2D(64,kernel_size =(2,2),activation='relu',padding = 'Same'))

model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2),strides = (2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
          
model.add(Dense(512,activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(4,activation='softmax'))
          
model.compile(loss = 'categorical_crossentropy',optimizer = 'Adamax',metrics=['accuracy'])
          
print(model.summary())

print(y_train.shape)
print(x_train.shape)

history = model.fit(x_train,y_train,epochs=4,batch_size=32,verbose=1,validation_data=(x_test,y_test))


# Save the model
model.save('test.h5')

# Save history as a pickle file
joblib.dump(history.history, 'Alzheimer_Training_History.pkl')


# Visualizing Training History (Loss & Accuracy)
def plot_training_history(history):
    epochs = range(1, len(history.history['loss']) + 1)
    
    # Bar chart for Loss
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.bar(epochs, history.history['loss'], width=0.4, label='Training Loss', align='center')
    plt.bar(epochs, history.history['val_loss'], width=0.4, label='Validation Loss', align='edge')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()

    # Bar chart for Accuracy
    plt.subplot(1, 2, 2)
    plt.bar(epochs, history.history['accuracy'], width=0.4, label='Training Accuracy', align='center')
    plt.bar(epochs, history.history['val_accuracy'], width=0.4, label='Validation Accuracy', align='edge')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training vs Validation Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Call the function to plot training history
plot_training_history(history)

# Save the model as a pickle in a file 
# joblib.dump(history, 'Alzheimer_Trained_Model.pkl')
  
# Load the model from the file 
# knn_from_joblib = joblib.load('filename.pkl') 
  
# # Use the loaded model to make predictions 
# knn_from_joblib.predict(X_test) 



# Make predictions on the test set
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

# Calculate confusion matrix
cm = confusion_matrix(y_test_classes, y_pred_classes)

# Calculate sensitivity, specificity, and accuracy
def calculate_metrics(cm):
    TP = np.diag(cm)
    FP = np.sum(cm, axis=0) - TP
    FN = np.sum(cm, axis=1) - TP
    TN = np.sum(cm) - (TP + FP + FN)

    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    accuracy = (TP + TN) / (TP + TN + FP + FN)

    return sensitivity, specificity, accuracy

sensitivity, specificity, accuracy = calculate_metrics(cm)

# Print classification report
print(classification_report(y_test_classes, y_pred_classes, target_names=['Non Demented', 'Mild Dementia', 'Moderate Dementia', 'Very Mild Dementia']))

# Function to plot the metrics as a bar chart
def plot_metrics(sensitivity, specificity, accuracy):
    labels = ['Sensitivity', 'Specificity', 'Accuracy']
    values = [np.mean(sensitivity), np.mean(specificity), np.mean(accuracy)]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, values, color=['#4CAF50', '#FFC107', '#2196F3'])
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.2f}', ha='center', va='bottom')

    plt.xlabel('Metrics')
    plt.ylabel('Values')
    plt.title('Performance Metrics')
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

# Plot the metrics
plot_metrics(sensitivity, specificity, accuracy)