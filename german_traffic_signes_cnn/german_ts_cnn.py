import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pickle
import pandas as pd
import cv2
import requests
from PIL import Image


np.random.seed(0)


def main():
    num_classes = 43
    train_data, val_data, test_data = unpickle()
    X_train, y_train, X_val, y_val, X_test, y_test = check_data(train_data, val_data, test_data)
    data = pd.read_csv('german-traffic-signs/signnames.csv')
    print(data)
    num_of_each_sign = show_training_samples(num_classes, data, X_train, y_train)
    plot_sample_distribution(num_classes, num_of_each_sign)
    examine_typical_image(X_train, y_train)
    X_train, X_val, X_test = apply_preprocessing(X_train, X_val, X_test)
    examine_random_image_after_preprocessing(X_train)
    X_train, X_val, X_test = reshape_for_cnn(X_train, X_val, X_test)
    y_train, y_val, y_test = one_hot_encode(num_classes, y_train, y_val, y_test)
    model = modified_model(num_classes)
    datagen = create_data_generator()
    evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test, datagen)
    url_30 = "https://c8.alamy.com/comp/G667W0/road-sign-speed-limit-30-kmh-zone-passau-bavaria-germany-G667W0.jpg"
    test_model_with_images(model, url_30)
    url_turn_left = "https://c8.alamy.com/comp/A0RX23/cars-and-automobiles-must-turn-left-ahead-sign-A0RX23.jpg"
    test_model_with_images(model, url_turn_left)
    url_slippery_road = "https://previews.123rf.com/images/bwylezich/bwylezich1608/bwylezich160800375/64914157-german-road-sign-slippery-road.jpg"
    test_model_with_images(model, url_slippery_road)
    url_yield = "https://previews.123rf.com/images/pejo/pejo0907/pejo090700003/5155701-german-traffic-sign-no-205-give-way.jpg"
    test_model_with_images(model, url_yield)
    url_bicycle = "https://c8.alamy.com/comp/J2MRAJ/german-road-sign-bicycles-crossing-J2MRAJ.jpg"
    test_model_with_images(model, url_bicycle)
    explore_datagen(datagen, X_train, y_train)
    
    
def explore_datagen(datagen, X_train, y_train):
    datagen.fit(X_train)
    batches = datagen.flow(X_train, y_train, batch_size=20)
    X_batch, y_batch = next(batches)
    fig, axs = plt.subplots(1, 15, figsize=(20,5))
    fig.tight_layout()
    for i in range(15):
        axs[i].imshow(X_batch[i].reshape(32, 32))
        axs[i].axis('off')
    plt.show()
    
def create_data_generator():
    datagen = ImageDataGenerator(width_shift_range = 0.1, height_shift_range=0.1, zoom_range=0.2, shear_range=0.1, rotation_range=10)
    return datagen
    
def test_model_with_images(model, url):
    r = requests.get(url, stream = True)
    img = Image.open(r.raw)
    # Preprocess the image to fit with the model
    img = np.asarray(img)
    img = cv2.resize(img, (32, 32))
    img = preprocessing(img)
    img = img.reshape(1, 32, 32, 1)
    print("Predicted sign: " + str(np.argmax(model.predict(img), axis=1)))
  
    
def evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test, datagen):
    print(model.summary())
    datagen.fit(X_train)
    history = model.fit(datagen.flow(X_train, y_train, batch_size = 50),  steps_per_epoch=int(X_train.shape[0]/50), epochs=10, validation_data=(X_val, y_val),verbose=1, shuffle=1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.legend(['training', 'validation'])
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.show()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(['training', 'validation'])
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.show()
    score = model.evaluate(X_test, y_test, verbose=0)
    print("Test score: ", score[0])
    print("Test accuracy: ", score[1])
    
    
def leNet_model(num_classes):
    model = Sequential()
    model.add(Conv2D(60, (5, 5), input_shape=(32, 32, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(30, (3, 3), input_shape=(32, 32, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model
    
def modified_model(num_classes):
    model = Sequential()
    model.add(Conv2D(60, (5, 5), input_shape=(32, 32, 1), activation='relu'))
    model.add(Conv2D(60, (5, 5), input_shape=(32, 32, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(30, (3, 3), input_shape=(32, 32, 1), activation='relu'))
    model.add(Conv2D(30, (3, 3), input_shape=(32, 32, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

 
def one_hot_encode(num_classes, y_train, y_val, y_test):
    y_train = to_categorical(y_train, num_classes)
    y_val = to_categorical(y_val, num_classes)
    y_test = to_categorical(y_test, num_classes)
    return y_train, y_val, y_test
    
def reshape_for_cnn(X_train, X_val, X_test):
    X_train = X_train.reshape(34799, 32, 32, 1)
    X_val = X_val.reshape(4410, 32, 32, 1)
    X_test = X_test.reshape(12630, 32, 32, 1)
    return X_train, X_val, X_test
  
def apply_preprocessing(X_train, X_val, X_test):
    X_train = np.array(list(map(preprocessing, X_train)))
    X_val = np.array(list(map(preprocessing, X_val)))
    X_test = np.array(list(map(preprocessing, X_test)))
    return X_train, X_val, X_test

def examine_random_image_after_preprocessing(X_train):
    plt.imshow(X_train[np.random.randint(0, len(X_train)-1)])
    plt.axis("off")
    plt.show()
    print(X_train.shape)
   
def preprocessing(img):
    img = grayscale(img) 
    img =  equalize(img)
    img = img/255
    return img

 
def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def equalize(img):
    # equalizeHist only works with grayscale images
    img = cv2.equalizeHist(img)
    return img
    

 

   
def examine_typical_image(X_train, y_train):
    pre_img = grayscale(X_train[1000])
    plt.imshow(pre_img)
    plt.show()
    img = preprocessing(X_train[1000])
    plt.imshow(img)
    plt.axis("off")
    plt.show()
    print(img.shape)
    print(y_train[1000])
    
    
def plot_sample_distribution(num_classes, num_of_each_sign):
    plt.figure(figsize=(12, 4))
    plt.bar(range(0, num_classes), num_of_each_sign)
    plt.title("Distribution of the training set")
    plt.xlabel("Class number")
    plt.ylabel("Number of images")
    plt.show()
    

def show_training_samples(num_classes, data, X_train, y_train):
    num_of_samples = []
    cols = 5
    
    fig, axs = plt.subplots(nrows=num_classes, ncols = cols, figsize=(5, 50))
    fig.tight_layout()
    for i in range(cols):
        for j, row in data.iterrows():
            x_selected = X_train[y_train==j]
            axs[j][i].imshow(x_selected[np.random.randint(0, len(x_selected) -1), :, :], cmap = plt.get_cmap('grey'))
            axs[j][i].axis("off")
            if i == 2:
                num_of_samples.append(len(x_selected))
                #axs[j][i].set_title(str(j) + "-" + row["SignName"])
    plt.show()
    return num_of_samples
        
    
    
    
def unpickle():
    with open('german-traffic-signs/train.p', 'rb') as f:
        train_data = pickle.load(f)
    with open('german-traffic-signs/valid.p', 'rb') as f:
        val_data = pickle.load(f)
    with open('german-traffic-signs/test.p', 'rb') as f:
        test_data = pickle.load(f)
    return train_data, val_data, test_data


def check_data(train_data, val_data, test_data):
    X_train, y_train = train_data['features'], train_data['labels']
    X_val, y_val = val_data['features'], val_data['labels']
    X_test, y_test = test_data['features'], test_data['labels']
    print(X_train.shape)
    print(X_val.shape)
    print(X_test.shape)
    assert(X_train.shape[0] == y_train.shape[0]), "The number of training images is not equal to the number of labels"
    assert(X_val.shape[0] == y_val.shape[0]), "The number of validation images is not equal to the number of labels"
    assert(X_test.shape[0] == y_test.shape[0]), "The number of test images is not equal to the number of labels"
    assert(X_train.shape[1:] == (32, 32, 3)), "The dimensions of the training images are not 32 x 32 x 3"
    assert(X_val.shape[1:] == (32, 32, 3)), "The dimensions of the validation images are not 32 x 32 x 3"
    assert(X_test.shape[1:] == (32, 32, 3)), "The dimensions of the test images are not 32 x 32 x 3"
    return X_train, y_train, X_val, y_val, X_test, y_test
    
    
    
    
if __name__ == "__main__":
    main()

