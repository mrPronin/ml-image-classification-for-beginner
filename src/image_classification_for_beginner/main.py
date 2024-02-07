import os
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
from keras.applications.vgg16 import VGG16
from keras.models import Sequential
from keras.layers import Input, Flatten, Dropout, Dense
import numpy as np
# import tensorflow


def main():
    img_path = "data/Rice_Image_Dataset/"

    rice_label = ['Arborio', 'Basmati', 'Ipsala', 'Jasmine', 'Karacadag']
    # create a dataframe of image path and label
    img_list = []
    label_list = []
    for label in rice_label:
        for img_file in os.listdir(img_path+label):
            img_list.append(img_path+label+'/'+img_file)
            label_list.append(label)
    df = pd.DataFrame({'img': img_list, 'label': label_list})

    # count the number of images of each rice category
    print(df['label'].value_counts())

    # show sample images
    fig, ax = plt.subplots(ncols=len(rice_label), figsize=(20, 4))
    fig.suptitle('Rice Category')
    random_num = 12
    for i, label in enumerate(rice_label):
        path = df[df['label'] == label]['img'].iloc[random_num]
        # print(f"i: {i} label: {label} path: {path}")
        ax[i].set_title(label)
        ax[i].imshow(plt.imread(path))

    # plt.show()

    # know image shape
    print(f"Image shape: {plt.imread(df['img'][0]).shape}")

    # Create a dataframe for mapping label
    df_labels = {
        'Arborio': 0,
        'Basmati': 1,
        'Ipsala': 2,
        'Jasmine': 3,
        'Karacadag': 4
    }
    # Encode
    df['encode_label'] = df['label'].map(df_labels)
    print(df.head())

    # Prepare a model training dataset
    X = []
    for img in df['img']:
        img = cv2.imread(str(img))
        # img = augment_function(img)
        img = cv2.resize(img, (96, 96))
        img = img/255
        X.append(img)

    y = df['encode_label']

    # Convert X and y into numpy arrays
    X = np.array(X)
    y = np.array(y)

    # Ensure X is reshaped into (n_samples, height, width, channels)
    # This step is crucial if your images have not been reshaped yet.
    # In your case, it seems like each image is already resized to (96, 96)
    #  and has 3 channels.
    # So, this step may already be done. Just ensure X's shape is correct.

    # Train/Validation/Test split
    # X_train, X_test_val, y_train, y_test_val = train_test_split(X, y)
    
    # Adjust test_size as needed
    X_train, X_test_val, y_train, y_test_val = train_test_split(X, y,
                                                                test_size=0.2)

    # Further split for test and validation
    X_test, X_val, y_test, y_val = train_test_split(X_test_val, y_test_val,
                                                    test_size=0.5)
    # X_test, X_val, y_test, y_val = train_test_split(X_test_val, y_test_val)

    # Use VGG16 as a base model
    base_model = VGG16(input_shape=(96, 96, 3), include_top=False,
                       weights='imagenet')

    print("Base model summary:")
    print(base_model.summary())

    # freeze the VGG16 model parameters
    for layer in base_model.layers:
        layer.trainable = False
    base_model.layers[-2].trainable = True
    base_model.layers[-3].trainable = True
    base_model.layers[-4].trainable = True

    # add layers to the model
    model = Sequential()
    model.add(Input(shape=(96, 96, 3)))
    model.add(base_model)
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(len(rice_label), activation='softmax'))

    print("Model summary:")
    print(model.summary())

    # train a model
    model.compile(optimizer="adam", loss='sparse_categorical_crossentropy',
                  metrics=['acc'])
    history = model.fit(X_train, y_train, epochs=5,
                        validation_data=(X_val, y_val))

    print(history)

    # model evaluation
    model.evaluate(X_test, y_test)

    # store trained model
    model_filename = './trained-model.h5'
    model.save(model_filename)

    # load model
    # tensorflow.keras.models.load_model(model_filename)

    # visualize the model

    # Plot accuracy of each epoch
    plt.plot(history.history['acc'], marker='o')
    plt.plot(history.history['val_acc'], marker='o')
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='lower right')
    plt.show()

    # Plot loss of each epoch
    plt.plot(history.history['loss'], marker='o')
    plt.plot(history.history['val_loss'], marker='o')
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper right')
    plt.show()


if __name__ == "__main__":
    main()
