import os
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
from keras.applications.vgg16 import VGG16


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

    # Train/Validation/Test split
    X_train, X_test_val, y_train, y_test_val = train_test_split(X, y)
    X_test, X_val, y_test, y_val = train_test_split(X_test_val, y_test_val)

    # Use VGG16 as a base model
    base_model = VGG16(input_shape=(96, 96, 3), include_top=False,
                       weights='imagenet')

    print(base_model.summary())

    for layer in base_model.layers:
        layer.trainable = False
    base_model.layers[-2].trainable = True
    base_model.layers[-3].trainable = True
    base_model.layers[-4].trainable = True


if __name__ == "__main__":
    main()
