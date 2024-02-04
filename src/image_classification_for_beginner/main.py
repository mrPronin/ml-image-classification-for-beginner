import pandas as pd
import os
import matplotlib.pyplot as plt


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


if __name__ == "__main__":
    main()
