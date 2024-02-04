import pandas as pd
import os


def main():
    img_path = "data/Rice_Image_Dataset/"
    rice_label = ['Arborio', 'Basmati', 'Ipsala', 'Jasmine', 'Karacadag']
    img_list = []
    label_list = []
    for label in rice_label:
        for img_file in os.listdir(img_path+label):
            img_list.append(img_path+label+'/'+img_file)
            label_list.append(label)
    df = pd.DataFrame({'img': img_list, 'label': label_list})
    print(df['label'].value_counts())


if __name__ == "__main__":
    main()
