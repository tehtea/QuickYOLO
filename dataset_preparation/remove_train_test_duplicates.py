from glob import glob
import os
import shutil

def remove_train_test_duplicates():
    train_files = glob("voc_data/train/*.xml")
    test_files = glob("voc_data/test/*.xml")
    duplicate_files = [(train_file, test_file)  \
        for train_file in train_files \
        for test_file in test_files \
        if '2007_' in train_file \
            and os.path.basename(train_file).split('2007_')[1] \
            == os.path.basename(test_file)]
    for duplicate_file in duplicate_files:
        duplicate_annotation = duplicate_file[1]
        duplicate_image = os.path.splitext(duplicate_file[1])[0] + '.jpg'
        os.remove(duplicate_annotation)
        os.remove(duplicate_image)
    print(f'{len(duplicate_files)} files removed from test set!')

if __name__ == '__main__':
    remove_train_test_duplicates()