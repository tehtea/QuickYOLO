from glob import glob
from pprint import pprint
import os

if __name__ == "__main__":
    train_files = glob("voc_data/train/*.xml")
    test_files = glob("voc_data/test/*.xml")
    test_files_based = list(map(os.path.basename, test_files))
    duplicate_files = [(train_file, test_file)  \
        for train_file in train_files \
        for test_file in test_files \
        if '2007_' in train_file \
            and os.path.basename(train_file).split('2007_')[1] \
            == os.path.basename(test_file)]
    print('see duplicate files:')
    pprint(duplicate_files)
    print('Number of files in train set: ', len(train_files))
    assert len(duplicate_files) == 0, f'Test set files found in train set. Number of duplicates: {len(duplicate_files)}'
    assert len(train_files) == 21667, f'Number of files in train set ({len(train_files)}) not the same as the known count'
    print('Train and test datasets are AOK!')
