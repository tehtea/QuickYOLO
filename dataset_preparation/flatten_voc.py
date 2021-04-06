import glob
import os
import shutil

def flatten_voc():
    os.chdir('VOCdevkit')
    for original_folder, dest_folder in [('VOC2012', 'train'), ('VOC2007', 'test')]:
        os.mkdir(dest_folder)
        dest_file_path = os.path.abspath(dest_folder)
        for subfolder in ['JPEGImages', 'Annotations']:
            nested_file_path = os.path.join(original_folder, subfolder)
            files = os.listdir(nested_file_path)
            for file in files:
                file_path = os.path.abspath(os.path.join(nested_file_path, file))
                shutil.move(file_path, dest_file_path)

if __name__ == '__main__':
    flatten_voc()