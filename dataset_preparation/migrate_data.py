import xml.etree.ElementTree as ET
import os
import glob
from shutil import move

def migrate_data():
    def rename_in_annotation(ann_dir):
        # Parse file
        for ann in sorted(os.listdir(ann_dir)):
            tree = ET.parse(ann_dir + ann)
            for elem in tree.iter(): 
                if 'filename' in elem.tag:
                    new_img_name = '2007_' + elem.text
                    elem.text = new_img_name
            tree.write(ann_dir + ann) # write back

    more_twenty_oseven_images = glob.glob(os.path.join('VOC2007Trainval/JPEGImages/*'))
    more_twenty_oseven_annotations = glob.glob(os.path.join('VOC2007Trainval/Annotations/*'))

    # rename the image paths in each annotation first
    rename_in_annotation('VOC2007Trainval/Annotations/')

    # perform the migration
    for old_image_path, old_annotation_path in zip(more_twenty_oseven_images, more_twenty_oseven_annotations):
        old_image_name = os.path.basename(old_image_path)
        old_annotation_name = os.path.basename(old_annotation_path)
        new_image_path = 'VOCdevkit/VOC2012/JPEGImages/2007_' + old_image_name
        new_annotation_path = 'VOCdevkit/VOC2012/Annotations/2007_' + old_annotation_name
        move(old_image_path, new_image_path)
        move(old_annotation_path, new_annotation_path)

if __name__ == '__main__':
    migrate_data()