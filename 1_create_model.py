import face_recognition as fr
import cv2
import numpy as np
import os
import xml.etree.ElementTree as ET

known_names = []
known_name_encodings = []

# Check if XML file exists
def main():

    train_dir = "./friends/"
    for img_name in os.listdir(train_dir):
        #if len(img_name) != 14:
           # continue
        img_path = os.path.join(train_dir, img_name)
        img = fr.load_image_file(img_path)
        encoding = fr.face_encodings(img)[0]
        known_name_encodings.append(encoding)
        known_names.append(os.path.splitext(os.path.basename(img_path))[0].capitalize())

    # Save known faces to XML file
    root = ET.Element('known_faces')
    for name, encoding in zip(known_names, known_name_encodings):
        child = ET.SubElement(root, 'face')
        child.set('name', name)
        child.text = ' '.join(str(x) for x in encoding)
    tree = ET.ElementTree(root)
    tree.write('known_faces.xml')


main()
print("XML File created")
