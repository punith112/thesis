import xml.etree.ElementTree as ET
import os
import csv
import pickle

def get_obj_pose(filename):
    tree = ET.parse(filename)
    root = tree.getroot()

    main_dict = {}
    main_dict['file'] = filename[58:]

    for elem in root[3].iter():
        if elem.tag == "name":
            obj_name = elem.text
            main_dict[obj_name] = {}
            # print(elem.tag, elem.text)

        if elem.tag == "pose":
            for child in elem:
                main_dict[obj_name][child.tag] = child.text


        if elem.tag == "dimensions":
            # import ipdb; ipdb.set_trace()
            for child in elem:
                main_dict[obj_name][child.tag] = child.text
                if abs(4.600571e+35 - float(child.text)) < 1:
                    print(filename)
                    import ipdb; ipdb.set_trace()

    return main_dict


directory = os.fsencode("../../kth-3d-total/xml-annotated")

main_list = []

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    filename = "../../kth-3d-total/xml-annotated/" + filename
    # print(filename)
    main_list.append(get_obj_pose(os.path.abspath(filename)))

with open("extracted_data.txt", "wb") as myFile:
    pickle.dump(main_list, myFile)
