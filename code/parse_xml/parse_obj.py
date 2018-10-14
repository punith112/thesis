import xml.etree.ElementTree as ET
import os
import csv

def get_obj_pose(filename):
    tree = ET.parse(filename)
    root = tree.getroot()

    main_dict = {}

    for elem in root.iter():
        if elem.tag == "name":
            obj_name = elem.text
            main_dict[obj_name] = {}
            # print(elem.tag, elem.text)
        if elem.tag == "pose":
            for child in elem:
                main_dict[obj_name][child.tag] = child.text
    return main_dict


directory = os.fsencode("../../kth-3d-total/xml-annotated")

main_list = []

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    filename = "../../kth-3d-total/xml-annotated/" + filename
    # print(filename)
    main_list.append(get_obj_pose(os.path.abspath(filename)))


with open('mouse_features.csv', mode = 'w') as mouse_file:
    csv_writer = csv.writer(mouse_file, delimiter=',')

    csv_writer.writerow(['x', 'y', 'z'])

    for i in range(len(main_list)):
        try:
            csv_writer.writerow([main_list[i]['Mouse']['x'], main_list[i]['Mouse']['y'], main_list[i]['Mouse']['z']])
        except KeyError:
            continue
