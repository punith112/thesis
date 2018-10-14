import xml.etree.ElementTree as ET
import os
import csv

def get_obj_pose(filename):
    tree = ET.parse(filename)
    root = tree.getroot()

    main_dict = {}

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

    return main_dict


directory = os.fsencode("../../kth-3d-total/xml-annotated")

main_list = []

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    filename = "../../kth-3d-total/xml-annotated/" + filename
    # print(filename)
    main_list.append(get_obj_pose(os.path.abspath(filename)))

# main_list.append(get_obj_pose('Akshaya_Table_131023_Aft_seg.xml'))


with open('mouse_features.csv', mode = 'w') as mouse_file:
    csv_writer = csv.writer(mouse_file, delimiter=',')

    csv_writer.writerow(['x', 'y', 'z'])

    for i in range(len(main_list)):
        try:
            csv_writer.writerow([round(float(main_list[i]['Mouse']['x']) + float(main_list[i]['Mouse']['length'])/2, 2),
                                round(float(main_list[i]['Mouse']['y']) + float(main_list[i]['Mouse']['width'])/2, 2),
                                round(float(main_list[i]['Mouse']['z']) + float(main_list[i]['Mouse']['height'])/2, 2)])
        except KeyError:
            continue
