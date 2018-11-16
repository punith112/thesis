import xml.etree.ElementTree as ET
import os
import csv
import pickle

with open("extracted_data.txt", "rb") as myFile:
    main_list = pickle.load(myFile)

with open('mouse_features.csv', mode = 'w') as mouse_file:
    csv_writer = csv.writer(mouse_file, delimiter=',')

    csv_writer.writerow(['x', 'y', 'z'])

    for i in range(len(main_list)):

        try:
            if abs(2.3002855e+35 - round(float(main_list[i]['Mouse']['z']) + float(main_list[i]['Mouse']['height'])/2, 2)) < 1:
                print("Yes")
                print(round(float(main_list[i]['Mouse']['z']) + float(main_list[i]['Mouse']['height'])/2, 2))
                print(i)
                print(float(main_list[i]['Mouse']['z']), float(main_list[i]['Mouse']['height']))
            csv_writer.writerow([main_list[i]['file'],
                                round(float(main_list[i]['Mouse']['x']) + float(main_list[i]['Mouse']['length'])/2, 2),
                                round(float(main_list[i]['Mouse']['y']) + float(main_list[i]['Mouse']['width'])/2, 2),
                                round(float(main_list[i]['Mouse']['z']) + float(main_list[i]['Mouse']['height'])/2, 2)])

        except KeyError:
            continue
