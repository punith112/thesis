import xml.etree.ElementTree as ET

# tree = ET.parse('Akshaya_Table_131023_Aft_seg.xml')
tree = ET.parse('sample_xml.xml')
root = tree.getroot()

# for child in root:
#     print(child.tag, child.attrib)

for movie in root.iter('movie'):
    print(movie.attrib)
