import xml.etree.ElementTree as ET
import os
import csv
import pickle
import pandas as pd

# Constants
INDEX_OF_OBJECTS_TAG_IN_XML_FILE = 3
FILE_NAME_END_TRIM_INDEX = -8

class DataExtractor:
    """
    A class implementation for extracting data from the xml files
    of the annotated PCL database
    """

    def __init__(self, data_path, attributes, dict_dump_file_name, df_dump_file_name):
    """
    Instantiation

    Parameters
    ----------
    data_path: String
    Path to the directory in which the xml database files are stored.

    attributes: List
    The attributes whose data needs to be extracted from the xml database.

    dict_dump_file_name: String
    String specifying the filepath of the file to which the parsed data of
    all the xml files will be written in the form of a dict

    df_dump_file_name: String
    String specifying the filepath of the file to which the extracted data for
    the specified attributes will be written in the form of a Pandas DataFrame.
    """

        self.data_path = data_path
        self.attributes = attributes
        self.dict_dump_file_name = dict_dump_file_name
        self.df_dump_file_name = df_dump_file_name

        self.scenes_list = []
        self.objects_in_scenes = []
        self.scenes_df = pd.DataFrame()

    def parse_xml(self, file_path, file_name):
    """
    Parses the xml file of an annotated point cloud and gets the
    name, pose and dimensions (all the useful data, basically)
    of all objects in each file.

    Parameters
    ----------
    file_path: String
    Path to the xml file to be parsed.

    file_name: String
    Name of the file (local path) so that it can be used
    as an identifier for the parsed data.

    Returns
    -------
    objects_dict: Dict
    Dictionary containing the object name as a key and a dict as its value.
    This dict has all attributes as keys and their corresponding values
    as umm, values.

    One dict per xml file.
    """

        tree = ET.parse(file_path)
        root = tree.getroot()

        objects_dict = {}
        objects_dict = {}
        objects_dict['file'] = file_name[:FILE_NAME_END_TRIM_INDEX]

        for elem in root[INDEX_OF_OBJECTS_TAG_IN_XML_FILE].iter():
            if elem.tag == "name":
                obj_name = elem.text
                objects_dict[obj_name] = {}
                # print(elem.tag, elem.text)

            if elem.tag == "pose":
                for child in elem:
                    objects_dict[obj_name][child.tag] = child.text


            if elem.tag == "dimensions":
                # import ipdb; ipdb.set_trace()
                for child in elem:
                    objects_dict[obj_name][child.tag] = child.text

        return objects_dict

    def generate_pd_series(self, object_name, attribute, scenes_list):
    """
    Generates a Pandas Series for the given object and attribute
    from all the scenes.

    Parameters
    ----------
    object_name: String
    Name of the object for which data is to be collected.

    attribute: String
    Attribute for which data is to be collected.

    Returns
    -------
    pd_series: Pandas Series
    Pandas Series with filenames as row labels and the attribute
    data in the column.
    """

        temp_dict = {}

        for index in range(len(scenes_list)):
            try:
                temp_dict[scenes_list[index]['file']] = round(float(scenes_list[index][object_name][attribute]), 3)
            except KeyError:
                temp_dict[scenes_list[index]['file']] = float('NaN')

        pd_series = pd.Series(temp_dict)

        return pd_series


    def generate_scenes_list(self):
    """
    Generates a list of dictionaries, with each dictionary representing
    the object attribute info from an xml file i.e. from each scene.

    Parameters
    ----------
    None

    Returns
    -------
    self.scenes_list: List
    List of dictionaries, each dict representing a scene.
    This list has everything that you need, seriously!
    """

        directory = os.fsencode(self.data_path)

        for file in os.listdir(directory):
            file_name = os.fsdecode(file)
            file_path = self.data_path + file_name

            self.scenes_list.append(self.parse_xml(os.path.abspath(file_path), file_name))

        with open(self.dict_dump_file_name, "wb") as myFile:
            pickle.dump(self.scenes_list, myFile)

        return self.scenes_list

    def get_objects_in_scenes(self):
    """
    Gets a list of all objects present in all scenes.

    Parameters
    ----------
    None

    Returns
    -------
    self.objects_in_scenes: List
    List of all objects present in the given scenes.
    """

        for i in range(len(self.scenes_list)):
            self.objects_in_scenes = list(set().union(self.objects_in_scenes, self.scenes_list[i].keys()))

        self.objects_in_scenes.remove('file')

        return self.objects_in_scenes

    def generate_scenes_df(self):
    """
    Collects the info of all attributes of all objects from all scenes
    and puts into into a nice Pandas DataFrame.

    Parameters
    ----------
    None

    Returns
    -------
    self.scenes_df: Pandas DataFrame
    A DataFrame that contains all the scenes as row labels and the attribute
    values as columns.
    """

        for obj in self.objects_in_scenes:
            for attribute in self.attributes:
                temp_pd_series = self.generate_pd_series(obj, attribute, self.scenes_list)
                self.scenes_df[obj + '_' + attribute] = temp_pd_series

        self.scenes_df.to_csv(self.df_dump_file_name, sep = '\t')

        return self.scenes_df
