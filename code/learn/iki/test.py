from single_object import SingleObjectWrapper
from object_pair import ObjectPairWrapper
import pickle

with open("extracted_data.txt", "rb") as myFile:
    main_list = pickle.load(myFile)

objects_in_scene = []

for i in range(len(main_list)):
    objects_in_scene = list(set().union(objects_in_scene, main_list[i].keys()))

objects_in_scene.remove('file')

features = ['x', 'y', 'z', 'length', 'width', 'height']

single_object_wrapper = SingleObjectWrapper(objects_in_scene, "database", features)
single_object_gmms = single_object_wrapper.get_gmm_params()

object_pair_wrapper = ObjectPairWrapper(objects_in_scene)
object_pair_gmms = object_pair_wrapper.get_gmm_params()
