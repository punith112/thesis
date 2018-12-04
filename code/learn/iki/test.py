# from single_object import SingleObjectFeatures
#
# features = ['x', 'y', 'z', 'length', 'width', 'height']
# test = SingleObjectFeatures("extracted_data.txt", "database", features)
# result = test.get_gmm_params()

from object_pair import ObjectPairFeatures

test = ObjectPairFeatures("extracted_data.txt")
result = test.get_gmm_params()
