
from copy import copy
class CombineTransforms(object):
    def __init__(self, list_of_transforms):
        super().__init__()
        self.transforms = list_of_transforms
    def __call__(self, data):
        list_of_datas = [transform(copy(data)) for transform in self.transforms]
        return list_of_datas

