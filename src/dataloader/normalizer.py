import numpy as np
class Normalizer:
    def __init__(self, range: list, target_range: list = [-1.0, 1.0]) -> None:
        # [[-0.14, 6], [-0.35, 0.35], [-0.15, 0.15]]
        self.range = np.array(range)  # act_dims, 2
        self.target_range = np.array(target_range)

    def normalize(self, data):
        # data (..., act_dims)
        # standard: x-x_min / (x_max-x_min) falls into [0,1]
        data_ = (data - self.range[:, 0]) / (self.range[:, 1] - self.range[:, 0])
        target_data = data_ * (self.target_range[1] - self.target_range[0]) + self.target_range[0]
        return target_data

    def denormalize(self, target_data):
        data_ = (target_data - self.target_range[0]) / (self.target_range[1] - self.target_range[0])
        data = data_ * (self.range[:, 1] - self.range[:, 0]) + self.range[:, 0]
        return data
if __name__ == '__main__':
    from utils import loading_data
    no = Normalizer([[-0.14, 6], [-0.35, 0.35], [-0.15,0.15]])
    data = loading_data('example_data')
    data_no = no.normalize(data)
    idx= 50
    print(data_no[0,idx])
    print(data[0,idx])
    print(no.denormalize(data_no)[0,idx])