import numpy as np
import h5py
from fuel.datasets.hdf5 import H5PYDataset
from fuel.utils import find_in_data_path

numpy_feature_train_file = "./data/emboot_onto/train_features_emboot.npy"
numpy_feature_test_file = "./data/emboot_onto/test_features_emboot.npy"
numpy_target_train_file = "./data/emboot_onto/train_targets_emboot.npy"
numpy_target_test_file = "./data/emboot_onto/test_targets_emboot.npy"

emboot_dataset = "./data/emboot_dataset.onto.hdf5"

def load_emboot_np():
    train_vector_features = np.load(numpy_feature_train_file)
    test_vector_features = np.load(numpy_feature_test_file)
    train_targets = np.expand_dims(np.load(numpy_target_train_file), axis=1)
    test_targets = np.expand_dims(np.load(numpy_target_test_file), axis=1)
    return train_vector_features, train_targets, test_vector_features, test_targets

def emboot_converter():
    train_vector_features, train_targets, test_vector_features, test_targets = load_emboot_np()
    f = h5py.File(emboot_dataset, mode='w')

    train_sz = train_vector_features.shape[0]
    test_sz = test_vector_features.shape[0]
    feat_sz = train_vector_features.shape[1]
    dataset_sz = train_sz + test_sz

    vector_features = f.create_dataset('features', (dataset_sz, feat_sz), dtype='float64')  ## train + test
    targets = f.create_dataset('targets', (dataset_sz, 1), dtype='uint8')

    ## put the data loaded into these objects
    vector_features[...] = np.vstack([train_vector_features, test_vector_features])
    targets[...] = np.vstack([train_targets, test_targets])

    ## label the dims with names
    vector_features.dims[0].label = 'batch'
    vector_features.dims[1].label = 'feature'
    targets.dims[0].label = 'batch'
    targets.dims[1].label = 'index'

    ## split attribute -- way to recover the splits
    # creating the split using an API
    split_dict = {
        'train': {'features': (0, train_sz), 'targets': (0, train_sz)},
         'test': {'features': (train_sz, dataset_sz), 'targets': (train_sz, dataset_sz)}}
    f.attrs['split'] = H5PYDataset.create_split_array(split_dict)

    f.flush()
    f.close()

class EMBOOT_ONTO(H5PYDataset):
    u"""EMBOOT dataset
        transformed to Fuel format
    """
    filename = 'emboot_dataset.onto.hdf5'

    def __init__(self, which_sets, **kwargs):
        kwargs.setdefault('load_in_memory', True)
        super(EMBOOT_ONTO, self).__init__(
            file_or_path=find_in_data_path(self.filename),
            which_sets=which_sets, **kwargs)


if __name__ == "__main__":
    # create_simple_data()
    # simple_converter()
    #
    # train_set = H5PYDataset('data/simple_dataset.hdf5', which_sets=('train',))
    emboot_converter()

    train_set = H5PYDataset(emboot_dataset, which_sets=('train',))

