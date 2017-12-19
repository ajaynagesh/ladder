import numpy as np
import h5py
from fuel.datasets.hdf5 import H5PYDataset
from fuel.utils import find_in_data_path

def create_simple_data():
    ## save as numpy vectors on disk
    ## dataset of 90 datapoints --> train, 10 --> test
    ## features --> 1) --> 3 x 5 x 5 2) --> 1 x 10
    np.save('data/simple_data/train_vector_features.npy', np.random.normal(size=(90, 10)).astype('float32'))
    np.save('data/simple_data/test_vector_features.npy', np.random.normal(size=(10, 10)).astype('float32'))
    np.save('data/simple_data/train_image_features.npy', np.random.randint(2, size=(90, 3, 5, 5)).astype('uint8'))
    np.save('data/simple_data/test_image_features.npy', np.random.randint(2, size=(10, 3, 5, 5)).astype('uint8'))
    np.save('data/simple_data/train_targets.npy', np.random.randint(10, size=(90, 1)).astype('uint8'))
    np.save('data/simple_data/test_targets.npy', np.random.randint(10, size=(10, 1)).astype('uint8'))

def load_simple_data():
    train_vector_features = np.load('data/simple_data/train_vector_features.npy')
    test_vector_features = np.load('data/simple_data/test_vector_features.npy')
    train_image_features = np.load('data/simple_data/train_image_features.npy')
    test_image_features = np.load('data/simple_data/test_image_features.npy')
    train_targets = np.load('data/simple_data/train_targets.npy')
    test_targets = np.load('data/simple_data/test_targets.npy')
    return train_vector_features, train_image_features, train_targets, test_vector_features, test_image_features, test_targets

def simple_converter():
    train_vector_features, train_image_features, train_targets, test_vector_features, test_image_features, test_targets = load_simple_data()
    f = h5py.File('data/simple_dataset.hdf5', mode='w')
    vector_features = f.create_dataset('vector_features', (100, 10), dtype='float32') ## train + test
    image_features = f.create_dataset('image_features', (100, 3, 5, 5), dtype='uint8')
    targets = f.create_dataset('targets', (100, 1), dtype='uint8')

    ## put the data loaded into these objects
    vector_features[...] = np.vstack([train_vector_features, test_vector_features])
    image_features[...] = np.vstack([train_image_features, test_image_features])
    targets[...] = np.vstack([train_targets, test_targets])

    ## label the dims with names
    vector_features.dims[0].label = 'batch'
    vector_features.dims[1].label = 'feature'
    image_features.dims[0].label = 'batch'
    image_features.dims[1].label = 'channel'
    image_features.dims[2].label = 'height'
    image_features.dims[3].label = 'width'
    targets.dims[0].label = 'batch'
    targets.dims[1].label = 'index'

    ## split attribute -- way to recover the splits
    ## 1D numpy array with six elements.
    ## The dtype for this array is a compound type: every element of the array is a tuple of (str, str, int, int, h5py.Reference, bool, str)
    ## commented ... for a simple way to create the split_attribute see below
    # split_array = np.empty(
    #      6,
    #      dtype=np.dtype([ # size is the length of the longest attribute of that class
    #          ('split', 'a', 5), # |train|
    #          ('source', 'a', 15), # |vector_features|
    #          ('start', np.int64, 1),
    #          ('stop', np.int64, 1),
    #          ('indices', h5py.special_dtype(ref=h5py.Reference)),
    #          ('available', np.bool, 1),
    #          ('comment', 'a', 1)])) # Due to a quirk in pickling empty strings, we put '.' as the comment value.
    # split_array[0:3]['split'] = 'train'.encode('utf8')
    # split_array[3:6]['split'] = 'test'.encode('utf8')
    # split_array[0:6:3]['source'] = 'vector_features'.encode('utf8')
    # split_array[1:6:3]['source'] = 'image_features'.encode('utf8')
    # split_array[2:6:3]['source'] = 'targets'.encode('utf8')
    # split_array[0:3]['start'] = 0
    # split_array[0:3]['stop'] = 90
    # split_array[3:6]['start'] = 90
    # split_array[3:6]['stop'] = 100
    # split_array[:]['indices'] = h5py.Reference()
    # split_array[:]['available'] = True
    # split_array[:]['comment'] = '.'.encode('utf8')
    # f.attrs['split'] = split_array

    # creating the split using an API
    split_dict = {
        'train': {'vector_features': (0, 90), 'image_features': (0, 90), 'targets': (0, 90)},
         'test': {'vector_features': (90, 100), 'image_features': (90, 100), 'targets': (90, 100)}}
    f.attrs['split'] = H5PYDataset.create_split_array(split_dict)

    f.flush()
    f.close()

numpy_feature_train_file = "./data/emboot_conll/train_features_emboot.npy"
numpy_feature_test_file = "./data/emboot_conll/test_features_emboot.npy"
numpy_target_train_file = "./data/emboot_conll/train_targets_emboot.npy"
numpy_target_test_file = "./data/emboot_conll/test_targets_emboot.npy"

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
    dataset_sz_new = 13000 

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
        'train': {'features': (0, 10400), 'targets': (0, 10400)},
         'test': {'features': (10400, 13000), 'targets': (10400, 13000)}}
    f.attrs['split'] = H5PYDataset.create_split_array(split_dict)

    f.flush()
    f.close()

def emboot_converter_traintrain(emboot_dataset):
    train_vector_features, train_targets, test_vector_features, test_targets = load_emboot_np()
    f = h5py.File(emboot_dataset, mode='w')

    train_sz = train_vector_features.shape[0]
    test_sz = test_vector_features.shape[0]
    feat_sz = train_vector_features.shape[1]
    dataset_sz = (train_sz + test_sz - 16) * 2 ## NOTE: 13900 * 2 (copy over the train data to the test dataset)

    vector_features = f.create_dataset('features', (dataset_sz, feat_sz), dtype='float64')  ## train + test
    targets = f.create_dataset('targets', (dataset_sz, 1), dtype='uint8')

    ## put the data loaded into these objects

    train_vector_features_aug = np.vstack([train_vector_features, test_vector_features])[:13900]
    train_targets_aug = np.vstack([train_targets, test_targets])[:13900]

    vector_features[...] = np.vstack([train_vector_features_aug, train_vector_features_aug])
    targets[...] = np.vstack([train_targets_aug, train_targets_aug])

    ## label the dims with names
    vector_features.dims[0].label = 'batch'
    vector_features.dims[1].label = 'feature'
    targets.dims[0].label = 'batch'
    targets.dims[1].label = 'index'

    ## split attribute -- way to recover the splits
    # creating the split using an API
    split_dict = {
        'train': {'features': (0, dataset_sz/2), 'targets': (0, dataset_sz/2)},
         'test': {'features': (dataset_sz/2, dataset_sz), 'targets': (dataset_sz/2, dataset_sz)}}
    f.attrs['split'] = H5PYDataset.create_split_array(split_dict)

    f.flush()
    f.close()


class EMBOOT_CONLL(H5PYDataset):
    u"""EMBOOT dataset
        transformed to Fuel format
    """
    filename = 'conll.traintrain.hdf5'

    def __init__(self, which_sets, **kwargs):
        kwargs.setdefault('load_in_memory', True)
        super(EMBOOT_CONLL, self).__init__(
            file_or_path=find_in_data_path(self.filename),
            which_sets=which_sets, **kwargs)


class EMBOOT_CONLL_SPLIT(H5PYDataset):
    u"""EMBOOT dataset
        transformed to Fuel format
    """
    filename = 'emboot_dataset.new.hdf5'

    def __init__(self, which_sets, **kwargs):
        kwargs.setdefault('load_in_memory', True)
        super(EMBOOT_CONLL_SPLIT, self).__init__(
            file_or_path=find_in_data_path(self.filename),
            which_sets=which_sets, **kwargs)


if __name__ == "__main__":
    # create_simple_data()
    # simple_converter()
    #
    # train_set = H5PYDataset('data/simple_dataset.hdf5', which_sets=('train',))
    # emboot_dataset = "./data/emboot_dataset.new.hdf5"
    # emboot_converter()

    emboot_dataset = "./data/conll.traintrain.hdf5"
    emboot_converter_traintrain(emboot_dataset)

    #
    # train_set = H5PYDataset(emboot_dataset, which_sets=('train',))

