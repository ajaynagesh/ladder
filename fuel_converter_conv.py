import numpy as np
import h5py
from fuel.datasets.hdf5 import H5PYDataset
from fuel.utils import find_in_data_path

def load_emboot_np(numpy_feature_train_file, numpy_target_train_file):
    train_vector_features = np.load(numpy_feature_train_file)
    train_targets = np.expand_dims(np.load(numpy_target_train_file), axis=1)
    return train_vector_features, train_targets

def emboot_converter_traintrain(emboot_dataset, numpy_feature_train_file, numpy_target_train_file):
    train_vector_features, train_targets = load_emboot_np(numpy_feature_train_file, numpy_target_train_file)
    f = h5py.File(emboot_dataset, mode='w')

    train_sz = train_vector_features.shape[0]
    num_ctxs = train_vector_features.shape[1]
    ctx_size = train_vector_features.shape[2]
    embed_sz = train_vector_features.shape[3]
    dataset_sz = (train_sz - 16) * 2 ## NOTE: 13900 * 2 (copy over the train data to the test dataset)

    vector_features = f.create_dataset('features', (dataset_sz, num_ctxs, ctx_size, embed_sz), dtype='float64')  ## train + test
    targets = f.create_dataset('targets', (dataset_sz, 1), dtype='uint8')

    ## put the data loaded into these objects

    train_vector_features_rounded = train_vector_features[:13900]
    train_targets_rounded = train_targets[:13900]

    vector_features[...] = np.vstack([train_vector_features_rounded, train_vector_features_rounded])
    targets[...] = np.vstack([train_targets_rounded, train_targets_rounded])

    ## label the dims with names
    vector_features.dims[0].label = 'batch'
    vector_features.dims[1].label = 'ent_ctx'
    vector_features.dims[2].label = 'word'
    vector_features.dims[3].label = 'embed'
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


class EMBOOT_CONV_CONLL(H5PYDataset):
    u"""EMBOOT dataset
        transformed to Fuel format
    """
    filename = 'conll.conv.traintrain.hdf5'

    def __init__(self, which_sets, **kwargs):
        kwargs.setdefault('load_in_memory', True)
        super(EMBOOT_CONV_CONLL, self).__init__(
            file_or_path=find_in_data_path(self.filename),
            which_sets=which_sets, **kwargs)

if __name__ == "__main__":
    # create_simple_data()
    # simple_converter()
    #
    # train_set = H5PYDataset('data/simple_dataset.hdf5', which_sets=('train',))
    # emboot_dataset = "./data/emboot_dataset.new.hdf5"
    # emboot_converter()

    numpy_feature_train_file = "./data/data_conll_conv/train_features_emboot_conv.npy"
    numpy_target_train_file = "./data/data_conll_conv/train_targets_emboot_conv.npy"
    emboot_dataset = "./data/conll.conv.traintrain.hdf5"
    emboot_converter_traintrain(emboot_dataset, numpy_feature_train_file, numpy_target_train_file)

    #
    # train_set = H5PYDataset(emboot_dataset, which_sets=('train',))

