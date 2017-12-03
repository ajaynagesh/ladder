import numpy as np
import h5py
from fuel.datasets.hdf5 import H5PYDataset
from fuel.utils import find_in_data_path

class SST1(H5PYDataset):
    u"""Stanford Senti treebank 1 dataset
        with 5 labels
        transformed to Fuel format
        Note: the train consists of both phrases as well as sentences
    """
    filename = 'SST1.fuel.hdf5'

    def __init__(self, which_sets, **kwargs):
        kwargs.setdefault('load_in_memory', True)
        super(SST1, self).__init__(
            file_or_path=find_in_data_path(self.filename),
            which_sets=which_sets, **kwargs)

class SST2(H5PYDataset):
    u"""Stanford Senti treebank 1 dataset
        with 2 labels (binary dataset)
        transformed to Fuel format
        Note: the train consists of both phrases as well as sentences
    """
    filename = 'SST2.fuel.hdf5'

    def __init__(self, which_sets, **kwargs):
        kwargs.setdefault('load_in_memory', True)
        super(SST2, self).__init__(
            file_or_path=find_in_data_path(self.filename),
            which_sets=which_sets, **kwargs)

class SST2mod(H5PYDataset):
    u"""Stanford Senti treebank 1 dataset
        with 2 labels (binary dataset)
        transformed to Fuel format
        Note: the train consists of both phrases as well as sentences
    """
    filename = 'SST2.fuel.mod.hdf5'

    def __init__(self, which_sets, **kwargs):
        kwargs.setdefault('load_in_memory', True)
        super(SST2mod, self).__init__(
            file_or_path=find_in_data_path(self.filename),
            which_sets=which_sets, **kwargs)


def load_harv_sst_dataset(sst_dataset):
    # Load the h5py file
    dataset_harv = h5py.File(sst_dataset, 'r')

    train_phrases = dataset_harv['train'].value
    train_targets = dataset_harv['train_label'].value

    test_sents = dataset_harv['test'].value
    test_targets = dataset_harv['test_label'].value

    dev_sents = dataset_harv['dev'].value
    dev_targets = dataset_harv['dev_label'].value

    #### VERY IMPT: THE TORCH ARRAYS ARE INDEXED FROM 1. But when loaded using HDF5 IT is ZERO INDEXED.
    ### So w2v INDICES WHERE GETTING SCREWED
    ### REDUCING THE INDICES BY 1
    train_phrases = train_phrases - 1
    train_targets = train_targets - 1
    test_sents = test_sents - 1
    test_targets = test_targets - 1
    dev_sents = dev_sents - 1
    dev_targets = dev_targets - 1

    w2v = dataset_harv['w2v'].value

    train_features = np.stack([w2v[sent] for sent in train_phrases], axis=0)
    test_features = np.stack([w2v[sent] for sent in test_sents], axis=0)
    dev_features = np.stack([w2v[sent] for sent in dev_sents], axis=0)

    return train_features, train_targets, \
           test_features, test_targets, \
           dev_features, dev_targets

def sst_converter(sst_harv_dataset, sst_fuel_dataset):
    train_features, train_targets, \
    test_features, test_targets, \
    dev_features, dev_targets, = load_harv_sst_dataset(sst_harv_dataset)

    f = h5py.File(sst_fuel_dataset, mode='w')

    train_sz = train_features.shape[0]
    test_sz = test_features.shape[0]
    dev_sz = dev_features.shape[0]
    sent_sz = train_features.shape[1]
    embedding_sz = train_features.shape[2]
    dataset_sz = train_sz + dev_sz + test_sz

    # dataset_sz x sent_sz x embed_sz
    # 160128 x 61 x 300 -- SST1
    #  79654 x 61 x 300 -- SST2
    sent_features = f.create_dataset('features', (dataset_sz, sent_sz, embedding_sz), dtype='float32')
    targets = f.create_dataset('targets', (dataset_sz, 1), dtype='uint8')

    ## put the data loaded into these objects
    sent_features[...] = np.vstack([train_features, dev_features, test_features])
    targets[...] = np.vstack([np.expand_dims(train_targets,axis=1), np.expand_dims(dev_targets,axis=1), np.expand_dims(test_targets,axis=1)])

    ## label the dims with names
    sent_features.dims[0].label = 'batch'
    sent_features.dims[1].label = 'word'
    sent_features.dims[2].label = 'embed'
    targets.dims[0].label = 'batch'
    targets.dims[1].label = 'index'

    ## split attribute -- way to recover the splits
    # creating the split using an API
    split_dict = {
        'train': {'features': (0, train_sz), 'targets': (0, train_sz)},
        'dev': {'features': (train_sz, train_sz+dev_sz), 'targets': (train_sz, train_sz+dev_sz)},
        'test': {'features': (train_sz+dev_sz, dataset_sz), 'targets': (train_sz+dev_sz, dataset_sz)}}
    f.attrs['split'] = H5PYDataset.create_split_array(split_dict)

    f.flush()
    f.close()

def sst_converter_expand_dims(sst_harv_dataset, sst_fuel_dataset):
    train_features, train_targets, \
    test_features, test_targets, \
    dev_features, dev_targets, = load_harv_sst_dataset(sst_harv_dataset)

    train_features = np.expand_dims(train_features, axis=1)
    test_features = np.expand_dims(test_features, axis=1)
    dev_features = np.expand_dims(dev_features, axis=1)

    f = h5py.File(sst_fuel_dataset, mode='w')

    train_sz = train_features.shape[0]
    test_sz = test_features.shape[0]
    dev_sz = dev_features.shape[0]
    channel_sz = train_features.shape[1]
    sent_sz = train_features.shape[2]
    embedding_sz = train_features.shape[3]
    dataset_sz = train_sz + dev_sz + test_sz

    # dataset_sz x sent_sz x embed_sz
    # 160128 x 1 x 61 x 300 -- SST1
    #  79654 x 1 x 61 x 300 -- SST2
    sent_features = f.create_dataset('features', (dataset_sz, channel_sz, sent_sz, embedding_sz), dtype='float32')
    targets = f.create_dataset('targets', (dataset_sz, 1), dtype='uint8')

    ## put the data loaded into these objects
    sent_features[...] = np.vstack([train_features, dev_features, test_features])
    targets[...] = np.vstack([np.expand_dims(train_targets,axis=1), np.expand_dims(dev_targets,axis=1), np.expand_dims(test_targets,axis=1)])

    ## label the dims with names
    sent_features.dims[0].label = 'batch'
    sent_features.dims[1].label = 'channel'
    sent_features.dims[2].label = 'word'
    sent_features.dims[3].label = 'embed'
    targets.dims[0].label = 'batch'
    targets.dims[1].label = 'index'

    ## split attribute -- way to recover the splits
    # creating the split using an API
    split_dict = {
        'train': {'features': (0, train_sz), 'targets': (0, train_sz)},
        'dev': {'features': (train_sz, train_sz+dev_sz), 'targets': (train_sz, train_sz+dev_sz)},
        'test': {'features': (train_sz+dev_sz, dataset_sz), 'targets': (train_sz+dev_sz, dataset_sz)}}
    f.attrs['split'] = H5PYDataset.create_split_array(split_dict)

    f.flush()
    f.close()

def sst_converter_expand_dims_mod(sst_harv_dataset, sst_fuel_dataset):
    train_features, train_targets, \
    test_features, test_targets, \
    dev_features, dev_targets, = load_harv_sst_dataset(sst_harv_dataset)

    train_features = np.expand_dims(train_features, axis=1)
    test_features = np.expand_dims(test_features, axis=1)
    dev_features = np.expand_dims(dev_features, axis=1)

    f = h5py.File(sst_fuel_dataset, mode='w')

    ### NOTE: Modified to be a whole number of the batch size ... TODO: Change this ?
    train_sz = train_features.shape[0] - 961
    test_sz = test_features.shape[0] - 21
    dev_sz = dev_features.shape[0]
    channel_sz = train_features.shape[1]
    sent_sz = train_features.shape[2]
    embedding_sz = train_features.shape[3]
    dataset_sz = train_sz + dev_sz + test_sz

    # dataset_sz x sent_sz x embed_sz
    # 160128 x 1 x 61 x 300 -- SST1
    #  79654 x 1 x 61 x 300 -- SST2
    sent_features = f.create_dataset('features', (dataset_sz, channel_sz, sent_sz, embedding_sz), dtype='float32')
    targets = f.create_dataset('targets', (dataset_sz, 1), dtype='uint8')

    ## put the data loaded into these objects
    sent_features[...] = np.vstack([train_features[:train_sz], dev_features, test_features[:test_sz]])
    targets[...] = np.vstack([np.expand_dims(train_targets[:train_sz],axis=1), np.expand_dims(dev_targets,axis=1),
                              np.expand_dims(test_targets[:test_sz],axis=1)])

    ## label the dims with names
    sent_features.dims[0].label = 'batch'
    sent_features.dims[1].label = 'channel'
    sent_features.dims[2].label = 'word'
    sent_features.dims[3].label = 'embed'
    targets.dims[0].label = 'batch'
    targets.dims[1].label = 'index'

    ## split attribute -- way to recover the splits
    # creating the split using an API
    split_dict = {
        'train': {'features': (0, train_sz), 'targets': (0, train_sz)},
        'dev': {'features': (train_sz, train_sz+dev_sz), 'targets': (train_sz, train_sz+dev_sz)},
        'test': {'features': (train_sz+dev_sz, dataset_sz), 'targets': (train_sz+dev_sz, dataset_sz)}}
    f.attrs['split'] = H5PYDataset.create_split_array(split_dict)

    f.flush()
    f.close()

if __name__ == "__main__":

    # sst_harv_dataset1 = "./data/SST1.harvard.hdf5" # This files are obtained from running : https://github.com/harvardnlp/sent-conv-torch/blob/pytorch/preprocess.py
    # sst_fuel_dataset1 = "./data/SST1.fuel.hdf5"
    # sst_converter(sst_harv_dataset1, sst_fuel_dataset1)
    #
    # train_set = H5PYDataset('data/simple_dataset.hdf5', which_sets=('train',))

    sst_harv_dataset2 = "./data/SST2.harvard.hdf5"  # This files are obtained from running : https://github.com/harvardnlp/sent-conv-torch/blob/pytorch/preprocess.py
    # sst_fuel_dataset2 = "./data/SST2.fuel.hdf5"
    # sst_converter(sst_harv_dataset2, sst_fuel_dataset2)

    sst_fuel_dataset2 = "./data/SST2.fuel.mod.hdf5"
    sst_converter_expand_dims_mod(sst_harv_dataset2, sst_fuel_dataset2) ## NOTE: calling the mod function to make the train and test whole numbers when divided by batch_sz TODO: Change this ... ?
