import json
import io
from w2v import Gigaword
import numpy as np
import h5py
from fuel.datasets.hdf5 import H5PYDataset
from fuel.utils import find_in_data_path

def readWikiData(filename, gigaW2vEmbed, lookupGiga, labels_dict=None):
    with io.open(filename, encoding='utf-8') as fh:
        cnt = 0
        num_datapoints = 0
        datapoints = []
        for line in fh:
            jsonSentence = json.loads(line)
            tokens = jsonSentence['tokens']
            mentions = jsonSentence['mentions']
            for mention in mentions:
                # if (mention['start'] - pattern_window_sz < 0) or (mention['end'] + pattern_window_sz + 1 > len(tokens)) :
                patterns = patternsArndMention(tokens, mention)

                labels = mention['labels']
                # print (" ".join(tokens[mention['start']:mention['end']]))
                mention_embedding = create_embedding(patterns, mention, gigaW2vEmbed, lookupGiga)
                for lbl in labels:
                    num_datapoints += 1
                    datapoints.append((lbl, mention_embedding))
                # cnt += 1
                # break
                if num_datapoints % 250000 == 0:
                    print ("Processed " + str(num_datapoints) + " number of datapoints.")

            # if cnt == 5:
            #     break
        print ("Processed " + str(num_datapoints) + " number of datapoints. [COMPLETED]")

    if labels_dict is None:
        labels_dict_new = dict( [(lbl, idx) for idx, lbl in enumerate(list({d[0] for d in datapoints}))] )
        labels_np = np.array([ labels_dict_new[d[0]] for d in datapoints])
        embeddings_np = np.array([d[1] for d in datapoints])
        return labels_dict_new, labels_np, embeddings_np
    else:
        labels_np = np.array([labels_dict[d[0]] for d in datapoints])
        embeddings_np = np.array([d[1] for d in datapoints])
        return None, labels_np, embeddings_np

def create_embedding(patterns, mention, gigaW2vEmbed, lookupGiga):
    # print (mention)
    # print (patterns)

    pat_embedding_list = list()
    for pattern in patterns:
        words_in_pat = [Gigaword.sanitiseWord(w) for w in pattern if w != "@ENTITY"]
        embedIds_in_pat = [lookupGiga[w] if w in lookupGiga else lookupGiga["<unk>"] for w in words_in_pat]
        if len(embedIds_in_pat) > 1:
            pat_embedding = np.mean(Gigaword.norm(gigaW2vEmbed[embedIds_in_pat]), axis=0) # avg of the embeddings for each word in the pattern
        else:
            pat_embedding = Gigaword.norm(gigaW2vEmbed[embedIds_in_pat])

        pat_embedding_list.append(pat_embedding)

    pat_embedding_list_np = np.array([pe for pe in pat_embedding_list])
    if len(pat_embedding_list) > 0:
        pat_embedding_avg = np.average(pat_embedding_list_np, axis=0)
    else: ## NO patterns .. add the <unk> embedding
        pat_embedding_avg = Gigaword.norm(gigaW2vEmbed[lookupGiga["<unk>"]])

    words_in_ent = [Gigaword.sanitiseWord(w) for w in mention]
    embedIds_in_ent = [lookupGiga[w] if w in lookupGiga else lookupGiga["<unk>"] for w in words_in_ent]
    if len(embedIds_in_ent) > 1:
        ent_embedding = np.mean(Gigaword.norm(gigaW2vEmbed[embedIds_in_ent]),
                                axis=0)  # avg of the embeddings for each word in the pattern
    else:
        ent_embedding = Gigaword.norm(gigaW2vEmbed[embedIds_in_ent])

    embedding_vector = np.concatenate([pat_embedding_avg, ent_embedding])
    return embedding_vector

def patternsArndMention(tokens, mention):
    start = mention['start']
    end = mention['end']
    tok_len = len(tokens)

    # print ("Sentence: ", " ".join(tokens))
    # print ("Mention: ", " ".join(tokens[start:end]))
    # print ("Start: " , start)
    # print ("End: ", end)

    left_ctx_start = max(0, start - pattern_window_sz)
    left_ctx_end = start

    right_ctx_start = end
    right_ctx_end =  min(end + pattern_window_sz, tok_len) + 1

    left_ctx = []
    for i in range(left_ctx_start, left_ctx_end):
        left_ctx.append(tokens[i: left_ctx_end])
        # print (" ".join(tokens[i: left_ctx_end]))

    right_ctx = []
    for i in range(right_ctx_start+1, right_ctx_end):
        right_ctx.append(tokens[right_ctx_start: i])
        # print (" ".join(tokens[i: right_ctx_end]))

    around_ctx = []
    for l in left_ctx:
        for r in right_ctx:
            arnd = [i for i in l]
            arnd.append("@ENTITY")
            for i in r:
                arnd.append(i)
            around_ctx.append(arnd)

    new_left_ctx = []
    for lc in left_ctx:
        new_left_ctx.append([i for i in lc] + ["@ENTITY"])

    new_right_ctx = []
    for rc in right_ctx:
        new_right_ctx.append(["@ENTITY"] + [i for i in rc])

    # print ("----")
    # for c in around_ctx:
    #     print (" ".join(c))
    # print ("----")
    #
    # for c in new_right_ctx:
    #     print (" ".join(c))
    # print ("----")
    #
    # for c in new_left_ctx:
    #     print (" ".join(c))
    # print ("----")

    return (new_left_ctx + new_right_ctx + around_ctx)

def fuel_converter(fuel_dataset, embeddings_train, labels_train, embeddings_test, labels_test):
    f = h5py.File(fuel_dataset, mode='w')

    train_sz = embeddings_train.shape[0] - embeddings_train.shape[0] % 100
    test_sz = embeddings_test.shape[0] - embeddings_test.shape[0] % 100
    feat_sz = embeddings_train.shape[1]
    dataset_sz = train_sz + test_sz

    print ("Actual Train size : " , embeddings_train.shape[0])
    print ("Train size in Fuel : " , train_sz)

    print ("Actual Test size : " , embeddings_test.shape[0])
    print ("Test size in Fuel : " , test_sz)

    vector_features = f.create_dataset('features', (dataset_sz, feat_sz), dtype='float64')
    targets = f.create_dataset('targets', (dataset_sz, 1), dtype='uint8')

    ## put the data loaded into these objects
    vector_features[...] = np.vstack([embeddings_train[0:train_sz], embeddings_test[0:test_sz]])
    targets[...] = np.vstack([labels_train[0:train_sz], labels_test[0:test_sz]])

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

class FIGER(H5PYDataset):
    u"""EMBOOT dataset
        transformed to Fuel format
    """
    filename = 'figer.hdf5'

    def __init__(self, which_sets, **kwargs):
        kwargs.setdefault('load_in_memory', True)
        super(FIGER, self).__init__(
            file_or_path=find_in_data_path(self.filename),
            which_sets=which_sets, **kwargs)

if __name__ == "__main__":

    pattern_window_sz = 4
    train_filename = "./data/Wiki/train.json" #""/Users/ajaynagesh/Research/LadderNetworks/ner_data/AFET/Data/Wiki/train.json"
    test_filename = "./data/Wiki/test.json" #"/Users/ajaynagesh/Research/LadderNetworks/ner_data/AFET/Data/Wiki/test.json"
    w2vfile = "./data/vectors.goldbergdeps.txt" #"/Users/ajaynagesh/Research/code/research/clint/data/vectors.goldbergdeps.txt"
    fuel_dataset = "./data/figer.hdf5" #"/Users/ajaynagesh/Research/LadderNetworks/ner_data/figer.hdf5"

    print("Loading the gigaword embeddings ...")
    gigaW2vEmbed, lookupGiga = Gigaword.load_pretrained_embeddings(w2vfile)

    labels_dict, labels_train, embeddings_train = readWikiData(train_filename, gigaW2vEmbed, lookupGiga, None)
    _, labels_test, embeddings_test = readWikiData(test_filename, gigaW2vEmbed, lookupGiga, labels_dict)

    print("Train ..")
    print(labels_train.shape)
    print(embeddings_train.shape)

    print("Test ..")
    print(labels_test.shape)
    print(embeddings_test.shape)

    print(labels_dict)

    fuel_converter(fuel_dataset, embeddings_train, labels_train, embeddings_test, labels_test)
