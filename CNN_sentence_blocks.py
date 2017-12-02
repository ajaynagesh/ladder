"""Convolutional network example.

Run the training for 50 epochs with
```
python __init__.py --num-epochs 50
```
It is going to reach around 0.8% error rate on the test set.

"""
import logging
from argparse import ArgumentParser

from blocks.algorithms import GradientDescent, CompositeRule, StepClipping, AdaDelta
from blocks.bricks import (MLP, Rectifier, Softmax)
from blocks.bricks.conv import (Convolutional, ConvolutionalSequence,
                                Flattener, MaxPooling)
from blocks.bricks.cost import CategoricalCrossEntropy, MisclassificationRate
from blocks.extensions import FinishAfter, Timing, Printing, ProgressBar
from blocks.extensions.monitoring import (DataStreamMonitoring,
                                          TrainingDataMonitoring)
from blocks.extensions.saveload import Checkpoint
from blocks.filter import VariableFilter
from blocks.graph import ComputationGraph
from blocks.graph import apply_dropout
from blocks.initialization import Constant, Uniform, IsotropicGaussian
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.monitoring import aggregation
from blocks.roles import INPUT
from fuel.schemes import ShuffledScheme
from fuel.streams import DataStream
from theano import tensor

from processData.fuel_converter_sst import SST2


def ConvBrick_1D(num_channels, image_shape,
             filter_size, feature_maps, pooling_size, i=0,
             weights_init=Uniform(width=.02), biases_init=Constant(0)):

    conv = Convolutional(filter_size=filter_size,
                   num_filters=feature_maps,
                   name='conv_{}'.format(i),
                   weights_init=weights_init,
                   biases_init=biases_init)

    act = Rectifier()

    pool = MaxPooling(pooling_size=pooling_size, name='pool_{}'.format(i))

    layers = [conv, act, pool]

    conv_sequence = ConvolutionalSequence(layers,
                                          num_channels,
                                          image_size=image_shape,
                                          name='conv_seq_{}'.format(i))

    return conv_sequence


def main(save_to, num_epochs, num_batches=None, batch_size=50):

    feature_maps = [100]
    mlp_hiddens = [100]
    conv_sizes = [(3, 300), (4, 300), (5, 300)]
    pool_sizes = [(61-3+1, 1), (61-4+1, 1), (61-5+1, 1)]
    image_size = (61, 300)

    x = tensor.tensor4('features')
    y = tensor.lmatrix('targets')

    conv_parameters = [(i, j) for i in conv_sizes for j in feature_maps]

    conv_1d = []     ## Each of the 1D convolutions as a separate brick
    for i, (filter_size, num_filter) in enumerate(conv_parameters):
        conv = ConvBrick_1D(1, image_size,
                           filter_size=filter_size,
                           feature_maps=num_filter,
                           pooling_size=pool_sizes[i], i=i,
                           weights_init=Uniform(width=.02),
                           biases_init=Constant(0))

        conv.push_allocation_config()
        conv.push_initialization_config()
        ### initialize each of the conv1d bricks
        # conv.layers[0].weights_init = Uniform(width=.02)
        # conv.layers[1].weights_init = Uniform(width=.09)
        # conv.layers[2].weights_init = Uniform(width=.09)
        conv.initialize()
        conv_1d.append(conv)

    flattener = Flattener()

    ## Apply all the convolutions and concatenate the output and flatten it to input to MLP
    mlp_input_tensor = flattener.apply(tensor.concatenate([conv.apply(x) for conv in conv_1d], axis=1))

    # Construct a top MLP
    mlp_activations = [Rectifier() for _ in mlp_hiddens] + [Softmax()]
    top_mlp = MLP(mlp_activations, dims=[300, 100, 2], ## TODO: Need to parameterize this
                  weights_init=IsotropicGaussian(std=1, mean=0.01), biases_init=Constant(0)) ## TODO: Need to parameterize this
##Uniform(width=.02),
    top_mlp.initialize()

    probs = top_mlp.apply(mlp_input_tensor)

    # TODO: Need to print appropriate info of the network constructed
    # logging.info("Input dim: {} {} {}".format(
    #     *convnet.children[0].get_dim('input_')))
    # for i, layer in enumerate(convnet.layers):
    #     if isinstance(layer, Activation):
    #         logging.info("Layer {} ({})".format(
    #             i, layer.__class__.__name__))
    #     else:
    #         logging.info("Layer {} ({}) dim: {} {} {}".format(
    #             i, layer.__class__.__name__, *layer.get_dim('output')))


    # Normalize input and apply the convnet
    cost = (CategoricalCrossEntropy().apply(y.flatten(), probs)
            .copy(name='cost'))
    error_rate = (MisclassificationRate().apply(y.flatten(), probs)
                  .copy(name='error_rate'))

    cg = ComputationGraph([cost, error_rate])

    dropout_var = [var for var in VariableFilter(roles=[INPUT])(cg.variables) if var.name == 'linear_0_apply_input_']
    cg_with_dropout = apply_dropout(cg, dropout_var, 0.5)

    sst2_train = SST2(("train",))
    sst2_train_stream = DataStream.default_stream(
        sst2_train, iteration_scheme=ShuffledScheme(
            sst2_train.num_examples, batch_size))

    sst2_test = SST2(("dev",))
    sst2_test_stream = DataStream.default_stream(
        sst2_test,
        iteration_scheme=ShuffledScheme(
            sst2_test.num_examples, batch_size))

    # Train with simple SGD
    algorithm = GradientDescent(
        cost=cost, parameters=cg_with_dropout.parameters,
        step_rule=CompositeRule([StepClipping(threshold=3), AdaDelta()]) )
    # `Timing` extension reports time for reading data, aggregating a batch
    # and monitoring;
    # `ProgressBar` displays a nice progress bar during training.
    extensions = [Timing(),
                  FinishAfter(after_n_epochs=num_epochs,
                              after_n_batches=num_batches),
                  DataStreamMonitoring(
                      [cost, error_rate],
                      sst2_test_stream,
                      prefix="dev"),
                  TrainingDataMonitoring(
                      [cost, error_rate,
                       aggregation.mean(algorithm.total_gradient_norm)],
                      prefix="train",
                      after_epoch=True),
                  Checkpoint(save_to),
                  ProgressBar(),
                  Printing()]

    model = Model(cost)

    main_loop = MainLoop(
        algorithm,
        sst2_train_stream,
        model=model,
        extensions=extensions)

    main_loop.run()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = ArgumentParser("An example of training a convolutional network "
                            "on the MNIST dataset.")
    parser.add_argument("--num-epochs", type=int, default=25,
                        help="Number of training epochs to do.")
    parser.add_argument("--save_to", default="sst2.pkl", nargs="?",
                        help="Destination to save the state of the training "
                             "process.")
    # parser.add_argument("--feature-maps", type=int, nargs='+',
    #                     default=[100], help="List of feature maps numbers.")
    # parser.add_argument("--mlp-hiddens", type=int, nargs='+', default=[100],
    #                     help="List of numbers of hidden units for the MLP.")
    # parser.add_argument("--conv-sizes", type=int, nargs='+', default=[(3, 300)],
    #                     help="Convolutional kernels sizes. The kernels are "
    #                     "always square.")
    # parser.add_argument("--pool-sizes", type=int, nargs='+', default=[(61+3-1, 1)],
    #                     help="Pooling sizes. The pooling windows are always "
    #                          "square. Should be the same length as "
    #                          "--conv-sizes.")
    parser.add_argument("--batch-size", type=int, default=50,
                        help="Batch size.")
    args = parser.parse_args()
    main(**vars(args))
    # main('sst2_100_with_dropout.pkl', 100)
