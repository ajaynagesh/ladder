"""Convolutional network example.

Run the training for 50 epochs with
```
python __init__.py --num-epochs 50
```
It is going to reach around 0.8% error rate on the test set.

"""
import logging
import numpy
from argparse import ArgumentParser

from theano import tensor

from blocks.algorithms import GradientDescent, Scale
from blocks.bricks import (MLP, Rectifier, Initializable, FeedforwardSequence,
                           Softmax, Activation)
from blocks.bricks.conv import (Convolutional, ConvolutionalSequence,
                                Flattener, MaxPooling)
from blocks.bricks.cost import CategoricalCrossEntropy, MisclassificationRate
from blocks.extensions import FinishAfter, Timing, Printing, ProgressBar
from blocks.extensions.monitoring import (DataStreamMonitoring,
                                          TrainingDataMonitoring)
from blocks.extensions.saveload import Checkpoint
from blocks.graph import ComputationGraph
from blocks.initialization import Constant, Uniform
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.monitoring import aggregation
from fuel.datasets import MNIST
from fuel.schemes import ShuffledScheme
from fuel.streams import DataStream
from toolz.itertoolz import interleave
from fuel_converter_sst import SST2

from blocks.graph import apply_dropout
from blocks.filter import VariableFilter
from blocks.roles import  INPUT

class LeNet(Initializable):
    """LeNet-like convolutional network.

    The class implements LeNet, which is a convolutional sequence with
    an MLP on top (several fully-connected layers). For details see
    [LeCun95]_.

    .. [LeCun95] LeCun, Yann, et al.
       *Comparison of learning algorithms for handwritten digit
       recognition.*,
       International conference on artificial neural networks. Vol. 60.

    Parameters
    ----------
    conv_activations : list of :class:`.Brick`
        Activations for convolutional network.
    num_channels : int
        Number of channels in the input image.
    image_shape : tuple
        Input image shape.
    filter_sizes : list of tuples
        Filter sizes of :class:`.blocks.conv.ConvolutionalLayer`.
    feature_maps : list
        Number of filters for each of convolutions.
    pooling_sizes : list of tuples
        Sizes of max pooling for each convolutional layer.
    top_mlp_activations : list of :class:`.blocks.bricks.Activation`
        List of activations for the top MLP.
    top_mlp_dims : list
        Numbers of hidden units and the output dimension of the top MLP.
    conv_step : tuples
        Step of convolution (similar for all layers).
    border_mode : str
        Border mode of convolution (similar for all layers).

    """
    def __init__(self, conv_activations, num_channels, image_shape,
                 filter_sizes, feature_maps, pooling_sizes,
                 top_mlp_activations, top_mlp_dims,
                 conv_step=None, border_mode='valid',input_tensor=None, **kwargs):
        if conv_step is None:
            self.conv_step = (1, 1)
        else:
            self.conv_step = conv_step
        self.num_channels = num_channels
        self.image_shape = image_shape
        self.top_mlp_activations = top_mlp_activations
        self.top_mlp_dims = top_mlp_dims
        self.border_mode = border_mode

        conv_parameters = [(i, j) for i in filter_sizes for j in feature_maps]

        # Construct convolutional layers with corresponding parameters
        self.layers = []
        self.conv_sequence = []
        for i, (filter_size, num_filter) in enumerate(conv_parameters):

            self.layers.append(list([
                (Convolutional(filter_size=filter_size,
                           num_filters=num_filter,
                           step=self.conv_step,
                           border_mode=self.border_mode,
                           name='conv_{}'.format(i))
                ),
                conv_activations[0],
                MaxPooling(pooling_size=pooling_sizes[i], name='pool_{}'.format(i))]))


            self.conv_sequence.append(ConvolutionalSequence(self.layers[i], num_channels,
                                                   image_size=image_shape, name='conv_seq_{}'.format(i)))


        conv_applications = [conv.apply for conv in self.conv_sequence]

        # Construct a top MLP
        self.top_mlp = MLP(top_mlp_activations, top_mlp_dims)

        # We need to flatten the output of the last convolutional layer.
        # This brick accepts a tensor of dimension (batch_size, ...) and
        # returns a matrix (batch_size, features)
        self.flattener = Flattener()

        merged_conv = tensor.concatenate([conv.apply(input_tensor) for conv in self.conv_sequence])
        mlp_input_tensor = self.flattener.apply(merged_conv)

        self.top_mlp.apply(mlp_input_tensor)

        application_methods = conv_applications + [self.flattener.apply, self.top_mlp.apply]
        # super(LeNet, self).__init__(application_methods, **kwargs)

    @property
    def output_dim(self):
        return self.top_mlp_dims[-1]

    @output_dim.setter
    def output_dim(self, value):
        self.top_mlp_dims[-1] = value

    def _push_allocation_config(self):
        [conv._push_allocation_config() for conv in self.conv_sequence]
        conv_dimensions = [conv.get_dim('output') for conv in self.conv_sequence]
        conv_out_dim = (300, 1, 1) #sum(conv_dimensions)

        self.top_mlp.activations = self.top_mlp_activations
        self.top_mlp.dims = [numpy.prod(conv_out_dim)] + self.top_mlp_dims


def main(save_to, num_epochs, feature_maps=None, mlp_hiddens=None,
         conv_sizes=None, pool_sizes=None, batch_size=100,
         num_batches=None):
    if feature_maps is None:
        feature_maps = [100]
    if mlp_hiddens is None:
        mlp_hiddens = [100]
    if conv_sizes is None:
        conv_sizes = [(3, 300), (4, 300), (5, 300)]
    if pool_sizes is None:
        pool_sizes = [(61-3+1, 1), (61-4+1, 1), (61-5+1, 1)]
    image_size = (61, 300)
    output_size = 2

    x = tensor.tensor4('features')
    y = tensor.lmatrix('targets')

    # Use ReLUs everywhere and softmax for the final prediction
    conv_activations = [Rectifier() for _ in feature_maps]
    mlp_activations = [Rectifier() for _ in mlp_hiddens] + [Softmax()]
    convnet = LeNet(conv_activations, 1, image_size,
                    filter_sizes=conv_sizes,
                    feature_maps=feature_maps,
                    pooling_sizes=pool_sizes,
                    top_mlp_activations=mlp_activations,
                    top_mlp_dims=mlp_hiddens + [output_size],
                    border_mode='valid',
                    weights_init=Uniform(width=.2),
                    biases_init=Constant(0),
                    input_tensor=x)
    # We push initialization config to set different initialization schemes
    # for convolutional layers.
    convnet.push_initialization_config()
    # convnet.layers[0].weights_init = Uniform(width=.2)
    # convnet.layers[1].weights_init = Uniform(width=.09)
    convnet.top_mlp.linear_transformations[0].weights_init = Uniform(width=.08)
    convnet.top_mlp.linear_transformations[1].weights_init = Uniform(width=.11)
    convnet.initialize()
    logging.info("Input dim: {} {} {}".format(
        *convnet.children[0].get_dim('input_')))
    for i, layer in enumerate(convnet.layers):
        if isinstance(layer, Activation):
            logging.info("Layer {} ({})".format(
                i, layer.__class__.__name__))
        else:
            logging.info("Layer {} ({}) dim: {} {} {}".format(
                i, layer.__class__.__name__, *layer.get_dim('output')))


    # Normalize input and apply the convnet
    probs = convnet.apply(x)
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
        step_rule=Scale(learning_rate=0.1))
    # `Timing` extension reports time for reading data, aggregating a batch
    # and monitoring;
    # `ProgressBar` displays a nice progress bar during training.
    extensions = [Timing(),
                  FinishAfter(after_n_epochs=num_epochs,
                              after_n_batches=num_batches),
                  DataStreamMonitoring(
                      [cost, error_rate],
                      sst2_test_stream,
                      prefix="test"),
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
    parser.add_argument("--num-epochs", type=int, default=100,
                        help="Number of training epochs to do.")
    parser.add_argument("save_to", default="sst2.pkl", nargs="?",
                        help="Destination to save the state of the training "
                             "process.")
    parser.add_argument("--feature-maps", type=int, nargs='+',
                        default=[100], help="List of feature maps numbers.")
    parser.add_argument("--mlp-hiddens", type=int, nargs='+', default=[100],
                        help="List of numbers of hidden units for the MLP.")
    parser.add_argument("--conv-sizes", type=int, nargs='+', default=[(3, 300)],
                        help="Convolutional kernels sizes. The kernels are "
                        "always square.")
    parser.add_argument("--pool-sizes", type=int, nargs='+', default=[(61+3-1, 1)],
                        help="Pooling sizes. The pooling windows are always "
                             "square. Should be the same length as "
                             "--conv-sizes.")
    parser.add_argument("--batch-size", type=int, default=100,
                        help="Batch size.")
    args = parser.parse_args()
    # main(**vars(args))
    main('sst2_100_with_dropout.pkl', 100)
