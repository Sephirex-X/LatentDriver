import torch.nn as nn
from typing import Callable
from typing import Optional
import torch.nn as nn
def sequential_pack(layers: list) -> nn.Sequential:
    r"""
    Overview:
        Pack the layers in the input list to a `nn.Sequential` module.
        If there is a convolutional layer in module, an extra attribute `out_channels` will be added
        to the module and set to the out_channel of the conv layer.
    Arguments:
        - layers (:obj:`list`): the input list
    Returns:
        - seq (:obj:`nn.Sequential`): packed sequential container
    """
    assert isinstance(layers, list)
    seq = nn.Sequential(*layers)
    for item in reversed(layers):
        if isinstance(item, nn.Conv2d) or isinstance(item, nn.ConvTranspose2d):
            seq.out_channels = item.out_channels
            break
        elif isinstance(item, nn.Conv1d):
            seq.out_channels = item.out_channels
            break
    return seq

def build_normalization(norm_type: str, dim: Optional[int] = None) -> nn.Module:
    r"""
    Overview:
        Build the corresponding normalization module
    Arguments:
        - norm_type (:obj:`str`): type of the normaliztion, now support ['BN', 'LN', 'IN', 'SyncBN']
        - dim (:obj:`int`): dimension of the normalization, when norm_type is in [BN, IN]
    Returns:
        - norm_func (:obj:`nn.Module`): the corresponding batch normalization function

    .. note::
        For beginers, you can refer to <https://zhuanlan.zhihu.com/p/34879333> to learn more about batch normalization.
    """
    if dim is None:
        key = norm_type
    else:
        if norm_type in ['BN', 'IN']:
            key = norm_type + str(dim)
        elif norm_type in ['LN', 'SyncBN']:
            key = norm_type
        else:
            raise NotImplementedError("not support indicated dim when creates {}".format(norm_type))
    norm_func = {
        'BN1': nn.BatchNorm1d,
        'BN2': nn.BatchNorm2d,
        'LN': nn.LayerNorm,
        'IN1': nn.InstanceNorm1d,
        'IN2': nn.InstanceNorm2d,
        'SyncBN': nn.SyncBatchNorm,
    }
    if key in norm_func.keys():
        return norm_func[key]
    else:
        raise KeyError("invalid norm type: {}".format(key))



def MLP(
    in_channels: int,
    hidden_channels: int,
    out_channels: int,
    layer_num: int,
    layer_fn: Callable = None,
    activation: nn.Module = None,
    norm_type: str = None,
    use_dropout: bool = False,
    dropout_probability: float = 0.5,
    output_activation: bool = True,
    output_norm: bool = True,
    last_linear_layer_init_zero: bool = False
):
    r"""
    Overview:
        create a multi-layer perceptron using fully-connected blocks with activation, normalization and dropout,
        optional normalization can be done to the dim 1 (across the channels).
        x -> fc -> norm -> act -> dropout -> out
    Arguments:
        - in_channels (:obj:`int`): Number of channels in the input tensor.
        - hidden_channels (:obj:`int`): Number of channels in the hidden tensor.
        - out_channels (:obj:`int`): Number of channels in the output tensor.
        - layer_num (:obj:`int`): Number of layers.
        - layer_fn (:obj:`Callable`): Layer function.
        - activation (:obj:`nn.Module`): The optional activation function.
        - norm_type (:obj:`str`): The type of the normalization.
        - use_dropout (:obj:`bool`): Whether to use dropout in the fully-connected block.
        - dropout_probability (:obj:`float`): The probability of an element to be zeroed in the dropout. Default: 0.5.
        - output_activation (:obj:`bool`): Whether to use activation in the output layer. If True,
            we use the same activation as front layers. Default: True.
        - output_norm (:obj:`bool`): Whether to use normalization in the output layer. If True,
            we use the same normalization as front layers. Default: True.
        - last_linear_layer_init_zero (:obj:`bool`): Whether to use zero initializations for the last linear layer
            (including w and b), which can provide stable zero outputs in the beginning,
            usually used in the policy network in RL settings.
    Returns:
        - block (:obj:`nn.Sequential`): a sequential list containing the torch layers of the fully-connected block.

    .. note::

        you can refer to nn.linear (https://pytorch.org/docs/master/generated/torch.nn.Linear.html).
    """
    assert layer_num >= 0, layer_num
    if layer_num == 0:
        return sequential_pack([nn.Identity()])

    channels = [in_channels] + [hidden_channels] * (layer_num - 1) + [out_channels]
    if layer_fn is None:
        layer_fn = nn.Linear
    block = []
    for i, (in_channels, out_channels) in enumerate(zip(channels[:-2], channels[1:-1])):
        block.append(layer_fn(in_channels, out_channels))
        if norm_type is not None:
            block.append(build_normalization(norm_type, dim=1)(out_channels))
        if activation is not None:
            block.append(activation)
        if use_dropout:
            block.append(nn.Dropout(dropout_probability))

    # The last layer
    in_channels = channels[-2]
    out_channels = channels[-1]
    block.append(layer_fn(in_channels, out_channels))
    """
    In the final layer of a neural network, whether to use normalization and activation are typically determined
    based on user specifications. These specifications depend on the problem at hand and the desired properties of
    the model's output.
    """
    if output_norm is True:
        # The last layer uses the same norm as front layers.
        if norm_type is not None:
            block.append(build_normalization(norm_type, dim=1)(out_channels))
    if output_activation is True:
        # The last layer uses the same activation as front layers.
        if activation is not None:
            block.append(activation)

    if last_linear_layer_init_zero:
        # Locate the last linear layer and initialize its weights and biases to 0.
        for _, layer in enumerate(reversed(block)):
            if isinstance(layer, nn.Linear):
                nn.init.zeros_(layer.weight)
                nn.init.zeros_(layer.bias)
                break

    return sequential_pack(block)