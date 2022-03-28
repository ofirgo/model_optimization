import keras.layers
from keras.engine.input_layer import InputLayer
from tensorflow_model_optimization.python.core.quantization.keras.graph_transformations import transforms

from tensorflow_model_optimization.python.core.quantization.keras.quantize_wrapper import QuantizeWrapper

from model_compression_toolkit.keras.quantizer.mixed_precision.quantization_config_factory import \
  quantization_config_builder_mixed_precision
from model_compression_toolkit.keras.quantizer.mixed_precision.selective_quantize_config import SelectiveQuantizeConfig


class InputLayerQuantizeTransform(transforms.Transform):
    """
    Quantizes InputLayer, by adding QuantizeLayer after it.
    InputLayer => InputLayer -> QuantizeLayer
    """

    def __init__(self, input_layer, fw_info):
        super(InputLayerQuantizeTransform, self).__init__()

        self.input_layer = input_layer
        self.fw_info = fw_info
        self.name = self.input_layer.name

    def pattern(self):
        return transforms.LayerPattern('InputLayer')

    def replacement(self, match_layer):
        quant_layer = QuantizeWrapper(InputLayer(input_shape=self.input_layer.input_shape),
                                      quantization_config_builder_mixed_precision(self.input_layer, self.fw_info))
        layer_config = keras.layers.serialize(quant_layer)
        layer_config['name'] = f"quant_{self.name}"
        layer_config['config']['name'] = f"quant_{self.name}"

        quant_layer_node = transforms.LayerNode(
          layer_config,
          input_layers=[match_layer])

        return quant_layer_node

    def custom_objects(self):
        return {
          'QuantizeWrapper': QuantizeWrapper,
          'SelectiveQuantizeConfig': SelectiveQuantizeConfig,
        }