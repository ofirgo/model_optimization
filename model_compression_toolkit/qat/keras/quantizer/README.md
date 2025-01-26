# Training Methods for QAT

## Introduction

Several training methods may be applied by the user to train the QAT ready model
created by `keras_quantization_aware_training_init` method in [`keras/quantization_facade`](../quantization_facade.py).
Each `TrainingMethod` (an enum defined in the [`qat_config`](../../common/qat_config.py)) 
and `QuantizationMethod`
selects a quantizer for weights and a quantizer for activations.

Currently, only the STE (straight through estimator) training method is implemented by the MCT.

## Make your own training method

Follow these steps in order to set the quantizers required by your training method:
- Add your `TrainingMethod` enum in [`qat_config`](../../common/qat_config.py).
- Add your quantizers for weights and activation as explained in [quantizer readme](../../../trainable_infrastructure/keras).
- Import your quantizer package in the quantizer [`__init.py__`](./__init__.py) file.
- Set your `TrainingMethod` in the `QATConfig` and generate the QAT ready model for training. 

   
## Example: Adding a new training method

In this example we'll add a new quantization method, called MTM (my training method).

First, we update the `TrainingMethod` enum in [`qat_config`](../../common/qat_config.py)
```python
class TrainingMethod(Enum):
    """
    An enum for selecting a QAT training method

    STE - Standard straight-through estimator. Includes PowerOfTwo, symmetric & uniform quantizers
    MTM - MyTrainingMethod.
    """
    STE = 0
    MTM = 1
```

Then we implement a weight quantizer class that implements the desired training scheme: MTMWeightQuantizer,
under a new package in `qat/keras/quantizer/mtm_quantizer/mtm.py`, and import it in the quantizer `__init__.py` file.

```python
import model_compression_toolkit.qat.keras.quantizer.mtm_quantizer.mtm
```

Finally, we're ready to generate the model for quantization aware training
by calling `keras_quantization_aware_training_init` method in [`keras/quantization_facade`](../quantization_facade.py)
with the following [`qat_config`](../../common/qat_config.py):

```python
from model_compression_toolkit.qat.common.qat_config import QATConfig, TrainingMethod

qat_config = QATConfig(weight_training_method=TrainingMethod.MTM)
```
