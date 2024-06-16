# PANNs AT (Audio Tagging) inference

**epanns_inference** provides an easy to use Python interface to audio tagging models from E-PANNS: Sound Recognition using Efficient Pre-Trained Audio Neural Networks: https://github.com/Arshdeep-Singh-Boparai/E-PANNs

## Installation
PyTorch>=1.0 is required.
```
$ pip install epanns_inference
```

## Usage
For example (CUDA available):

```
from epanns_inference import models

model = models.Cnn14_pruned(pre_trained=True)

with torch.inference_mode():
    model.cuda()
    result_dict = model.eval()(x_in.float().cuda())
```

## References
[1] Arshdeep Singh, Haohe Liu and Mark D PLumbley, "E-PANNS: Sound Recognition using Efficient Pre-Trained Audio Neural Networks", accepted in Internoise 2023.

[2] Singh, Arshdeep, and Mark D. Plumbley. "Efficient CNNs via Passive Filter Pruning." arXiv preprint arXiv:2304.02319 (2023).

[3] Official GitHub repository: https://github.com/Arshdeep-Singh-Boparai/E-PANNs
