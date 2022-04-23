[![PyPi Version](https://img.shields.io/pypi/v/mbrl)](https://pypi.org/project/mbrl/)
[![Master](https://github.com/facebookresearch/mbrl-lib/workflows/CI/badge.svg)](https://github.com/facebookresearch/mbrl-lib/actions?query=workflow%3ACI)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/facebookresearch/mbrl-lib/tree/master/LICENSE)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
 
## Value Gradient weighted Model-Based Reinforcement Learning.

This is the official code for VAGRAM published at ICLR 2022.  
The code framework builds on [MBRL-lib](https://github.com/facebookresearch/mbrl-lib)

### Experiments

To run the experiments presented in the paper, install the required libraries found in `requirements.txt` and use the `vagram/mbrl/examples/main.py` script provided by mbrl-lib.

The exact settings for the hopper experiments can be found in `vagram/scripts`:

Distraction (2nd cmd parameter sets the number of distracting dimensions):
```
python3 -m mbrl.examples.main \
	seed=$1 \
	algorithm=mbpo \
	overrides=mbpo_hopper_distraction \
	overrides.num_steps=500000 \
	overrides.model_batch_size=1024 \
	overrides.distraction_dimensions=$2
```

Reduced model size (num_layers sets the model size):
```
python3 -m mbrl.examples.main \
	seed=$RANDOM \
	algorithm=mbpo \
	overrides=mbpo_hopper \
	dynamics_model.model.num_layers=3 \
	dynamics_model.model.hid_size=64 \
	overrides.model_batch_size=1024
```

To use MSE/MLE instead of VaGraM, run:

```
python3 -m mbrl.examples.main \
	seed=$1 \
	algorithm=mbpo \
	overrides=mbpo_hopper_distraction \
	overrides.num_steps=500000 \
	overrides.model_batch_size=256 \
	dynamics_model=gaussian_mlp_ensemble \
	overrides.distraction_dimensions=$2
```

### Using VaGraM

The core implementation of the VaGraM algorithm can be found in `vagram/mbrl/models/vaml_mlp.py`. The code offers three variants, one for IterVAML, on for the unbounded VaGraM objective and finally the bounded VaGraM objective used in the paper. THe default configuration used in all experiments can be found in `vagram/mbrl/examples/conf/dynamics_model/vaml_ensemble.yaml`.

In addition to the implementation details in the paper, we introduced a cache for the computed value function gradients. This does not change any detail of the optimization, but saves gradients of the state samples until the value function is updated for faster computation.

## Citing
If you use this project in your research, please cite:

```BibTeX
@inproceedings{voelcker2022vagram,
  title={{Value Gradient weighted Model-Based Reinforcement Learning}}, 
  author={Claas A Voelcker and Victor Liao and Animesh Garg and Amir-massoud Farahmand},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2022},
  url={https://openreview.net/forum?id=4-D6CZkRXxI}
}
```

## License
`VaGRAM` is released under the MIT license. See [LICENSE](LICENSE) for additional details about it. 
