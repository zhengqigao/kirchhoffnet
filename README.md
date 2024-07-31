# kirchhoffnet

### Introduction

This repo contains the code for our paper titled: [KirchhoffNet: A Scalable Ultra Fast Analog Neural Network](https://arxiv.org/pdf/2310.15872), accepted by the top Electronic Design Automation (EDA) conference, ICCAD'24. In one sentence, we argue an interesting fact that an analog integrated circuit mathematically corresponds to a [Neural ODE](https://arxiv.org/pdf/1806.07366). Thus, if we fabricate such an analog circuit as a specific hardware acclerator for Neural ODE, it can run extremely fast (compared to running Neural ODE on GPUs). 

This is the first work along this line. In the future, we will attempt various aspects and wish for a real hardware tape out.

### Barebone Implementation

Our code developed initially can be improved in many ways. For this purpose, we provide a barebone and clean implementation in the `bare` folder, containing less than 100 lines of code. We highily recommend people who wish to learn more to start there. It is very straightforward and can be understood in 30 mins. Moreover, to ease the understanding, we provide another README file under the `bare` folder. Please refer to it for more details about the barebone implementation.

### Reproduce Results

If you wish to reproduce the results reported in the paper, please refer to the `src` folder. The code under that haven't been fully cleaned up, but should be clear on each part's functionality. To justify our results, we have stored the trained models, so that users can directly run the inference procedure to double confirm the numbers we reported in our paper. To do that, users can change the directory to the `src` folder, and use our provided trained models (under `src/results/xxx`) to reproduce the results for generation and density matching in our paper, by using the inference script: 

```shell
python inference_gen_den.py --exp_name xxx --gpu 0
```

Here `xxx` can be `genmnist`, `2spirals`, `8guassians`, `pinwheel`, `swissroll`, `twomoon`, `potential1`, `potential2`. Additionally, training such a model is also straightforward using the `main_gen.py` for generation, and `main_den.py` for density matching. Take twomoon generation as an example, we can use the following command:

```shell
python main_gen.py --config_path ./configs/config_twomoon.yaml --gpu 0
```

As users might notice, inside `src/results/twomoon`, we provide the `config.yaml` file, which was used to train the model stored there, this config file is exactly the same as `src/configs/config_twomoon.yaml`. Users can also use the `config.yaml` under other `src/results/xxx` as the commandline argument for the script `main_gen.py` and `main_den.py` to re-run the training procedure.

### Citation

Please cite our paper if you use it in your research:

```bibtex
@misc{gao2024kirchhoffnetscalableultrafast,
      title={KirchhoffNet: A Scalable Ultra Fast Analog Neural Network}, 
      author={Zhengqi Gao and Fan-Keng Sun and Ron Rohrer and Duane S. Boning},
      year={2024},
      eprint={2310.15872},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2310.15872}, 
}
```