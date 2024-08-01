## KirchhoffNet: A Scalable Ultra Fast Analog Neural Network

### Introduction

This repo contains the code for our paper titled: [KirchhoffNet: A Scalable Ultra Fast Analog Neural Network](https://arxiv.org/pdf/2310.15872), accepted by the top Electronic Design Automation (EDA) conference, ICCAD'24. In one sentence, we argue an interesting fact that an analog integrated circuit mathematically corresponds to a [Neural ODE](https://arxiv.org/pdf/1806.07366). Thus, if we fabricate such an analog circuit as a specific hardware acclerator for Neural ODE, it can run extremely fast (compared to running Neural ODE on GPUs). This is the first work in this area. In the future, we aim to explore various aspects and pursue real hardware tape-outs.

### Prerequisite

You need to have [`pytorch`](https://pytorch.org/get-started/locally/) and [`torchdiffeq`](https://github.com/rtqichen/torchdiffeq) installed; please use the embedded links for instructions on installations. Next, clone the code to your local machine. To understand how we build the ODE right-hand side, we suggest readers to understand Kirchhoff current law and basic RC circuits.

### Barebone Implementation

Our initial code can be improved in many ways. To facilitate this, we provide a barebone and clean implementation in the `bare` folder, containing less than 100 lines of code. We highly recommend that those who wish to learn more start here. It is straightforward and can be understood in about 30 minutes. Additionally, for better understanding, we provide another README file under the `bare` folder with more details about the barebone implementation.


### Reproduce Results

To reproduce the results reported in the paper, please refer to the `src` folder. The code in this folder has not been fully cleaned up but should be clear regarding each part's functionality. To validate our results, we have stored the trained models, allowing users to directly run the inference procedure to confirm the numbers reported in our paper. To do this, change the directory to the `src` folder and use the provided trained models (under `src/results/xxx`) to reproduce the results for generation and density matching in our paper by running the inference script:

```shell
python inference_gen_den.py --exp_name xxx --gpu 0
```

Here `xxx` can be `genmnist`, `2spirals`, `8guassians`, `pinwheel`, `swissroll`, `twomoon`, `potential1`, `potential2`. Additionally, training such a model is also straightforward using the `main_gen.py` for generation, and `main_den.py` for density matching. Take twomoon generation as an example, we can use the following command:

```shell
python main_gen.py --config_path ./configs/config_twomoon.yaml --gpu 0
```

As users might notice, inside `src/results/twomoon`, we provide the `config.yaml` file, which was used to train the model stored there. This config file is exactly the same as `src/configs/config_twomoon.yaml`. Users can also use the `config.yaml` under other `src/results/xxx` as the commandline argument for the script `main_gen.py` and `main_den.py` to re-run the training procedure.

Similarly, we also provide trained model checkpoints for image classification task. But first please revise the data path in line 22-28 in `src/utils/testbench.py` for the program to successfully locate the data. Afterwards, 

```shell
python inference_image_classify.py --exp_name xxx --gpu 0
```
Here `xxx` can be `mnist`, `svhn`, and `cifar10`. The `config.yaml` file under `src/results/xxx` can be used along with `main_image_classify.py` to do training.

### Remarks

We have been asked a few questions frequently, please refer to this [Q&A](https://zhengqigao.github.io/articles/what_is_kirchhoffnet.pdf) for some common questions and our ansers. Also, here is a one-page summary of [our work](https://zhengqigao.github.io/articles/kirchhoffnet.pdf). There are many future works we want to explore, such as power and area of such an analog integrated circuit. Also, we want to redo this work based on commercial simulators such as Hspice and Spectre. We are also considering fabricating a real hardware. We are always looking for collaborators on this topic.


### Citation

At this moment, the ICCAD proceeding is not publicly available. Please cite our Arxiv paper if you use it in your research:

```bibtex
@misc{gao2024kirchhoffnet,
      title={KirchhoffNet: A Scalable Ultra Fast Analog Neural Network}, 
      author={Zhengqi Gao and Fan-Keng Sun and Ron Rohrer and Duane S. Boning},
      year={2024},
      eprint={2310.15872},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2310.15872}, 
}
```