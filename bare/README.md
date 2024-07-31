
This folder provides a barebone implementation of the key concepts in KirchhoffNet. To get started, ensure you have `pytorch` and `torchdiffeq` installed. Change your directory to the current folder `bare` and run the command `python example.py`. By tracing the code execution, users can gain insights into how an analog integrated circuit can be utilized as a Neural ODE.

#### Key Files

- **`device.py`**: This script implements the nonlinear I-V relationships of a device in analog integrated circuits, corresponding to the function g in our paper. Note that this implementation represents an ideal case, as we abstract the I-V relation using Pytorch. In practical scenarios, the device I-V relationship would first need to be simulated using commercial circuit simulators like SPICE or Spectre, which is our future work.

- **`circuit.py`**: The `forward` function in this script is crucial. Given a topology (represented as a 2-by-N matrix) that describes how devices are connected among nodes, it builds the ODE right-hand side based on Kirchhoff's current law (KCL). Our paper specifies that a fixed and unlearnable capacitor is connected to every node; however, this detail has been simplified in the code (with the capacitance value set to 1). The `forward` function in `circuit.py` concretely implements the right-hand side of the ODE in Eq. (7) of our paper.

#### Practical Concerns

To make such a KirchhoffNet really work on complex ML tasks, it is crucial to add the following ingredients:

1) A well-designed topology matrix, such as those depicted in our paper (fully connected, neighbor emphasizing, etc.)
2) Increase the depth. See the difference between Fig. 1 and Fig. 2 of our paper to understand the concept of 'depth' in our context.
3) For two specific nodes (e.g., n1 and n2), connect multiple devices between them.

We have provided our code implementing these points in the other folder 'src', which can be used to fully recover the results shown in our paper. Our choices might not be optimal, and users are **highly** suggested to consider implementation variations on the above three aspects to attain better performance (and beat us!).