## Is a Modular Architecture Enough?
___
This repository contains the official implementation for the paper **[Is a Modular Architecture Enough?](to-do)**


<details>
  <summary>
    <b>Abstract</b>
  </summary>
    Inspired from human cognition, machine learning systems are gradually revealing advantages of sparser and more modular architectures. Recent work demonstrates that not only do some modular architectures generalize well, but they also lead to better out-of-distribution generalization, scaling properties, learning speed, and interpretability. A key intuition behind the success of such systems is that the data generating system for most real-world settings is considered to consist of sparsely interacting parts, and endowing models with similar inductive biases will be helpful. However, the field has been lacking in a rigorous quantitative assessment of such systems because these real-world data distributions are complex and unknown. In this work, we provide a thorough assessment of common modular architectures, through the lens of simple and known modular data distributions. We highlight the benefits of modularity and sparsity and reveal insights on the challenges faced while optimizing modular systems. In doing so, we propose evaluation metrics that highlight the benefits of modularity, the regimes in which these benefits are substantial, as well as the sub-optimality of current end-to-end learned modular systems as opposed to their claimed potential.
</details>

---
Our work consists of analysis done on different types of domains, based on simple feedforward settings to more complex recurrent sequence prediction or set-based problems. Corresponding to each of these settings, we also consider a further sub-division into regression and classification problems for an unbiased analysis of modular systems.

---

We refer the readers to the respective sub-directories for details regarding each of the experiments. Once the sub-directory experiments are done, we use the `plot_main.py` file to obtain the plots for the main paper, and `plot.py` to give additional plots that are present in the Appendix. We also use the `plot_heatmap.py` to provide an illustrative visualization of the collapse problem, but one can easily make this plot from actual trained models too once the activation probability matrix has been computed, details of which are present in the sub-directories.

Please do cite our work if you build up on it or find it useful and feel free to create an issue or contact me at `sarthmit@gmail.com` in case of any questions.

```
todo
```