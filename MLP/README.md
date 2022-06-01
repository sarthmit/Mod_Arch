## MLP Architectural Tasks
___
### Motivation
This task is based on simple feed-forward structures in both the models as well as the data. The data distribution in this setting can simply be described as task conditioned linear problem. We refer the readers to the sub-directories for the Regression and Classification settings respectively.

---
### Running Experiments
To train a model in the respective sub-directories, run the following command:

```
python main.py --gt-rules {Rules} --data-seed {DSeed} --model {Model} --num-rules {Rules} --encoder-dim {Enc} --dim {Dim} --seed {Seed} [--op --joint]
```
where 

- The number of rules in the data and the model can be specified via `Rules`.
- The seed for data, which determines a specific instantiation of a task, can be specified through `DSeed`.
- The encoding dimension in the model can be specified through `Enc`. The encoder is shared between the different modules in modular systems.
- The dimension of the specific models (Monolithic, Modular, ...) can be specified through `Dim`.
- The type of the model can be specified through `Model`, and can be Monolithic, Modular or GT_Modular.
- The seed for the model can be specified via `Seed`.

To run a Modular-op system, one needs to mention the `--op` flag with `--model Modular`, and for training a typical end to end modular system, the flag `--joint` with `--model Modular` needs to be present for each module to provide a score for its computation.
___
### Experimentation Setup

We train four different kinds of models `Modular/Modular-op/GT-Modular/Monolithic` through the above-mentioned script. We consider the data and model seeds in the range 0-5, leading to a total of 25 runs. Further, we consider encoding dimension in the set `{32, 64, 128, 256, 512}`, and the dimensionality of the model is generally set at `2 x Enc`. We also perform our analysis on the number of rules ranging from `{2, 4, 8, 16, 32}`, and all of this is done for both the regression and classification settings.

Having trained the models, we use the following files for evaluation:

- `perf.py` is used to evaluate a trained model and output the obtained performance metrics under the model directory. It takes similar arguments as `main.py`.
- `spec.py` is used to compute the Adaptation metric of our paper. It computes the adaptation metric and logs it under the model directory. It also takes similar arguments as `main.py`
- `prob.py` is used to compute the activation probability matrix, which will be used to compute the other downstream metrics. Similar arguments as `main.py`
- `metrics.py` is used once `prob.py` has already stored the activation matrix. This will result in additional logs containing the different collapse metrics as well as the normalized mutual information metric. It runs over all the trained models, and thus doesn't need model- or data-specific parameters as it pulls them based on the directory names.
- `hungarian.py` is also used once `prob.py` has already stored the activation matrix. This will result in logs of Alignment metrics to be saved, and again doesn't require additional arguments as it runs over the saved directories.

The above scripts can also be provided the flag `--best` to perform evaluation on the version of the model with best validation performance. This doesn't impact results much as we are in the infinite data regime. Having run the above evaluation protocols, `compute.py` is run which aggregates all the metrics in a dataframe and saves it for visualization and for plots.