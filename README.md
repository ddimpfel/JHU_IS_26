# Experimentation and Analysis on Mixture of Expert Modeling Applications in Continual Learning

This repository contains all code and the latest results for model training and evaluation in a Data Science Independent Study at Johns Hopkins during the 2026 Summer semester.

## Study

The primary goal of the study is to evaluate varying submodel types in a Mixture of Experts (MoE) architecture, particularly how well they perform in a continual learning scenario. A class-incremental learning scenario is artificially induced by splitting 9 classes in the GTSRB dataset into 3 training tasks. To see the full class breakdown and eda, view the <**insert notebook with class breakdown**>.

A secondary goal is evaluating the introduction of a joint embedding backbone as the generalist to predict latent space vectors for routers and experts to predict with.

Specifically, the study hypothesizes:

1. Layered routers are more resilient to expert collapse and more reliable for routing tasks than MLP routers.
2. Joint embedding models learn a separable latent space between GTSRB classes and across tasks, creating reliable embedding vectors.
3. MoE models leveraging a joint embedding backbone have enhanced prediction performance and reliability when compared to a traditional CNN backbone MoE.

## Files

### `continual_learning.py`: the continual learning engine

This file holds the entire setup for the continual learning scenario. It implements a PyTorch wrapper class with all state and utilities required for training a class-incremental learning model in `CILComputerVisionModel`. It can accomodate any typical computer vision model or MoE models. The core continual learning model setup includes:

- An `exemplar_set` of previously seen images, randomly sampled at some `exemplar_ratio` defined on class instantiation.
- Pre-initialization of the number of classes expected to see over the lifetime of the model. This is accompanied with a class mask to ensure the model does not learn to predict classes that have not been encountered yet.
  - The purpose of preinitialization is to avoid complexity when needed to extend the model during pseudo-task training.
- Knowledge divergence loss utilizing the prior task as a frozen teacher model to retain performance on the previously seen classes.

Utilities in the file include:

- Continual learning metrics, such as setup and evaluation of the evaluation matrix:
  - `average_accuracy`, `task_forgetting`, `average_forgetting`, `backward_transfer`, and `forward_transfer`.
- Bootstrap statistical testing utilizing these metrics with `bootstrap_learning`, `bootstrap_learning_diff`, and `bootstrap_performance_diff`.
- The full training and evaluation pipeline in `train_and_evaluate_cil_model`:
  - Sets up evaluation matrix, metrics, and datasets for training/evaluation.
  - Iterates over pseudo-time training tasks.
  - Decreases learning rate dynamically after task 1 by maintaining a reduced lr \* 0.1.
  - Gathers all results, loss histories, and predictions for later visualization and/or further statistical testing.

### `gre_model_base.py`: the core mixture of experts model setup

The `GeneralistRouterExperts` PyTorch module is the primary engine for implementing the MoE models experimented with. It's core MoE features are:

- Hard routed expert classifications: the router will always choose $k$ experts (typically 2) for predicting on the generalists features. This reduces training and inference time compute by only requiring a subset of model parameters to be used during prediction.
- Router auxiliary loss to mitigate the possibility for expert collapse.
- Post-training tuning for reliability using a learnable temperature variable.

The prediction pipeline is:

`Generalist(x) --> generalist prediction logits, features --> Router(features) --> Expert selection logits, router-expert selection probabilities -(1-k)-> Expert(features) --> k-expert prediction logits`

The final prediction is: `generalist prediction logits` + `router-expert selection probabilities` \* `k-expert prediction logits`

In this setup, the generalist still maintains the primary predictive capability as it is the heaviest model capable of extracting features. The experts are utilized as post generalist prediction fine-tuning so a large pool of predictive capacity is available.

The additional utilities in this file include:

- MoE specific metrics: `expert_metrics` (expected expert calls (always == k in hard-routing) and expert selection ratio (observers expert collapse)), `cost_proxy`, and `router_entropy`.
  - Expert selection ratio and router entropy can be used to infer if the model architecture is behaving well based on balancing the experts specialization along with balancing using multiple experts for k > 1.
  - Normalized router entropy near 1.0 may indicate the router is randoming guessing where to send inputs.
- `calibrate_temperature` is also included to provide fine-tuning of the model's predicted probabilities for a given input so they better align with it's accuracy aligns more closely with it's confidence.
  - `expected_calibration_error` is used to analyze how well calibrated the model's confidence is to it's accuracy.

1. [What is MoE?](https://www.ibm.com/think/topics/mixture-of-experts)
2. [Adaptive Mixtures of Local Experts](https://www.cs.toronto.edu/~fritz/absps/jjnh91.pdf)
3. [Sparsely Gated MoE](https://arxiv.org/pdf/1701.06538)

### `joint_embedding.py`: the joint embedding backbone model setup

This contains the model wrapper for creating the joint embedding model, including the loss functions experimented with.

- The joint embedding model takes in one of the backbone CNNs experimented with to extract features for projection into a learned latent space (currently supports `mobilenet` and `convnext` only).
  - It extracts the features, embeds them in a lower dimensional space, the learns to project them into a "shared" latent space (this would typically be shared between modalities, but is shared between classes in this case).
  - It also makes it's own predictions on the vector embeddings to provide the required structure for the mixture of experts model setup.
- The loss function used is `SupervisedContrastiveLoss`, which can compare many positive and negative class samples at once to learn the shared latent space.
  - It is essentially an extension of InfoNCE loss to utilize multiple positive data poitns at once.
  - The positive samples in a training batch are pushed together in the shared latent space, while the negative samples are pushed away from the positive centroid.

Ideally, a strongly separated learned latent space will allow for high classification accuracy using just the MLP experts, as the CNN joint embedding backbone is doing the heavy lifting of clustering the classes and reducing noise.

The use of the exemplar set from the continual learning setup may allow the JE backbone to anchor the shared latent space centroids or result in overfitting.

1. [VL-JEPA](https://arxiv.org/pdf/2512.10942)
2. [Self-Supervised JEPA Learning from Images](https://arxiv.org/pdf/2301.08243)

### `model_experiments.ipynb`: the primer experiments between non-joint embedding augmented MoE models

This notebook is used as the pipeline for training the various non-JE augmented MoE models and displaying their immediate, back of the napkin results. The models evaluated using this notebook are combinations of:

- Generalists: `MobileNet` and `ConvNext`.
- Router types: Single layer MLP router (`MlpRouter`) and a deeper `LayeredRouter`.
- Experts: `MlpExpert`, `TransformerExpert`, and `ResidualGatedMlpExpert`.

> Usage Instructions

1. Open the notebook in a Google Colab GPU runtime.
2. Make any parameter changes you are interested in and scroll down to the `Model Training Pipeline` section. Any models you wish to run should not be commented out.
3. Run all cells (packages will be installed by the notebook, files will be pulled from a repo for use in Colab).
4. Accept the permissions for mounting Google Colab to your Google Drive.
5. Wait a while and view preliminary comparison results at the end of the notebook.

### `model_experiments_comparison.ipynb`: primer experiments results visualization

This notebook is simply displaying the results gathered from `model_experiments.ipynb` runs.

> Usage Instructions

1. Load the json results files from `model_experimetns.ipynb` training runs into a `./results/` directory.
2. Run all cells (packages required will be installed in the notebook if missing).

### Notes

Despite literature, the layered router does show immediate positive improvement over the MLP router. The layered router has a much lower normalized entropy, which indicates it's more confident in assigning experts to given input images. It does come with a small but notable increase in total parameters, so the tradeoff depends on the use-case.

The various expert architectures give a small glimpse at how varying MoE submodels can impact results. The residual gated MLP experts appear to be significantly worse at predicting on the base CNN backbone features than the standard MLP experts. The transformer experts appear to perform slightly worse at generalizing in post training, but they do perform slightly better at forward transfer.

- The transformer experts resulted in the lowest normalizd router entropy, so their use may have resulted in routers that are more confident in particular experts.

There does not appear to be a statistically significant difference (p=0.05) between the router and expert configurations from the validation dataset metrics captured. Further investigation is required with the full test dataset prior to concluding if the simplest MLP router/experts are preferable.
