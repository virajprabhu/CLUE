## Code for [Active Domain Adaptation via Clustering UNcertainty-weighted Embeddings](https://virajprabhu.github.io/adaclue-web/) (ICCV 2021)
### Viraj Prabhu, Arjun Chandrasekaran, Kate Saenko, Judy Hoffman

Generalizing deep neural networks to new target domains is critical to their real-world utility. In practice, it may be feasible to get some target data labeled, but to be cost-effective it is desirable to select a maximally-informative subset via active learning (AL). We study the problem of AL under a domain shift, called Active Domain Adaptation (Active DA). We demonstrate how existing AL approaches based solely on model uncertainty or diversity sampling are less effective for Active DA. We propose Clustering Uncertainty-weighted Embeddings (CLUE), a novel label acquisition strategy for Active DA that performs uncertainty-weighted clustering to identify target instances for labeling that are both uncertain under the model and diverse in feature space. CLUE consistently outperforms competing label acquisition strategies for Active DA and AL across learning settings on 6 diverse domain shifts for image classification.

![method](plots/clue_mme.gif)

Table of Contents
=================

   * [Setup and Dependencies](#setup-and-dependencies)
   * [Usage](#usage)
      * [Train Active Domain Adaptation model](#train-active-domain-adaptation-model)
      * [Download data](#data-download)
      * [Pretrained checkpoints](#pretrained-checkpoints)
      * [Evaluation and plotting Results](#evaluation-and-plotting-results)
      * [Demo](#demo)
   * [Reference](#reference)
   * [License](#license)

## Setup and Dependencies

1. Create an anaconda environment with [Python 3.6](https://www.python.org/downloads/release/python-365/) and activate: 
```
conda create -n CLUE python=3.6.8
conda activate CLUE
```
2. Navigate into the code directory: ```cd CLUE/```
3. Install dependencies: (Takes ~2-3 minutes) 
```
pip install -r requirements.txt
``` 
4. [If running the demo] Install nb_conda:
```
conda install -c anaconda-nb-extensions nb_conda
``` 

And you're all set up! 

## Usage 

### Train Active Domain Adaptation model

Run ```python train.py``` to train an active adaptation model from scratch, by passing it appropriate arguments.

We include hyperparameter configurations to reproduce paper numbers on DIGITS and DomainNet as configurations inside the ```config``` folder. For instance, to reproduce DIGITS (SVHN->MNIST) results with CLUE+MME, run:

```
python train.py --load_from_cfg True \ 
                --cfg_file config/digits/clue_mme.yml \
                --use_cuda False
```

To run a custom train job, you can create a custom config file and pass it to the train script. Pass `--use_cuda False` if you'd like to train on CPU instead.

### Download data

Data for SVHN->MNIST is downloaded automatically via PyTorch. For DomainNet, follow the following steps:
1. Download the original dataset for the domains of interest from [this link](http://ai.bu.edu/M3SDA/) – eg. Clipart and Sketch.
2. Run: 
```
python preprocess_domainnet.py --input_dir <input_directory> \
                               --domains 'clipart,sketch' \
                               --output_dir 'data/'
```

### Pretrained checkpoints

At round 0, active adaptation begins from a model trained on the source domain, or from a model first trained on source and then
adapted to the target via unsupervised domain adaptation. Checkpoints for reproducing DIGITS experiments have been included in the
```checkpoints/``` directory, and those for reproducing DomainNet results on Clipart->Sketch can be downloaded at [this link](https://drive.google.com/drive/u/0/folders/1iaGouaz-KWPEbOqPjOEkPZpcwijVxtPX). Note that checkpoints for models after active adaptation are not included.

### Evaluation and plotting Results

Run ```python evaluate.py``` by passing it appropriate arguments (see file for instructions). It will pretty-print raw results as well as save them as a figure in the ```plots/``` directory. By default, it will generate a figure comparing CLUE + MME against a subset of representative Active DA and AL baselines and save it to the ```plots/``` directory.

### Demo

1. Start a jupyter notebook with ```jupyter notebook''', and set the conda environment to adaclue
2. Run the Jupyter notebook ```demo.ipynb```, which will walk you through:
    * Loading SVHN, MNIST datasets and pretrained checkpoints
    * Label acquisition with baseline strategies and CLUE+MME
    * Training (on CPU) with acquired labels
    * Plotting performance after one round of Active DA on SVHN->MNIST

## Reference

If you found this code useful, please consider citing:
```
@inproceedings{prabhu2021active,
  title={Active domain adaptation via clustering uncertainty-weighted embeddings},
  author={Prabhu, Viraj and Chandrasekaran, Arjun and Saenko, Kate and Hoffman, Judy},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={8505--8514},
  year={2021}
}
```

## Acknowledgements

We would like to thank the developers of PyTorch for building an excellent framework, the [Deep Active Learning](https://github.com/ej0cl6/deep-active-learning) repository for implementations of some of our baselines, and the numerous contributors to all the open-source packages we use.

## License

MIT
