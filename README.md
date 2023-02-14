## GOGGLE: Generative Modelling of Tabular Data By Learning Relational Structure

PyTorch implementation of ICLR'23 paper [GOGGLE: Generative Modelling for Tabular Data by Learning Relational Structure](https://openreview.net/forum?id=fPVRcJqspu&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DICLR.cc%2F2023%2FConference%2FAuthors%23your-submissions)). Authors: Tennison Liu, Zhaozhi Qian, Jeroen Berrevoets, Mihaela van der Schaar

---
###Abstract

Generative modelling of tabular data entails a particular set of challenges, including heterogeneous relationships, limited number of samples, and difficulties in incorporating prior knowledge. This work introduces **GOGGLE**, a generative model that learns a relational structure underlying tabular data to better model variable dependencies, to introduce regularization, and to incorporate prior knowledge.

![GOGGLE Overview](./figures/goggle_recipe.png?raw=True)
**Key components of GOGGLE Framework.** 1. Simultaneous learning of relational structure $G_\phi$ and $F_\theta$ s.t. generative process respects relational structure. 2. Injection of prior knowledge and regularization on variable dependence. 3. Synthetic sample generated using $\hat{x} = F_\theta(z; G_\phi) \:, \: z\sim p_Z$.

---
###Experiments

To setup the virtual environment and necessary packages, please run the following commands:
```
$ conda create --name goggle_env python=3.8
$ conda activate goggle_env
```
Clone this repository and navigate to the root directory:
```
$ git clone https://github.com/tennisonliu/goggle.git
$ cd goggle
```
Install the required modules:
```
$ pip install requirements.txt
```

Place dataset in ```exps/data```, see experiment notebooks with instructions in ```exps/exp_{data}.ipynb```.

---

###Citation
If our paper or code helped you in your own research, please cite our work as:
```
@inproceedings{liugoggle,
  title={GOGGLE: Generative Modelling for Tabular Data by Learning Relational Structure},
  author={Liu, Tennison and Qian, Zhaozhi and Berrevoets, Jeroen and van der Schaar, Mihaela},
  booktitle={International Conference on Learning Representations}
}
```



