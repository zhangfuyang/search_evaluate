# Structured Outdoor Architecture Reconstruction by Exploration and Classification

Fuyang Zhang, Xiang Xu, Nelson Nauata, Yasutaka Furukawa.


[[`arXiv`](https://arxiv.org/abs/2108.07990)]
[[`Project Page`](xxx)]
[[`Bibtex`](#Citing)]

In ICCV 2021

[<img src="images/teaser.png" width="2000">](https://arxiv.org/abs/2108.07990)

## Prerequisites
- Linux
- NVIDIA GPU, CUDA 11+
- Python 3.7+, PyTorch 1.7+

## Dependencies

Install additional python package dependencies:

```bash
$ pip install -r requirements.txt
```


## Data 
Download the processed data from this [link](https://drive.google.com/file/d/1T7l1UbS4MtdbUCxpAwgJhHCESAZvtvqa/view?usp=sharing). This includes the original cities dataset from ["Vectorizing World Buildings: Planar Graph Reconstruction by Primitive Detection and Relationship Classification"](https://arxiv.org/abs/1912.05135) and predictions from Conv-MPN, IP and Per-Edge models.

Download the pretrained heatmap weights from this [link](https://drive.google.com/file/d/162V03dUC4Zxj-RK4N8rUOjOau4cFgUX3/view?usp=sharing). 

Both data are required for training and evaluation, unzip and move them to the `data` folder. 

## Running the Code

### Training
```
python train_evaluators.py
```
This will start both the train and search threads. 

You can change settings like beam search depth or number of training epochs in the `config.py`.

### Evaluation
First, perform search over the test data:

```
python search_result.py
```

Then, evaluate scores using:
```
python metric_for_result.py
```

## Pretrained models
We provide pretrained models here...


## <a name="Citing"></a>Citation
If you use find this code helpful, please consider citing:
```BibTeX
@InProceedings{zhang2021structured,
      title={Structured Outdoor Architecture Reconstruction by Exploration and Classification}, 
      author={Fuyang Zhang and Xiang Xu and Nelson Nauata and Yasutaka Furukawa},
      year={2021},
      eprint={2108.07990},
      archivePrefix={International Conference on Computer Vision (ICCV)},
      primaryClass={cs.CV}
}
```
## Contact
If you have any questions, please contact fuyangz@sfu.ca or xuxiangx@sfu.ca

