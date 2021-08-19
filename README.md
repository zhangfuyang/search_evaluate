# Structured Outdoor Architecture Reconstruction by Exploration and Classification

Fuyang Zhang, Xiang Xu, Nelson Nauata, Yasutaka Furukawa.


[[`arXiv`](https://arxiv.org/pdf/2108.07990.pdf)]
[[`Project Page`](xxx)]
[[`Bibtex`](#Citing)]

In ICCV 2021

[<img src="images/teaser.png" width="2000">](xxx)

## Prerequisites
- Linux
- NVIDIA GPU + CUDA CuDNN
- Python 3.7+, PyTorch 1.7+

## Dependencies

Install python package dependencies:

```bash
$ pip install -r requirements.txt
```


## Data 
Download data from here...

## Running the Code

### Training
```
python train_evaluators.py
```

You can change the configurations in `config.py`.

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
@InProceedings{zhang2020phosa,
    title = {Perceiving 3D Human-Object Spatial Arrangements from a Single Image in the Wild},
    author = {Zhang, Jason Y. and Pepose, Sam and Joo, Hanbyul and Ramanan, Deva and Malik, Jitendra and Kanazawa, Angjoo},
    booktitle = {European Conference on Computer Vision (ECCV)},
    year = {2020},
}

@InProceedings{zhang2021structured,
      title={Structured Outdoor Architecture Reconstruction by Exploration and Classification}, 
      author={Fuyang Zhang and Xiang Xu and Nelson Nauata and Yasutaka Furukawa},
      year={2021},
      eprint={2108.07990},
      archivePrefix={EInternational Conference on Computer Vision (ICCV)},
      primaryClass={cs.CV}
}
```
## Contact
If you have any questions, please contact fuyangz@sfu.ca or xuxiangx@sfu.ca

