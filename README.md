# Structured Outdoor Architecture Reconstruction by Exploration and Classification

Fuyang Zhang, Xiang Xu, Nelson Nauata, Yasutaka Furukawa.


[[`arXiv`](xxx)]
[[`Project Page`](xxx)]
[[`Bibtex`](#Citing)]

In ICCV 2021

[<img src="images/teaser.png" width="2000">](xxx)

## Requirements
* Python (tested on 3.7)
* Pytorch (tested on 1.4)
* [Neural Mesh Renderer](https://github.com/JiangWenPL/multiperson/tree/master/neural_renderer)
* [PointRend (from Detectron)](https://github.com/facebookresearch/detectron2/tree/master/projects/PointRend)
* [BodyMocap](https://github.com/facebookresearch/frankmocap/)


## Installation

We recommend using a conda environment:

```bash
conda create -n phosa python=3.7
conda activate phosa
pip install -r requirements.txt
```

Install the torch version that corresponds to your version of CUDA, eg for CUDA 10.0,
use:
```
conda install pytorch=1.4.0 torchvision=0.5.0 cudatoolkit=10.0 -c pytorch
```

Alternatively, you can check out our interactive [Colab Notebook](https://colab.research.google.com/drive/1QIoL2g0jdt5E-vYKCIojkIz21j3jyEvo?usp=sharing).



## Running the Code

```
python demo.py --filename input/000000038829.jpg
```

We also have a [Colab Notebook](https://colab.research.google.com/drive/1QIoL2g0jdt5E-vYKCIojkIz21j3jyEvo?usp=sharing)
to interactively visualize the outputs.


## Pretrained Models


## License

Our code is released under CC BY-NC 4.0. However, our code depends on other libraries,
including SMPL, which each have their own respective licenses that must also be
followed.


## <a name="Citing"></a>Citing Our Work

If you use find this code helpful, please consider citing:
```BibTeX
@InProceedings{zhang2020phosa,
    title = {Perceiving 3D Human-Object Spatial Arrangements from a Single Image in the Wild},
    author = {Zhang, Jason Y. and Pepose, Sam and Joo, Hanbyul and Ramanan, Deva and Malik, Jitendra and Kanazawa, Angjoo},
    booktitle = {European Conference on Computer Vision (ECCV)},
    year = {2020},
}
```


