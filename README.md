# Structured Outdoor Architecture Reconstruction by Exploration and Classification

Fuyang Zhang, Xiang Xu, Nelson Nauata, Yasutaka Furukawa.


[[`arXiv`](xxx)]
[[`Project Page`](xxx)]
[[`Bibtex`](#Citing)]

In ICCV 2021

[<img src="images/teaser.gif" width="500">](xxx)

## Requirements
* Python (tested on 3.7)
* Pytorch (tested on 1.4)
* [Neural Mesh Renderer](https://github.com/JiangWenPL/multiperson/tree/master/neural_renderer)
* [PointRend (from Detectron)](https://github.com/facebookresearch/detectron2/tree/master/projects/PointRend)
* [BodyMocap](https://github.com/facebookresearch/frankmocap/)

There is currently no CPU-only support.

## License

Our code is released under CC BY-NC 4.0. However, our code depends on other libraries,
including SMPL, which each have their own respective licenses that must also be
followed.

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

### Setting up External Dependencies

Install the [fast version of Neural Mesh Renderer](https://github.com/JiangWenPL/multiperson/tree/master/neural_renderer):
```
mkdir -p external
git clone https://github.com/JiangWenPL/multiperson.git external/multiperson
pip install external/multiperson/neural_renderer
```


Install [Detectron2](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md):
```
mkdir -p external
git clone --branch v0.2.1 https://github.com/facebookresearch/detectron2.git external/detectron2
pip install external/detectron2
```

Install [FrankMocap](https://github.com/facebookresearch/frankmocap) (The body module is the same regressor trained on [EFT](https://github.com/facebookresearch/eft) data that we used in the paper):
```
mkdir -p external
git clone https://github.com/facebookresearch/frankmocap.git external/frankmocap
sh external/frankmocap/scripts/download_data_body_module.sh
```

You will also need to download the SMPL model from the [SMPLify website](http://smplify.is.tue.mpg.de/). Make an account, and download the neutral model basicModel_neutral_lbs_10_207_0_v1.0.0.pkl and place it in extra_data/smpl/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl


If you did not clone detectron2 and frankmocap in the `external` directory, you will need to update the paths in the constants file.

Currently, the mesh interpenetration loss is not included, so the results may look
slightly different from the paper.

### Meshes

The repository only includes a bicycle mesh that we created. For other object
categories and mesh instances, you will need to download your own meshes. We list some
recommended sources [here](doc/mesh.md).

## Running the Code

```
python demo.py --filename input/000000038829.jpg
```

We also have a [Colab Notebook](https://colab.research.google.com/drive/1QIoL2g0jdt5E-vYKCIojkIz21j3jyEvo?usp=sharing)
to interactively visualize the outputs.


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
