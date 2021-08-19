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
First, perform beam search over all the test data:

```
python search_result.py
```

Then, evaluate the scores for all searched results:
```
python metric_for_result.py
```

## Pretrained models
Download individual pretrained model and its beam search results.

| **Trained on** | **Input** | **Train**            | Method                                       | **Model Download**                                                                                                            |
|--------------------|-----------|----------------------|---------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------|
| NYUv2              | RGBD      | raw sensor depth     | [saic](https://github.com/saic-vul/saic_depth_completion/tree/94bececdf12bb9867ce52c970bb2d11dee948d37) | [saic_rawD.zip](http://aspis.cmpt.sfu.ca/projects/mirrors/mirror3d_zip_release/checkpoint/nyu/saic_rawD.zip)                  |
| NYUv2              | RGBD      | refined sensor depth | [saic](https://github.com/saic-vul/saic_depth_completion/tree/94bececdf12bb9867ce52c970bb2d11dee948d37) | [saic_refD.zip](http://aspis.cmpt.sfu.ca/projects/mirrors/mirror3d_zip_release/checkpoint/nyu/saic_refD.zip)                  |
| NYUv2              | RGB       | raw sensor depth     | [BTS](https://github.com/cogaplex-bts/bts)                                                              | [bts_nyu_v2_pytorch_densenet161.zip](https://cogaplex-bts.s3.ap-northeast-2.amazonaws.com/bts_nyu_v2_pytorch_densenet161.zip) |
| NYUv2              | RGB       | refined sensor depth | [BTS](https://github.com/cogaplex-bts/bts)                                                              | [bts_refD.zip](http://aspis.cmpt.sfu.ca/projects/mirrors/mirror3d_zip_release/checkpoint/nyu/bts_refD.zip)                    |
| NYUv2              | RGB       | raw sensor depth     | [VNL](https://github.com/YvanYin/VNL_Monocular_Depth_Prediction)                                        | [nyu_rawdata.pth](https://cloudstor.aarnet.edu.au/plus/s/7kdsKYchLdTi53p)                                                     |
| NYUv2              | RGB       | refined sensor depth | [VNL](https://github.com/YvanYin/VNL_Monocular_Depth_Prediction)                                        | [vnl_refD.zip](http://aspis.cmpt.sfu.ca/projects/mirrors/mirror3d_zip_release/checkpoint/nyu/vnl_refD.zip)                    |
| Matterport3D       | RGBD      | raw mesh depth     | [Mirror3DNet](https://github.com/3dlg-hcvc/mirror3d/tree/main/mirror3dnet)                              | [mirror3dnet_rawD.zip](http://aspis.cmpt.sfu.ca/projects/mirrors/mirror3d_zip_release/checkpoint/mp3d/mirror3dnet_rawD.zip)   |
| Matterport3D       | RGBD      | refined mesh depth | [Mirror3DNet](https://github.com/3dlg-hcvc/mirror3d/tree/main/mirror3dnet)                              | [mirror3dnet_refD.zip](http://aspis.cmpt.sfu.ca/projects/mirrors/mirror3d_zip_release/checkpoint/mp3d/mirror3dnet_refD.zip)   |
| Matterport3D       | RGBD      | raw mesh depth     | [PlaneRCNN](https://github.com/NVlabs/planercnn/tree/01e03fe5a97b7afc4c5c4c3090ddc9da41c071bd)          | [planercnn_rawD.zip](http://aspis.cmpt.sfu.ca/projects/mirrors/mirror3d_zip_release/checkpoint/mp3d/planercnn_rawD.zip)       |
| Matterport3D       | RGBD      | refined mesh depth | [PlaneRCNN](https://github.com/NVlabs/planercnn/tree/01e03fe5a97b7afc4c5c4c3090ddc9da41c071bd)          | [planercnn_refD.zip](http://aspis.cmpt.sfu.ca/projects/mirrors/mirror3d_zip_release/checkpoint/mp3d/planercnn_refD.zip)       |
| Matterport3D       | RGBD      | raw mesh depth     | [saic](https://github.com/saic-vul/saic_depth_completion/tree/94bececdf12bb9867ce52c970bb2d11dee948d37) | [saic_rawD.zip](http://aspis.cmpt.sfu.ca/projects/mirrors/mirror3d_zip_release/checkpoint/mp3d/saic_rawD.zip)                 |
| Matterport3D       | RGBD      | refined mesh depth | [saic](https://github.com/saic-vul/saic_depth_completion/tree/94bececdf12bb9867ce52c970bb2d11dee948d37) | [saic_refD.zip](http://aspis.cmpt.sfu.ca/projects/mirrors/mirror3d_zip_release/checkpoint/mp3d/saic_refD.zip)                 |
| Matterport3D       | RGB       | *                    | [Mirror3DNet](https://github.com/3dlg-hcvc/mirror3d/tree/main/mirror3dnet)                              | [mirror3dnet.zip](http://aspis.cmpt.sfu.ca/projects/mirrors/mirror3d_zip_release/checkpoint/mp3d/mirror3dnet_normal_10.zip)   |
| Matterport3D       | RGB       | raw mesh depth     | [BTS](https://github.com/cogaplex-bts/bts)                                                              | [bts_rawD.zip](http://aspis.cmpt.sfu.ca/projects/mirrors/mirror3d_zip_release/checkpoint/mp3d/bts_rawD.zip)                   |
| Matterport3D       | RGB       | refined mesh depth | [BTS](https://github.com/cogaplex-bts/bts)                                                              | [bts_refD.zip](http://aspis.cmpt.sfu.ca/projects/mirrors/mirror3d_zip_release/checkpoint/mp3d/bts_refD.zip)                   |
| Matterport3D       | RGB       | raw mesh depth     | [VNL](https://github.com/YvanYin/VNL_Monocular_Depth_Prediction)                                        | [vnl_rawD.zip](http://aspis.cmpt.sfu.ca/projects/mirrors/mirror3d_zip_release/checkpoint/mp3d/vnl_rawD.zip)                   |
| Matterport3D       | RGB       | refined mesh depth | [VNL](https://github.com/YvanYin/VNL_Monocular_Depth_Prediction)                                        | [vnl_refD.zip](http://aspis.cmpt.sfu.ca/projects/mirrors/mirror3d_zip_release/checkpoint/mp3d/vnl_refD.zip)                   |



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

