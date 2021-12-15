# Structured Outdoor Architecture Reconstruction by Exploration and Classification

Fuyang Zhang, Xiang Xu, Nelson Nauata, Yasutaka Furukawa.


[[`arXiv`](https://arxiv.org/abs/2108.07990)]
[[`Project Page`](https://zhangfuyang.github.io/expcls/)]
[[`Bibtex`](#Citing)]

In ICCV 2021

[<img src="images/teaser.png" width="2000">](https://arxiv.org/abs/2108.07990)

## Prerequisites
- Linux
- NVIDIA GPU, CUDA 11+
- Python 3.7+, PyTorch 1.7+

## Dependencies

Install additional dependencies:

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

| **Training Dataset** | **Model Weights** | **Beam Search Results**|
|--------------------|-----------|----------------------|
| Conv-MPN           | [convmpn_weights.zip](https://drive.google.com/file/d/1CkX2E_WtlVMYOUXHDKLkdC1_EqMFv4B6/view?usp=sharing)      | [convmpn_beamsearch.zip](https://drive.google.com/file/d/15PzDz1ibeFoHtXXQD8kSvZoC_KcbZEu5/view?usp=sharing)    |
| IP                 | [ip_weights.zip](https://drive.google.com/file/d/1z2cNS2js5pILNksxhlKkIBWRQ8wpu7eG/view?usp=sharing)      | [ip_beamsearch.zip](https://drive.google.com/file/d/1jolL4xFWkS6bmBFIjtRgkayOmXcwC8aI/view?usp=sharing)    |
| Per-Edge           | [peredge_weights.zip](https://drive.google.com/file/d/1wDfqwOa6xVWlDG93AjHX7OGdbDzsxGDd/view?usp=sharing)      | [peredge_beamsearch.zip](https://drive.google.com/file/d/1OJQVfP0dEkNBdB4QscLxaB44S5_6wlAb/view?usp=sharing)     |

## <a name="Citing"></a>Citation
If you find this code helpful, please consider citing:
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

## Acknowledgement
This research is partially supported by NSERC Discovery Grants with Accelerator Supplements and DND/NSERC Discovery Grant Supplement.

