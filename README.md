# Counting-Bench

## Prerequisites

* Packages: Python3.7.8. and the packages in `requirements.txt` are required to run the scripts.

* Datasets: Please download the datasets from [Google Drive Link?](https://google.com) and unzip them under the folder `data/`.

* Pre-Trained Checkpoints: Please download the pre-trained checkpoints from [Google Drive Link?](https://google.com) and put them in the folder `cps/`.

## Code Structure
Every file under the folder `pys` has the same structure as illustrated below:
* Data: Preprocess data, and define and load the datasets.
* Model: Define the crowd counting model.
* Training: Define the optimizer and train the model. 
* Evaluation: Evaluate the model. 

## Main Hyper-Parameters
This part introduces the main common hyper-parameters in the code. Parameters in a single file are not included. 

### Dataset Parameters
* `crop_size`: The size of the randomly cropped part of the images.
* `downsample`: The ratio by which the density maps shrink. For example, the size of the density map of a 256x256 image will be 32x32 if `downsample` is 8. 

### Training Parameters
* `num_workers`: Number of workers created by the Pytorch dataloader. 
* `batch_size`: Number of samples in one mini-batch.
* `n_epochs`: Number of epoches for the training process.
* `max_lr`: Maximum learning rate in the OneCycleLR.
* `pct_start`: Percentage of the cycle when the learning rate is increasing in the OneCycleLR.
* `anneal_strategy`: Specify the cosine or linear annealing strategy. 
* `final_div_factor`: Ratio of the maximum learning rate over the minimum learning rate. 

### Other Parameters
* `seed`: Random seed of the whole process. 
* `log_para`: Number by which the density maps are divided. 

## Models

* AMRNet: [Paper](https://arxiv.org/pdf/2005.05776.pdf) | [Code](https://github.com/xiyang1012/Local-Crowd-Counting)
* CSRNet: [Paper](https://arxiv.org/pdf/1802.10062.pdf) | [Code](https://github.com/leeyeehoo/CSRNet-pytorch)
* DENet: [Paper](https://arxiv.org/ftp/arxiv/papers/1904/1904.08056.pdf) | [Code](https://github.com/liuleiBUAA/DENet)
* DSSINet: [Paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Liu_Crowd_Counting_With_Deep_Structured_Scale_Integration_Network_ICCV_2019_paper.pdf) | [Code](https://github.com/shizenglin/Counting-with-Focus-for-Free)
* SACANet: [Paper](https://arxiv.org/pdf/1909.03839.pdf) | [Code](https://github.com/HaoyueBaiZJU/SACANet-VisDrone-Crowd)
* VisionTransformer: [Paper](https://arxiv.org/pdf/2010.11929.pdf)