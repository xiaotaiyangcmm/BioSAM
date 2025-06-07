
# BioSAM
**Code for the Paper:**
> **BioSAM: Generating SAM Prompts From Superpixel Graph for Biological Instance Segmentation**  
> Miaomiao Cai, Xiaoyu Liu, Zhiwei Xiong, Xuejin Chen  
> *IEEE Journal of Biomedical and Health Informatics (J-BHI), 2024*

---

## 🔧 Installation

This codebase is tested with **Python 3.8** and **PyTorch 1.7**.

Install the required dependencies:

```shell
pip install numexpr
pip install segment_anything
```
Then build cython modules：
```shell
cd ./third_party/cython
python setup.py install
cd ../../cython_function
python setup.py install
cd ../
```
## 🚀 Training
To train the BioSAM model on your dataset:
```shell
CUDA_VISIBLE_DEVICES=0  python -m torch.distributed.launch --nproc_per_node=1 train.py
```
## 🔍 Inference
To perform inference with a trained model:
```shell
python inference.py
```
You can download the pretrained checkpoint (checkpoint.pth) from the following link:
https://pan.baidu.com/s/18InvWLENrD8iq8_iuPhHKg?pwd=w7ix (Access code: w7ix)

## 📫 Contact

If you have any problem with the released code, please contact me by email (mmcai@mail.ustc.edu.cn).

## Citation
```shell
@article{cai2024biosam,
  title={BioSAM: Generating SAM Prompts From Superpixel Graph for Biological Instance Segmentation},
  author={Cai, Miaomiao and Liu, Xiaoyu and Xiong, Zhiwei and Chen, Xuejin},
  journal={IEEE Journal of Biomedical and Health Informatics},
  year={2024},
  publisher={IEEE}
}
```
