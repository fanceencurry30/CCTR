# CCTR

## 1. Methods and Results
Table 1， Table 2 and Table 3.

*Note: Only the results from Table 2 are shown here for clarity. Other tables also can using this code to get results.*


## 2. Environment

- [PyTorch](http://pytorch.org/) version >= 1.13.0
- Python version >= 3.7

```shell
# Ubuntu 20.04 Cuda 11.8
conda create -n openocr python==3.8
conda activate cctr
conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```



## 3. Model Training / Evaluation

### 3.1 Dataset Preparation

```
python ./dataset_Pre/convert_to_lmdb_format.py
python ./dataset_Pre/mdb_image_to_line.py
```

### 3.2 Training

```shell
# For language model
python ./LLM-2/train.py

# For fusion model
python ./1_prepare_ctc.py # 生成原始ocr的ocr100
python ./2_prepare_fusion_data.py # 根据1_prepare_ctc.py处理结果，对样本进行掩码，然后生成相应的lm100
python ./3_normalize_data.py # 对ocr100和lm100序列进行归一化
python ./4_train_fusion_modern.py # 接受3_normalize_data.py的数据，训练融合层
```

### 3.3 Evaluation

```shell
python ./5_test_line.py # 测试我们的方法的行级指标
python ./5_test_page.py # 测试我们的方法的行级指标
python ./tools/eval_rec_all_ch # 测试openocr的行级指标
python ./eval _image_level_acc.py # 测试openocr的页级指标
```