# Data
This directory contains some scripts to read, modify or make the datasets.

## Make lmdb-format handwriting dataset
1. Download the SCUT-HCCDoc dataset from [link](https://github.com/HCIILAB/SCUT-HCCDoc_Dataset_Release).
2. Modify two paths in ```convert_to_lmdb_format.py``` and run ```python convert_to_lmdb_format.py``` to generate the training and testing sets for the first step.
```python
data_root = 'The absolute path of SCUT-HCCDoc dataset' # eg, '/home/dataset/SCUT-HCCDoc_Dataset_Release_v2'
output_dir = 'hccdoc_lmdb' # eg, '/home/dataset/my_path'
3. Modify two paths in ```mdb_image_to_line.py``` and run ```python mdb_image_to_line.py``` to generate the training and testing sets for the last step.
data_root = 'hccdoc_lmdb' # eg, '/home/dataset/my_path'
output_dir = 'The empty directory for saving images' # eg, '/home/dataset/my_path'
```


