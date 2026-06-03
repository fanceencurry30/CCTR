import csv
import os
import sys
import numpy as np
import glob
import lmdb
import shutil
import tempfile

__dir__ = os.path.dirname(os.path.abspath(__file__))

sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))

from tools.data import build_dataloader
from tools.engine.config import Config
from tools.engine.trainer import Trainer
from tools.utility import ArgsParser


def get_config_files(method_yml_dir='./method_smtr'):
    """自动获取method_yml目录下的所有yml配置文件"""
    if not os.path.exists(method_yml_dir):
        raise FileNotFoundError(f"目录 {method_yml_dir} 不存在")
    
    # 查找所有yml文件
    yml_files = glob.glob(os.path.join(method_yml_dir, "*.yml"))
    if not yml_files:
        raise FileNotFoundError(f"在 {method_yml_dir} 目录下未找到yml文件")
    
    # 按文件名排序以确保一致性
    yml_files.sort()
    
    # 从文件名提取方法名（去掉扩展名）
    method_names = [os.path.splitext(os.path.basename(f))[0] for f in yml_files]
    
    return yml_files, method_names


def parse_args():
    parser = ArgsParser()
    args = parser.parse_args()
    return args


def clean_label_text(label):
    """清理标签，去掉<image_id=..._line_id=...>部分"""
    # 查找 <image_id= 的位置
    start_idx = label.find('<image_id=')
    if start_idx != -1:
        # 返回标签的真实文本部分
        return label[:start_idx].strip()
    return label.strip()


def process_lmdb_labels(original_lmdb_path, cleaned_base_dir):
    """处理LMDB文件的标签，创建只包含真实标签的新LMDB"""
    # 从原始路径提取文件夹名
    original_dir_name = os.path.basename(original_lmdb_path.rstrip('/'))
    # 创建新LMDB路径
    new_lmdb_path = os.path.join(cleaned_base_dir, f"{original_dir_name}_cleaned")
    
    print(f"处理LMDB标签: {original_lmdb_path}")
    print(f"新LMDB路径: {new_lmdb_path}")
    
    # 确保目标目录存在
    os.makedirs(new_lmdb_path, exist_ok=True)
    
    # 打开原始LMDB
    env_original = lmdb.open(original_lmdb_path, readonly=True, lock=False, readahead=False)
    
    # 创建新LMDB
    env_new = lmdb.open(new_lmdb_path, map_size=1099511627776)  
    
    with env_original.begin() as txn_original:
        with env_new.begin(write=True) as txn_new:
            # 获取所有key
            for key, value in txn_original.cursor():
                try:
                    # 尝试解码key
                    key_str = key.decode('utf-8')
                    
                    if key_str.startswith('label-'):
                        try:
                            value_str = value.decode('utf-8')
                            cleaned_label = clean_label_text(value_str)
                            txn_new.put(key, cleaned_label.encode('utf-8'))
                            #print(f"  处理标签: {value_str} -> {cleaned_label}")
                        except UnicodeDecodeError:
                            #print(f"  警告: 标签数据无法UTF-8解码，直接复制: {key_str}")
                            txn_new.put(key, value)
                    elif key_str.startswith('image-') or key_str == 'num-samples':
                        txn_new.put(key, value)
                    else:
                        txn_new.put(key, value)
                        
                except UnicodeDecodeError:
                    txn_new.put(key, value)
                except Exception as e:
                    print(f"  处理key时出错: {key}, 错误: {e}")
                    txn_new.put(key, value)
    
    env_original.close()
    env_new.close()
    
    print(f"LMDB处理完成: {original_lmdb_path} -> {new_lmdb_path}")
    return new_lmdb_path


def eval_line_acc(input_dir, original_data_dirs_list, output_dir, dataset_names):
    # 自动获取配置文件和对应的方法名
    try:
        config_paths, method_names = get_config_files(input_dir)
        print(f"找到 {len(config_paths)} 个配置文件:")
        for config_path, method_name in zip(config_paths, method_names):
            print(f"  {method_name}: {config_path}")
    except FileNotFoundError as e:
        print(f"错误: {e}")
        return
        
    # 创建cleaned目录,存储临时mdb数据
    cleaned_base_dir = './cleaned'
    if os.path.exists(cleaned_base_dir):
        print(f"清理已存在的目录: {cleaned_base_dir}")
        shutil.rmtree(cleaned_base_dir)
    os.makedirs(cleaned_base_dir)
    print(f"创建清理目录: {cleaned_base_dir}")
    
    # 处理所有LMDB，创建清理后的版本
    cleaned_data_dirs_list = []
    cleaned_dirs_to_remove = []  # 保存需要清理的目录
    
    print("开始处理LMDB标签...")
    for original_dirs in original_data_dirs_list:
        cleaned_dirs = []
        for original_dir in original_dirs:
            if os.path.exists(original_dir):
                try:
                    cleaned_lmdb_path = process_lmdb_labels(original_dir, cleaned_base_dir)
                    cleaned_dirs.append(cleaned_lmdb_path)
                    cleaned_dirs_to_remove.append(cleaned_lmdb_path)
                except Exception as e:
                    print(f"处理LMDB失败 {original_dir}: {e}")
                    # 如果处理失败，使用原始路径
                    cleaned_dirs.append(original_dir)
            else:
                print(f"警告: 路径不存在 {original_dir}")
                cleaned_dirs.append(original_dir)  # 如果路径不存在，保持原样
        cleaned_data_dirs_list.append(cleaned_dirs)
    
    # 初始化结果存储字典
    results = {
        'acc': {method: [] for method in method_names},
        'ar': {method: [] for method in method_names}, 
        'cr': {method: [] for method in method_names}
    }
    
    # 循环处理每个方法
    for config_path, method_name in zip(config_paths, method_names):
        print(f"\n正在处理 {method_name} ...")
        print(f"配置文件: {config_path}")
        
        try:
            # 设置配置文件路径
            sys.argv = [sys.argv[0], '--config', config_path]
            FLAGS = parse_args()
            cfg = Config(FLAGS.config)
            FLAGS = vars(FLAGS)
            opt = FLAGS.pop('opt')
            cfg.merge_dict(FLAGS)
            cfg.merge_dict(opt)
            
            # 检测数据集类型
            msr = False
            if 'RatioDataSet' in cfg.cfg['Eval']['dataset']['name']:
                msr = True

            # 配置预处理
            if cfg.cfg['Global']['output_dir'][-1] == '/':
                cfg.cfg['Global']['output_dir'] = cfg.cfg['Global']['output_dir'][:-1]
            if cfg.cfg['Global']['pretrained_model'] is None:
                cfg.cfg['Global']['pretrained_model'] = cfg.cfg['Global']['output_dir'] + '/best.pth'
            
            cfg.cfg['Global']['use_amp'] = False
            cfg.cfg['PostProcess']['with_ratio'] = True
            cfg.cfg['Metric']['with_ratio'] = True
            cfg.cfg['Metric']['max_len'] = 200
            cfg.cfg['Metric']['max_ratio'] = 12
            
            trainer = Trainer(cfg, mode='eval')

            # 测试每个子数据集（使用清理后的LMDB）
            method_acc_results = []
            method_ar_results = [] 
            method_cr_results = []
            
            for i, cleaned_dirs in enumerate(cleaned_data_dirs_list):
                print(f"  测试数据集 {i+1}/{len(cleaned_data_dirs_list)}: {cleaned_dirs[0]}")
                
                try:
                    config_each = cfg.cfg.copy()
                    # 使用清理后的数据路径
                    if msr:
                        config_each['Eval']['dataset']['data_dir_list'] = cleaned_dirs
                    else:
                        config_each['Eval']['dataset']['data_dir'] = cleaned_dirs[0]
                        
                    valid_dataloader = build_dataloader(config_each, 'Eval', trainer.logger)
                    trainer.logger.info(f'{cleaned_dirs[0]} valid dataloader has {len(valid_dataloader)} iters')
                    trainer.valid_dataloader = valid_dataloader
                    metric = trainer.eval()
                    
                    # 记录三个关键指标
                    acc_value = metric['acc'] * 100
                    ar_value = metric['ar'] * 100  
                    cr_value = metric['cr'] * 100
                    
                    method_acc_results.append(acc_value)
                    method_ar_results.append(ar_value)
                    method_cr_results.append(cr_value)
                    
                    print(f"    {dataset_names[i]}: acc={acc_value:.2f}, ar={ar_value:.2f}, cr={cr_value:.2f}")
                    
                except Exception as e:
                    print(f"    ✗ 测试数据集 {dataset_names[i]} 时出错: {e}")
                    # 如果单个数据集测试失败，记录0值
                    method_acc_results.append(0.0)
                    method_ar_results.append(0.0)
                    method_cr_results.append(0.0)
            
            # 存储结果
            results['acc'][method_name] = method_acc_results
            results['ar'][method_name] = method_ar_results
            results['cr'][method_name] = method_cr_results
            
            print(f"✓ {method_name} 处理完成")
            
        except Exception as e:
            print(f"✗ 处理 {method_name} 时出错: {e}")
            import traceback
            traceback.print_exc()
            # 如果某个方法处理失败，用0值填充结果
            results['acc'][method_name] = [0.0] * len(cleaned_data_dirs_list)
            results['ar'][method_name] = [0.0] * len(cleaned_data_dirs_list)
            results['cr'][method_name] = [0.0] * len(cleaned_data_dirs_list)
            continue
    
    # 写入CSV文件
    write_results_to_csv(results, method_names, dataset_names, output_dir)
    
    # 清理临时目录
    print("\n清理临时文件...")
    if os.path.exists(cleaned_base_dir):
        try:
            shutil.rmtree(cleaned_base_dir)
            print(f"已清理目录: {cleaned_base_dir}")
        except Exception as e:
            print(f"清理目录失败 {cleaned_base_dir}: {e}")


def write_results_to_csv(results, method_names, dataset_names, output_dir):
    """将结果按指定格式写入CSV文件"""
    with open(output_dir, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        # 写入第一个表格：acc指标
        writer.writerow(['acc(行级)'] + dataset_names)
        for method in method_names:
            # 处理可能以运算符开头的数据
            formatted_values = []
            for value in results['acc'][method]:
                str_value = f"{value:.2f}"
                if str_value.startswith(('-', '+', '=')):
                    str_value = "'" + str_value
                formatted_values.append(str_value)
            row = [method] + formatted_values
            writer.writerow(row)
        
        # 空行分隔两个表格
        writer.writerow([])
        
        # 写入第二个表格：ar/cr指标（x/y格式）
        writer.writerow(['ar/cr(行级)'] + dataset_names)
        
        # 每个方法一行，格式为 ar值/cr值
        for method in method_names:
            ar_values = results['ar'][method]
            cr_values = results['cr'][method]
            # 将ar和cr组合成 x/y 格式
            combined_values = []
            for ar, cr in zip(ar_values, cr_values):
                combined = f"{ar:.2f}/{cr:.2f}"
                if combined.startswith(('-', '+', '=')):
                    combined = "'" + combined
                combined_values.append(combined)
            row = [method] + combined_values
            writer.writerow(row)
    
    print(f"\n✓ 结果已保存到{output_dir}")


if __name__ == '__main__':
    # 各个测试方法所在目录，自动评测多个方法。
    # 如若显存空间有限，请自行减少单次测试时的测试集数目/方法数目
    input_dir = './method_yml'
    # 待测试数据集路径
    original_data_dirs_list = [
         ['./image_lmdb/image_base_lmdb'],
         ['./image_lmdb/image_底纹_lmdb'],
         ['./image_lmdb/image_反光_lmdb'],
        ['./image_lmdb/image_复杂背景_lmdb'],
        ['./image_lmdb/image_光线不足_lmdb'],
         ['./image_lmdb/image_扭曲斜角度_lmdb'],
        ['./image_lmdb/image_特殊材质_lmdb'],
        ['./image_lmdb/image_小图虚图_lmdb'],
        ['./image_lmdb/image_遮挡_lmdb'],
        ['./image_lmdb/image_hw_lmdb']
    ]
    # 最终结果存放在csv表中
    output_dir = 'result.csv'
    # csv表表头
    dataset_names = ['base', '底纹', '反', '复杂', '光线', '扭曲', '特殊材质', '小图', '遮挡', 'hw']
    eval_line_acc(input_dir, original_data_dirs_list, output_dir, dataset_names)


