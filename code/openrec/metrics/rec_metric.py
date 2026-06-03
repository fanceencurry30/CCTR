import string
import numpy as np
from rapidfuzz.distance import Levenshtein
import re
import unicodedata
try:
    from opencc import OpenCC
    cc = OpenCC('t2s')  # 繁体转简体
except:
    cc = None
    
def match_ss(ss1, ss2):
    s1_len = len(ss1)
    for c_i in range(s1_len):
        if ss1[c_i:] == ss2[:s1_len - c_i]:
            return ss2[s1_len - c_i:]
    return ss2


def stream_match(text):
    bs = len(text)
    s_list = []
    conf_list = []
    for s_conf in text:
        s_list.append(s_conf[0])
        conf_list.append(s_conf[1])
    s_n = bs
    s_start = s_list[0][:-1]
    s_new = s_start
    for s_i in range(1, s_n):
        s_start = match_ss(
            s_start, s_list[s_i][1:-1] if s_i < s_n - 1 else s_list[s_i][1:])
        s_new += s_start
    return s_new, sum(conf_list) / bs


class RecMetric(object):

    def __init__(self,
                 main_indicator='acc',
                 is_filter=False,
                 is_lower=True,
                 ignore_space=True,
                 stream=False,
                 with_ratio=False,
                 max_len=200,
                 max_ratio=4,
                 **kwargs):
        self.main_indicator = main_indicator
        self.is_filter = is_filter
        self.is_lower = is_lower
        self.ignore_space = ignore_space
        self.stream = stream
        self.eps = 1e-5
        self.with_ratio = with_ratio
        self.max_len = max_len
        self.max_ratio = max_ratio
        self.reset()
    def _calculate_edit_operations(self, pred, target):
        """计算编辑操作：删除(De)、替换(Se)、插入(Ie)错误数"""
        m, n = len(pred), len(target)
        
        # 使用动态规划矩阵分析编辑操作(从pred->target)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # 初始化
        for i in range(m + 1):
            dp[i][0] = i  
        for j in range(n + 1):
            dp[0][j] = j 
        
        # 填充DP矩阵
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if pred[i-1] == target[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = min(dp[i-1][j] + 1,    
                                dp[i][j-1] + 1,     
                                dp[i-1][j-1] + 1)  
        
        # 回溯分析具体操作(从target->pred)
        de, se, ie = 0, 0, 0
        i, j = m, n
        
        while i > 0 or j > 0:
            if i > 0 and j > 0 and pred[i-1] == target[j-1]:
                # 字符匹配，无操作
                i -= 1
                j -= 1
            else:
                if i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + 1:
                    se += 1  # 替换错误
                    i -= 1
                    j -= 1
                elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
                    # 从target到pred需要插入 → 插入错误(Ie)
                    ie += 1  
                    i -= 1
                else:
                    # 从target到pred需要删除 → 删除错误(De)  
                    de += 1  
                    j -= 1
        
        return de, se, ie
    
    def _unify_evaluation_text(self, text):
        """统一评估文本的规范化处理"""
        # 1. 全角转半角
        text = unicodedata.normalize('NFKC', text)
        
        # 2. 繁体转简体（需要安装opencc）
        if cc:
            text = cc.convert(text)
        else:
            print("繁体转简体失败")
        
        # 3. 大写转小写
        text = text.lower()
        
        # 4. 移除所有空格
        text = re.sub(r'\s+', '', text)
        
        # 5. 其他符号转换
        punctuation_map = {
            '【': '[', '】': ']',
            '：': ':', '，': ',', '；': ';', '！': '!', '？': '?',
            '（': '(', '）': ')', '《': '<', '》': '>', '＂': '"', '＇': "'"
        }
        for cn_punct, en_punct in punctuation_map.items():
            text = text.replace(cn_punct, en_punct)
        return text
    
    def _normalize_text(self, text):
        text = ''.join(
            filter(lambda x: x in (string.digits + string.ascii_letters),
                   text))
        return text

    def __call__(self,
                 pred_label,
                 batch=None,
                 training=False,
                 *args,
                 **kwargs):
        if self.with_ratio and not training:
            return self.eval_all_metric(pred_label, batch)
        else:
            return self.eval_metric(pred_label)

    def eval_metric(self, pred_label, *args, **kwargs):
        preds, labels = pred_label
        correct_num = 0
        all_num = 0
        norm_edit_dis = 0.0
        for (pred, pred_conf), (target, _) in zip(preds, labels):
            if self.stream:
                assert len(labels) == 1
                pred, _ = stream_match(preds)
            if self.ignore_space:
                pred = pred.replace(' ', '')
                target = target.replace(' ', '')
            if self.is_filter:
                pred = self._normalize_text(pred)
                target = self._normalize_text(target)
            if self.is_lower:
                pred = pred.lower()
                target = target.lower()
            norm_edit_dis += Levenshtein.normalized_distance(pred, target)
            if pred == target:
                correct_num += 1
            all_num += 1
        self.correct_num += correct_num
        self.all_num += all_num
        self.norm_edit_dis += norm_edit_dis
        return {
            'acc': correct_num / (all_num + self.eps),
            'norm_edit_dis': 1 - norm_edit_dis / (all_num + self.eps),
        }
        
    def eval_all_metric(self, pred_label, batch=None, *args, **kwargs):
        preds, labels = pred_label
        correct_num = 0
        correct_num_real = 0
        correct_num_lower = 0
        correct_num_ignore_space = 0
        correct_num_ignore_space_lower = 0
        correct_num_ignore_space_symbol = 0
        all_num = 0
        norm_edit_dis = 0.0
        each_len_num = [0 for _ in range(self.max_len)]
        each_len_correct_num = [0 for _ in range(self.max_len)]
        each_len_norm_edit_dis = [0 for _ in range(self.max_len)]
        
        total_de = 0  # 总删除错误
        total_se = 0  # 总替换错误  
        total_ie = 0  # 总插入错误
        total_chars = 0  # 总字符数
        for (pred, pred_conf), (target, _) in zip(preds, labels):
            if self.stream:
                assert len(labels) == 1
                pred, _ = stream_match(preds)
            # 统一规范化处理
            unified_pred = self._unify_evaluation_text(pred)
            unified_target = self._unify_evaluation_text(target)
            # 原始比较（不做任何处理）
            if pred == target:
                correct_num_real += 1
            
            # 小写比较
            if pred.lower() == target.lower():
                correct_num_lower += 1
            
            # 忽略空格比较
            if pred.replace(' ', '') == target.replace(' ', ''):
                correct_num_ignore_space += 1
            
            # 忽略空格+小写比较
            if pred.replace(' ', '').lower() == target.replace(' ', '').lower():
                correct_num_ignore_space_lower += 1
            
            # 忽略空格+符号过滤比较
            norm_pred = self._normalize_text(pred.replace(' ', ''))
            norm_target = self._normalize_text(target.replace(' ', ''))
            if norm_pred == norm_target:
                correct_num_ignore_space_symbol += 1
            
            # 主acc使用统一规范化后的比较
            if unified_pred == unified_target:
                correct_num += 1
            
            # 计算编辑距离（使用统一规范化后的文本）
            dis = Levenshtein.normalized_distance(unified_pred, unified_target)
            norm_edit_dis += dis
            # ar,cr
            de, se, ie = self._calculate_edit_operations(unified_pred, unified_target)
            total_de += de
            total_se += se  
            total_ie += ie
            total_chars += len(unified_target)
            # 分组统计（使用原始长度）
            len_i = max(0, min(self.max_len, len(target)) - 1)
            
            if unified_pred == unified_target:
                each_len_correct_num[len_i] += 1
                
            each_len_num[len_i] += 1
            each_len_norm_edit_dis[len_i] += dis
            all_num += 1
        
        # 更新累积统计量
        self.correct_num += correct_num
        self.correct_num_real += correct_num_real
        self.correct_num_lower += correct_num_lower
        self.correct_num_ignore_space += correct_num_ignore_space
        self.correct_num_ignore_space_lower += correct_num_ignore_space_lower
        self.correct_num_ignore_space_symbol += correct_num_ignore_space_symbol
        self.all_num += all_num
        self.norm_edit_dis += norm_edit_dis
        self.total_de += total_de 
        self.total_se += total_se
        self.total_ie += total_ie
        self.total_chars += total_chars
        self.each_len_num = self.each_len_num + np.array(each_len_num)
        self.each_len_correct_num = self.each_len_correct_num + np.array(
            each_len_correct_num)
        self.each_len_norm_edit_dis = self.each_len_norm_edit_dis + np.array(
            each_len_norm_edit_dis)
        
        return {
            'acc': correct_num / (all_num + self.eps),  # 使用统一规范化后的比较结果
        }

    def get_all_metric(self):
        acc = 1.0 * self.correct_num / (self.all_num)
        acc_real = 1.0 * self.correct_num_real / (self.all_num + self.eps)
        acc_lower = 1.0 * self.correct_num_lower / (self.all_num + self.eps)
        acc_ignore_space = 1.0 * self.correct_num_ignore_space / (
            self.all_num + self.eps)
        acc_ignore_space_lower = 1.0 * self.correct_num_ignore_space_lower / (
            self.all_num + self.eps)
        acc_ignore_space_symbol = 1.0 * self.correct_num_ignore_space_symbol / (
            self.all_num + self.eps)
        nt = self.total_chars
        if nt > 0:
            ar = (nt - self.total_de - self.total_se - self.total_ie) / nt
            cr = (nt - self.total_de - self.total_se) / nt
        else:
            ar = 0.0
            cr = 0.0
            
        norm_edit_dis = 1 - self.norm_edit_dis / (self.all_num + self.eps)
        num_samples = self.all_num
        each_len_acc = (self.each_len_correct_num /
                        (self.each_len_num + self.eps)).tolist()
        each_len_norm_edit_dis = (1 -
                                ((self.each_len_norm_edit_dis) /
                                ((self.each_len_num) + self.eps))).tolist()
        each_len_num = self.each_len_num.tolist()
        print(self.correct_num)
        print(self.all_num)
        self.reset()
        return {
            'ar': ar,
            'cr': cr,
            'acc': acc, 
            'acc_real': acc_real,
            'acc_lower': acc_lower,
            'acc_ignore_space': acc_ignore_space,
            'acc_ignore_space_lower': acc_ignore_space_lower,
            'acc_ignore_space_symbol': acc_ignore_space_symbol,
            'each_len_num': each_len_num,
            'each_len_acc': each_len_acc,
            'each_len_norm_edit_dis': each_len_norm_edit_dis,
            'norm_edit_dis': norm_edit_dis,
            'num_samples': num_samples
        }


    def get_metric(self, training=False):
        """
        return metrics {
                 'acc': 0,
                 'norm_edit_dis': 0,
            }
        """
        if self.with_ratio and not training:
            return self.get_all_metric()
        acc = 1.0 * self.correct_num / (self.all_num + self.eps)
        norm_edit_dis = 1 - self.norm_edit_dis / (self.all_num + self.eps)
        num_samples = self.all_num
        self.reset()
        return {
            'acc': acc,
            'norm_edit_dis': norm_edit_dis,
            'num_samples': num_samples
        }


    def reset(self):
        self.correct_num = 0
        self.all_num = 0
        self.norm_edit_dis = 0
        self.correct_num_real = 0
        self.correct_num_lower = 0
        self.correct_num_ignore_space = 0
        self.correct_num_ignore_space_lower = 0
        self.correct_num_ignore_space_symbol = 0
        self.total_de = 0
        self.total_ie = 0
        self.total_se = 0
        self.total_chars = 0
        self.each_len_num = np.array([0 for _ in range(self.max_len)])
        self.each_len_correct_num = np.array([0 for _ in range(self.max_len)])
        self.each_len_norm_edit_dis = np.array(
            [0. for _ in range(self.max_len)])
        self.each_ratio_num = np.array([0 for _ in range(self.max_ratio)])
        self.each_ratio_correct_num = np.array(
            [0 for _ in range(self.max_ratio)])
        self.each_ratio_norm_edit_dis = np.array(
            [0. for _ in range(self.max_ratio)])
