import re
import json
import ast
import nltk
from pathlib import Path
from collections import defaultdict
from nltk.tokenize import TweetTokenizer

nltk.download('punkt')


class HFMrocessor:
    def __init__(self, img_root='./HFM/HFM_all'):
        # 初始化路径配置
        self.img_root = Path(img_root)
        self.img_root_str = str(img_root)  # 保留原始路径字符串
        self.bese = './HFM_all'

        # 初始化NLTK组件
        self.tknzr = TweetTokenizer()

        # 配置过滤规则（包含论文所有禁用词）
        self.filter_regex = re.compile(
            r'(?:^|\s)(exgag|sarcasm|sarcastic|irony|ironic|jokes?|humour|humor|reposting)(?:$|\s)',
            flags=re.IGNORECASE
        )

        # 初始化统计系统
        self.vocab = defaultdict(int)
        self.filter_stats = defaultdict(int)

    def _clean_text(self, text, is_train=False):
        # 替换提及
        text = re.sub(r'@\w+', '<user>', text)

        # 移除URL
        text = re.sub(r'https?://\S+|www\.\S+', '', text)

        # 合并连续句点（超过2个的变为...）
        text = re.sub(r'\.{3,}', '...', text)

        # 移除标点后的多余空格（保留1个空格）
        text = re.sub(r'([!?.,])\s+', r'\1 ', text)

        # 清理多余空格
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def _parse_line(self, line):
        """增强型数据解析方法"""
        try:
            # 统一引号处理
            sanitized = line.strip().replace("'", '"')

            # 尝试JSON解析
            data = json.loads(sanitized)
            if isinstance(data, list) and len(data) >= 3:
                return data
        except json.JSONDecodeError:
            try:
                # 回退到AST解析
                return ast.literal_eval(line)
            except:
                pass
        return None

    def _validate_data(self, data):
        """数据格式验证"""
        return data and len(data) >= 3 and str(data[0]).isdigit()

    def _check_filters(self, text, img_id):
        """综合过滤检查"""
        img_path = f"{self.bese}/{img_id}.jpg"
        # 文本内容过滤
        if self.filter_regex.search(text.lower()):
            return "content_filter"

        # 图片存在性检查
        if not Path(img_path).exists():
            return "missing_image"
        return None

    def process_file(self, input_path, output_path, dataset_type):
        """核心处理流程"""
        stats = defaultdict(int)

        with open(input_path, 'r', encoding='utf-8', errors='ignore') as f_in, \
                open(output_path, 'w', encoding='utf-8') as f_out:

            for line in f_in:
                stats['total'] += 1
                data = self._parse_line(line)

                # 数据格式验证
                if not self._validate_data(data):
                    stats['invalid_format'] += 1
                    continue

                img_id, text = data[0], data[1]

                # 标签处理
                try:
                    label = int(data[-1])  # 使用最后一位作为标签
                except (IndexError, ValueError):
                    stats['label_error'] += 1
                    continue

                # 执行过滤
                filter_reason = self._check_filters(text, img_id)
                if filter_reason:
                    stats[filter_reason] += 1
                    continue

                # 文本清洗（区分训练模式）
                is_train = (dataset_type == 'train')
                cleaned_text = self._clean_text(text, is_train)

                if not cleaned_text.strip():
                    stats['empty_text'] += 1
                    continue

                # 生成最终输出路径
                img_path = f"{self.img_root_str}/{img_id}.jpg"

                # 写入处理结果
                f_out.write(f"{img_path}\t{label}\t{cleaned_text}\n")
                stats['valid'] += 1

            # 打印统计信息
            self._print_stats(stats, input_path)

    def _print_stats(self, stats, filename):
        """统计信息输出"""
        total = stats['total']
        print(f"\n=== 处理结果：{Path(filename).name} ===")
        print(f"总样本数：{total}")
        print(f"有效样本：{stats['valid']} ({stats['valid'] / total:.1%})")
        print(f"过滤样本：{total - stats['valid']} ({(total - stats['valid']) / total:.1%})")

        print("\n详细过滤原因：")
        for reason in ['invalid_format', 'label_error', 'content_filter', 'missing_image', 'empty_text']:
            count = stats.get(reason, 0)
            if count > 0:
                print(f"- {reason.replace('_', ' ')}: {count} ({count / total:.1%})")
if __name__ == "__main__":
    processor = HFMrocessor()

    # 处理训练集（单标签）
    processor.process_file(
        './train.txt',
        './formatted_train.txt',
        dataset_type='train'
    )

    # 处理测试集（双标签）
    processor.process_file(
        './test.txt',
        'formatted_test.txt',
        dataset_type='test'
    )

    # 处理验证集（双标签）
    processor.process_file(
        './valid.txt',
        'formatted_valid.txt',
        dataset_type='valid'
    )