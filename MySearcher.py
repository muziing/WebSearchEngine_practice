import os
from math import log10

import jieba

import Spider


def highlight(item, query: str, side_len: int = 12) -> str:
    positions = list()
    query_words = list(jieba.cut(query))  # 把生成器强制转换为列表
    i = 0
    content_lower = item[2].lower()
    word_start_map = list()
    word_end_map = list()
    last_word_end = -1
    len_content_lower = len(content_lower)
    segments = list()
    for keyword in query_words:
        idx = content_lower.find(keyword.lower())
        positions.append(idx)
    for keyword in jieba.cut(content_lower):
        # 用于实现提取摘要时“整词切分”，避免出现截取摘要时首尾的词被截断
        current_word_start = last_word_end + 1
        current_word_end = current_word_start + len(keyword) - 1
        for _ in range(current_word_start, current_word_end + 1):
            word_start_map.append(current_word_start)
            word_end_map.append(current_word_end)
        last_word_end = current_word_end
    positions.sort()
    while i < len(positions):
        start_pos = max(positions[i] - side_len, 0)
        end_pos = min(positions[i] + side_len, len_content_lower - 1)
        # 用于实现合并相邻且有部分重合的摘要
        while (i < len(positions) - 1) and (positions[i + 1] - positions[i] < side_len * 2):
            end_pos = min(positions[i + 1] + side_len, len_content_lower - 1)
            i += 1
        start_ellipsis = '...' if start_pos > 0 else ''
        end_ellipsis = '...' if end_pos < len_content_lower else ''
        segments.append(start_ellipsis + item[2][word_start_map[start_pos]: word_end_map[end_pos]] + end_ellipsis)
        i += 1
    result = text = item[1] + '\n' + ''.join(segments)
    text_lower = text.lower()
    for keyword in query_words:
        # 高亮部分
        idx = text_lower.find(keyword.lower())
        if idx >= 0:
            ori_word = text[idx:idx + (len(keyword))]
            result = result.replace(ori_word, f'<span style="color:red";>{ori_word}</span>')
    return result


class MySearcher:
    """
    第十一次课升级的搜索类版本：
    改善文档频和文档长度加权的影响
    改善IDF权值
    采用BM25打分函数
    """

    def __init__(self):
        self.docs = list()  # 所有文档原始数据
        self.load_data()
        self.cache = dict()
        self.vocab = set()  # 索引词表
        self.lower_preprocess()
        jieba.load_userdict('./dict.txt')
        self.df = dict()
        self.avg_dl = 0
        self.build_cache()

    def load_data(self, data_file_name='./news_list.pkl'):
        if os.path.exists(data_file_name):
            self.docs = Spider.pickle_load(data_file_name)
        else:
            Spider.pickle_save(Spider.spider())
            self.docs = Spider.pickle_load(data_file_name)

    def search(self, query):
        result = None
        for keyword in jieba.cut(query.lower()):
            if keyword in self.cache:
                if result is None:
                    result = self.cache[keyword]
                else:
                    result = result & self.cache[keyword]
            else:
                result = set()
                break
        if result is None:
            result = set()
        sorted_result = self.rank(query, result)
        return sorted_result

    def rank(self, query, result_set):
        result = list()
        for doc_id in result_set:
            result.append([doc_id, self.score(self.docs[doc_id], query)])
        result.sort(key=lambda x: x[1], reverse=True)
        return result

    def render_search_result(self, query):
        """
        返回带有高亮和摘要的查询结果
        """
        count = 0
        result = ''
        for item in self.search(query):
            count += 1
            result += f'{count}[{item[1]}] {highlight(self.docs[item[0]], query)}\n'
        return result

    def score(self, item, query, k1=2, b=0.75):
        score = 0
        for keyword in jieba.cut(query.lower()):
            f = item[2].lower().count(keyword.lower())
            dl = len(item[2])
            tf = (f * (k1 + 1)) / (f + k1 * (1 - b + b * (dl / self.avg_dl)))
            idf = log10((len(self.docs) - self.df[keyword] + 0.5) / (self.df[keyword] + 0.5))
            score += tf * idf
        return score

    def build_cache(self):
        """
        用分词（用文档过滤词库）的方式构建索引
        """
        doc_id = 0
        doc_length_sum = 0
        for doc in self.docs:
            doc_word_set = set()
            doc_length_sum += len(doc[3])
            for word in jieba.cut_for_search(doc[3]):
                if word not in doc_word_set:
                    result_item = doc_id
                    if word not in self.cache:
                        self.cache[word] = {result_item}
                    else:
                        self.cache[word].add(result_item)
                    self.vocab.add(word)
                    doc_word_set.add(word)
                    if word in self.df:
                        self.df[word] += 1
                    else:
                        self.df[word] = 1
            doc_id += 1
        self.avg_dl = doc_length_sum / len(self.docs)

    def lower_preprocess(self):
        for doc_id in range(len(self.docs)):
            self.docs[doc_id].append(
                (self.docs[doc_id][1] + ' ' + self.docs[doc_id][2]).lower())
