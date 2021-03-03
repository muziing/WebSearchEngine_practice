from os.path import exists as os_file_exists
from math import log10

import jieba

import Spider


def highlight(item, query: str, side_len: int = 12) -> str:
    """
    返回带有HTML高亮的标题及摘要文本

    :param item:
    :param query: 查询内容
    :param side_len: 摘要中关键字两侧字符串长度
    :return:
    """
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
    result = text = f'<a href="{item[0]}"{item[1]}</a>' + '</br>' + ''.join(segments)  # TODO 修复a标签导致标题全部高亮的问题
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
    改善文档频和文档长度加权的影响
    采用BM25打分函数
    """

    def __init__(self, data_file_name='./news_list.pkl'):
        self.docs = list()  # 所有文档原始数据, [link, title, content, lower_preprocess]
        self.load_data(data_file_name)
        self.index = dict()  # 倒排索引
        self.vocab = set()  # 索引词表
        self.lower_preprocess()  # 为提高召回，额外保存一份全小写的文章标题内容拼接字符串
        # jieba.load_userdict('./dict.txt')
        self.df = dict()
        self.avg_dl = 0
        self.build_index()

    def load_data(self, data_file_name: str) -> None:
        if os_file_exists(data_file_name):
            self.docs = Spider.pickle_load(data_file_name)
        else:
            Spider.pickle_save(Spider.spider())
            self.docs = Spider.pickle_load(data_file_name)

    def lower_preprocess(self):
        for doc_id in range(len(self.docs)):
            self.docs[doc_id].append(
                (self.docs[doc_id][1] + ' ' + self.docs[doc_id][2]).lower())

    def build_index(self):
        """
        倒排索引
        """
        doc_id = 0
        doc_length_sum = 0  # 为获取avg_dl
        for doc in self.docs:
            doc_word_set = set()  # 文章中所有词的词表
            doc_length_sum += len(doc[3])
            for word in jieba.cut_for_search(doc[3]):
                if word not in doc_word_set:
                    result_item = doc_id
                    if word not in self.index:
                        self.index[word] = {result_item}
                    else:
                        self.index[word].add(result_item)
                    self.vocab.add(word)
                    doc_word_set.add(word)
                    if word in self.df:
                        self.df[word] += 1
                    else:
                        self.df[word] = 1
            doc_id += 1
        self.avg_dl = doc_length_sum / len(self.docs)

    def search(self, query: str):
        """
        搜索方法

        :param query: 查询内容
        :return: 返回排序后的查询结果[doc_id, score, query]
        """
        result = None
        for keyword in jieba.cut(query.lower()):
            if keyword in self.index:
                if result is None:
                    result = self.index[keyword]
                else:
                    result = result & self.index[keyword]
            else:
                result = set()
                break
        if result is None:
            result = set()
        sorted_result = self.rank(query, result)
        return sorted_result

    def rank(self, query: str, result_set):
        result = list()
        for doc_id in result_set:
            result.append([doc_id, self.score(self.docs[doc_id], query)])
        result.sort(key=lambda x: x[1], reverse=True)
        return result

    def render_search_result(self, query: str) -> tuple:
        """
        返回带有高亮和摘要的查询结果
        """
        result = list()
        for item in self.search(query):
            result.append((item[1], highlight(self.docs[item[0]], query)))
        return tuple(result)

    def score(self, item, query, k1=2, b=0.75) -> int:
        """
        采用BM25打分函数

        :param item: 单篇文档
        :param query: 查询内容
        :param k1: BM25参数
        :param b: BM25参数，调节文本长度对相关性的影响
        :return: 返回得分
        """
        score = 0
        for keyword in jieba.cut(query.lower()):
            f = item[2].lower().count(keyword.lower())
            dl = len(item[2])
            tf = (f * (k1 + 1)) / (f + k1 * (1 - b + b * (dl / self.avg_dl)))
            idf = log10((len(self.docs) - self.df[keyword] + 0.5) / (self.df[keyword] + 0.5))
            score += tf * idf
        return score


if __name__ == '__main__':
    my_searcher = MySearcher()
    print(my_searcher.render_search_result("华为手机")[0])
