from time import sleep
import requests
from lxml import etree
import pickle


def spider():
    url = 'http://news.163.com/special/0001386F/rank_tech.html'
    headers = {
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4412.0 Safari/537.36 Edg/90.0.796.0'}
    r = requests.get(url, headers=headers)
    sel = etree.HTML(r.text)
    link_set = set()
    news_list = list()
    count = 0
    for item in sel.xpath('//td/a'):
        title = item.text
        link = item.attrib['href']
        # print(link, title)
        if link not in link_set:
            r = requests.get(link, headers=headers)
            sel = etree.HTML(r.text)
            text_block = sel.xpath("//div[@id='content']/div[@class='post_body']")
            # print(''.join(text_block[0].itertext()))
            content = ''.join(text_block[0].xpath('./p/text()'))
            title = sel.xpath('//h1/text()')[0]
            news_list.append([link, title, content])
            link_set.add(link)
        count += 1
        sleep(0.5)
        if count % 20 == 0:
            print(count, 'processed.', end='\r')

    return news_list


def pickle_save(to_save):
    if to_save:
        with open('./news_list.pkl', 'wb') as p:
            pickle.dump(str(spider()), p, True)
    else:
        print("news_list为空！")


def pickle_load(file):
    with open(file, 'rb') as p:
        return eval(pickle.load(p))


if __name__ == '__main__':
    spider()
