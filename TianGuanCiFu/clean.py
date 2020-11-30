from bs4 import BeautifulSoup
import re
from TianGuanCiFu.wordFreq import *
import time, random

import requests, urllib

# 读取html文件
def read_html(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

# 获取小组名称
def get_topic(bs4_html):
    topics = bs4_html.find("table", class_="olt").find_all("td")
    topic_content = ""
    index = 1
    line = ""
    for topic in topics:
        line = line + topic.text.strip(" ") + "\t"
        if index % 4 == 0:
            topic_content = topic_content + line + "\r\n"
            line = ""
        index = index + 1
    return topic_content
# 获取热点评论
def get_comments(bs4_html):
    comments = bs4_html.find_all("div", class_="main-bd")
    commentContent = ""
    for comment in comments:
        line = comment.h2.text.strip() + "\t" \
               + comment.find("div", class_="short-content").text.strip(" ")
        line = re.sub("\\(展开\\)|(\r\n)", "", line)
        commentContent = commentContent + line
        # commentContent = commentContent + comment.text.strip() + "\r\n"
    return commentContent

def write_file(write_path, content):
    with open(write_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
def truncate_file(file_path):
    with open(file_path, 'w') as f:
        f.truncate()
            
if __name__ == '__main__':
    user_agent = "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.198 Safari/537.36"
    headers = {'User-Agent': user_agent}


    headers = [
        "Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.153 Safari/537.36",
        "Mozilla/5.0 (Windows NT 6.1; WOW64; rv:30.0) Gecko/20100101 Firefox/30.0"
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_2) AppleWebKit/537.75.14 (KHTML, like Gecko) Version/7.0.3 Safari/537.75.14",
        "Mozilla/5.0 (compatible; MSIE 10.0; Windows NT 6.2; Win64; x64; Trident/6.0)",
        "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.198 Safari/537.36"
        ]

    ## 小组讨论内容获取
    # topic_html = read_html("D:\\sklearn\\TianGuanCiFu\\data\\topics.html")
    topic_spide_log = ""
    topic_content = ""
    for index in range(0, 500, 25):
        random_header = {'User-Agent': random.choice(headers)}
        time.sleep(random.randint(3, 5))
        topic_url = f"https://www.douban.com/group/706041/discussion?start={index}"
        topic_html = requests.get(url=topic_url, headers=random_header)
        
        if topic_html.status_code == 200:
            bs4_topic_html = BeautifulSoup(topic_html.text, 'html.parser')
            topic_content = topic_content + get_topic(bs4_topic_html)
        else:
            topic_spide_log = topic_spide_log + f"failed to spide {topic_url} " + "\n"
        time.sleep(random.randint(4, 7))
    
    # 评论
    comment_spide_log = ""
    comment_content = ""
    for index in range(0, 200, 20):
        # comment_html = read_html("D:\\sklearn\\TianGuanCiFu\\data\\comment.html")
        time.sleep(random.randint(3, 5))
        comment_url = f'https://movie.douban.com/subject/34908091/reviews?start={index}'
        comment_html = requests.get(url=comment_url, headers=random_header)
        
        if comment_html.status_code == 200:
            bs4_comment_html = BeautifulSoup(comment_html.text, 'html.parser')
            comment_content = comment_content + get_comments(bs4_comment_html)
        else:
            comment_spide_log = comment_spide_log + f"failed to spide {comment_url}" + "\n"
        time.sleep(random.randint(4, 7))
    
    # 指定数据路径
    topic_write_path = "D:\\sklearn\\TianGuanCiFu\\data\\topic.txt"
    topic_spide_log_path = "D:\\sklearn\\TianGuanCiFu\\data\\topic_log.txt"
    
    comment_write_path = "D:\\sklearn\\TianGuanCiFu\\data\\comment.txt"
    comment_spide_log_path = "D:\\sklearn\\TianGuanCiFu\\data\\comment_log.txt"
    
    
    # 写入获取的数据
    write_file(topic_spide_log_path, topic_spide_log)
    write_file(topic_write_path, topic_content)
    
    write_file(comment_spide_log_path, comment_spide_log)
    write_file(comment_write_path, comment_content)