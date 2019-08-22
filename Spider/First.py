#-*- coding:utf8 -*-

import requests
url = "https://www.baidu.com"
def getHtmlText(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        print(response.encoding, response.apparent_encoding)
        response.encoding = response.apparent_encoding  # 正常写法
        # response.encoding = 'ISO-8859-1'
        return response.text[:500]
    except:
        return "产生异常"

print(getHtmlText(url))




# def mothos_request(url):
#     kv = {'pwewewe': 3}
#     try:
#         response = requests.request(method='get',url=url,params=kv)
#         response.raise_for_status()
#         response.encoding = response.apparent_encoding
#         return response.text[:500]
#     except:
#         return "产生异常"