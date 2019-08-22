import requests

url = "https://www.amazon.cn/dp/B01ION3VWI/ref=sr_1_2?__mk_zh_CN=%E4%BA%9A%E9%A9%AC%E9%80%8A%E7%BD%91%E7%AB%99&keywords=python&qid=1561963273&s=books&sr=1-2"
kv = {"user-agent": "Mozilla/5.0"}  # 标准身份标识

r = requests.get(url, headers=kv)
r.raise_for_status()
r.encoding = r.apparent_encoding
print(r.request.headers)

# print(keyword.index("门"))  # 391879



