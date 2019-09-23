import requests,random,time

url = 'http://172.19.241.251:8080/get_comment'
headers = {'Host': '172.19.241.251:8080',
           'Content-Length': '201',
           'Accept': 'text/plain, */*; q=0.01',
           'Origin': 'http://172.19.241.251:8080',
           'X-Requested-With': 'XMLHttpRequest',
           'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/73.0.3683.86 Safari/537.36',
           'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
           'Referer': 'http://172.19.241.251:8080/comment?id=1',
           'Accept-Encoding': 'gzip, deflate',
           'Accept-Language': 'zh-CN,zh;q=0.9',
           'Cookie': 'JSESSIONID=C94EAEA1526AE0AAEC27BA1E28151BA4',
           'Pragma': 'no-cache',
           'Cache-Control': 'no-cache',
           'Connection': 'keep-alive'}
payload = 'scorea1=7&scoreb1=3&scorea2=7&scoreb2=3&scorea3=5&scoreb3=3&scorea4=5&scoreb4=3&scorea5=5&scoreb5=4&' \
          'scorea6=5&scoreb6=6&scorea7=5&scoreb7=3&scorea8=5&scoreb8=3&scorea9=5&scoreb9=4&scorea10=5&scoreb10=6'


resp = requests.post(url, headers=headers, data=payload)

# for i in range(3):
#     t = random.uniform(0.5, 2)
#     time.sleep(t)
#
print(resp.text)