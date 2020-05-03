import urllib.request
print(urllib.request.urlopen("http://www.naver.com").read().decode('utf-8'))
