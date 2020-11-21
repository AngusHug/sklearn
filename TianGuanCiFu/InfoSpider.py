import re
import requests

class MovieInfo:
    def __init__(self, web_url, headers, encoding):
        self.request_url = web_url
        self.headers = headers
        self.encoding = encoding
    
    def get_web(self, web_url, headers, encoding):
        response = requests.get()