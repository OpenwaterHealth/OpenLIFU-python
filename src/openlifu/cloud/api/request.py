from openlifu.cloud.utils import to_json
from openlifu.cloud.const import API_URL, DEBUG
import requests


class Request:

    def __init__(self):
        self.headers = {}

    def get(self, url: str) -> str:
        response = requests.get(API_URL + url, headers=self.headers)
        if DEBUG:
            print(f"GET: {url}, status_code: {response.status_code}\nresponse: {response.text}")
        response.raise_for_status()
        return response.text

    def get_bytes(self, url: str) -> bytes:
        response = requests.get(API_URL + url, headers=self.headers)
        if DEBUG:
            print(f"GET bytes: {url}, status_code: {response.status_code}")
        response.raise_for_status()
        return response.content

    def post(self, url: str, dto) -> str:
        response = requests.post(API_URL + url, data=to_json(dto), headers=self.headers)
        if DEBUG:
            print(f"POST: {url}, body: {to_json(dto)}, status_code: {response.status_code}\nresponse: {response.text}")
        response.raise_for_status()
        return response.text

    def post_bytes(self, url: str, data) -> str:
        response = requests.post(API_URL + url, data=data, headers=self.headers)
        if DEBUG:
            print(f"POST bytes: {url}, status_code: {response.status_code}\nresponse: {response.text}")
        response.raise_for_status()
        return response.text

    def put(self, url: str, dto) -> str:
        response = requests.put(API_URL + url, data=to_json(dto), headers=self.headers)
        if DEBUG:
            print(f"PUT: {url}, body: {to_json(dto)}, status_code: {response.status_code}\nresponse: {response.text}")
        response.raise_for_status()
        return response.text

    def delete(self, url: str) -> str:
        response = requests.delete(API_URL + url, headers=self.headers)
        if DEBUG:
            print(f"DELETE: {url}, status_code: {response.status_code}\nresponse: {response.text}")

        response.raise_for_status()
        return response.text
