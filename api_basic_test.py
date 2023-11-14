import requests
import json

base_url = 'http://nomad-lab.eu/prod/v1/api/v1'

response = requests.post(
    f'{base_url}/entries/query',
    json={
        'query': {
            'external_db': 'The Perovskite Database Project'
        },
        'pagination': {
            'page_size': 1
        },
        'required': {
            'include': ['entry_id']
        }
    })
response_json = response.json()
print(json.dumps(response.json(), indent=2))