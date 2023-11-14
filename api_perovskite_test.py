import requests

import pandas as pd

url = 'http://nomad-lab.eu/prod/v1/api/v1/entries/query'

query =  {
    "external_db:any": [
      "The Perovskite Database Project"
    ]
  }
pagination = {
    'page_size': 1
},
required = {
    'results': {
        'material': {
            'chemical_formula_reduced': "*"
        }
    }
}

page_after_value = None

rows = []

while True:
    print(len(rows))
    response = requests.post(
        url, json=dict(
            query=query,
            required=required,
            pagination=dict(
                page_after_value=page_after_value,
                page_size=100
            )
        )
    )
    if response.status_code != 200:
        print("Failed request, code: ", response.status_code)
        continue
    data = response.json()
    if len(data['data']) == 0:
        break
    try:
        page_after_value = data['pagination']['next_page_after_value']
    except KeyError:
        print("no next page found.")
        break
    for entry in data['data']:
        #print(entry['entry_id'])
        #for key, val in entry["results"].items():
        #    print(5*'-')
        #    print(key, val)
    #break
        row = entry["results"]
        try:
            row["band_gap"] = row["properties"]["electronic"]["band_structure_electronic"][0]["band_gap"][0]["value"]
            row["band_gap"] *= 1/1.60218E-19 # convert to eV
            row["entry_id"] = entry["entry_id"]
            #del row["band_structure_electronic"]
        except KeyError:
            pass
        rows.append(row)

df = pd.json_normalize(rows)

col_rename = lambda name: name.split('.')[-1]
df = df.rename(columns=col_rename)
print(df.describe())
print(df.keys())
print(df.corr())

df.to_csv("psc_data.csv")
    



