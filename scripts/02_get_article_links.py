import requests
from pathlib import Path
import json

API_URL='http://en.wikipedia.org/w/api.php'

headers={
    'User-Agent': 'ADM-Hackathon/0.1 (cerdabarrera.2242483@studenti.uniroma1.it)'
}

params={
    'action': 'query',
    'format': 'json',
    'prop': 'links',
    'titles': 'Algorithm',
    'plnamespace':0,
    'pllimit':500
}

response=requests.get(
    API_URL,
    params=params,
    headers=headers,
    timeout=31
)

response.raise_for_status()

data=response.json()

pages=data['query']['pages']

PROJECT_DIR=Path(__file__).resolve().parent.parent

#extract titles from API responde

titles=[]

for page in pages.values():
    links=page.get('links',[])

    for link in links:
        titles.append(link['title'])

#remove duplicates while preserving order

titles=list(dict.fromkeys(titles))

#save the titles

output_path=PROJECT_DIR/'data'/'raw'/'article_links.json'

with open(output_path,'w',encoding='utf-8') as f:
    json.dump(titles,f,ensure_ascii=False,indent=4)

print('Number of articles discovered: ',len(titles))
print('Saved to: ',output_path)