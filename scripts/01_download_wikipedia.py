import requests
from bs4 import BeautifulSoup
from pathlib import Path
import json

#project directory
PROJECT_DIR= Path(__file__).resolve().parent.parent

#Wikipedia article
title="Alan Turing"

url=(
    'https://en.wikipedia.org/w/rest.php/v1/page/' \
    f'{title.replace(' ','_')}/with_html'
)

headers={
    'User-Agent': 'ADM-Hackathon/0.1 (cerdabarrera.2242483@studenti.uniroma1.it)'
}

#request data from Wikipedia
response=requests.get(url,headers=headers,timeout=31)
response.raise_for_status()

article=response.json()

#extract the HTML and convert into text
html=article['html']
soup=BeautifulSoup(html,'html.parser')

for tag in soup(['script','style','sup']):
    tag.decompose()

text=soup.get_text(separator=' ',strip=True)

#Store the original response
raw_path=PROJECT_DIR/'data'/'raw'/'alain_turing.json'

with open(raw_path,'w',encoding='utf-8') as f:
    json.dump(article,f,ensure_ascii=False,indent=4)

#inspect the results
print('Title: ',article['title'])
print('Page ID: ',article['id'])
print('Number of words: ',len(text.split()))
print('\nFirst 500 characters of the article: ')
print(text[:500])
print('\nSaved to: ',raw_path)