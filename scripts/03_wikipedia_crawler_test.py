import json
from pathlib import Path
from collections import deque

PROJECT_DIR=Path(__file__).resolve().parent.parent

input_path=PROJECT_DIR /'data'/'raw'/'article_links.json'

with open(input_path,'r',encoding='utf-8') as f:
    titles=json.load(f)

queue=deque(titles)
visited=set()

print('Articles in queue: ',len(queue))
print('First article: ',queue[0])