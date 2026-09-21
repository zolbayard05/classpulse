#!/bin/bash
for i in $(seq 1 15); do
  echo "=== CHUNK RUN $i ===" >> autolabel_full.log
  timeout 180 python -u autolabel.py 30 >> autolabel_full.log 2>&1
  rc=$?
  echo "=== chunk $i exit code: $rc ===" >> autolabel_full.log
  remaining=$(python -c "
from roboflow import Roboflow
rf = Roboflow(api_key='8VBIDNN9UJookIj75lfb')
project = rf.workspace('zolbayar-danzan-zeno').project('classpulse-g8l1n')
processed = set(open('processed_ids.txt').read().split()) if __import__('os').path.exists('processed_ids.txt') else set()
res = project.search(in_dataset=False, limit=300, fields=['id'])
print(len([r for r in res if r['id'] not in processed]))
" 2>/dev/null | tail -1)
  echo "=== remaining after chunk $i: $remaining ===" >> autolabel_full.log
  if [ "$remaining" = "0" ]; then
    echo "ALL_DONE" >> autolabel_full.log
    break
  fi
done
