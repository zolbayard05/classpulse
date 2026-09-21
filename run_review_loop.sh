#!/bin/bash
> review_priority.log
for i in $(seq 1 40); do
  echo "=== RUN $i ===" >> review_priority.log
  timeout 240 python -u review_priority.py >> review_priority.log 2>&1
  rc=$?
  echo "=== run $i exit code: $rc ===" >> review_priority.log
  if grep -q "^Done\. analyzed=" review_priority.log; then
    tail -3 review_priority.log | grep -q "^Done" && break
  fi
  # check if fully done by comparing counts
  done_count=$(python -c "import json,os; print(len(json.load(open('review_priority.json'))) if os.path.exists('review_priority.json') else 0)" 2>/dev/null)
  echo "=== progress so far: $done_count ===" >> review_priority.log
done
echo "LOOP_COMPLETE" >> review_priority.log
