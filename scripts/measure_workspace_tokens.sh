#!/bin/bash
# Measure token usage in OpenClaw workspace files
# Rough estimate: word count * 1.3

TOTAL=0
for f in ~/.openclaw/workspace/*.md; do
  T=$(wc -w "$f" | awk '{print int($1 * 1.3)}')
  echo "$f: ~$T tokens"
  TOTAL=$((TOTAL + T))
done
echo "---"
echo "TOTAL: ~$TOTAL tokens (target: < 1500)"
[ $TOTAL -lt 1500 ] && echo "PASS: Under budget" || echo "OVER BUDGET"
