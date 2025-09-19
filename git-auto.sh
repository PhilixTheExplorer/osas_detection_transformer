#!/bin/bash
MESSAGE="$1"
if [ -z "$MESSAGE" ]; then
  echo "Usage: ./git-auto.sh \"commit message\""
  exit 1
fi

git add .
git commit -m "$MESSAGE"
git push
