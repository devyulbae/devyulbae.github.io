#!/bin/bash

# Post Manager 실행
echo "Running Post Manager..."
python scripts/post_manager.py

# Jekyll 서버 실행
echo "Starting Jekyll server..."
bundle exec jekyll serve --livereload 