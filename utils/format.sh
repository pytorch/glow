#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/.."

CLANG_COMMAND=${CLANG_COMMAND:-clang-format}

print_usage() {
  echo "Missing format option:"
  echo "  Run: './format.sh fix' if you want to fix format."
  echo "  Run: './format.sh check' if you want to check format."
}

fix_format() {
  find lib tests/unittests/ tools/ include examples \
    -name \*.h -print0 \
    -o -name \*.cpp -print0 \
    -o -name \*.cl -print0 \
  | xargs -0 -P8 -n1 $CLANG_COMMAND -i;
}

check_format() {
  touch pre.status
  touch post.status

  git status > pre.status
  fix_format
  git status > post.status

  if ! diff pre.status post.status; then
    echo "ERROR: files need to be formatted"
    echo "***pre.status***"
    cat pre.status
    echo "***post.status***"
    cat post.status
    echo "***git diff***"
    git diff

    exit 1
  fi
}

if [[ -n ${1:0} ]]; then
  if ! command -v $CLANG_COMMAND; then
    echo "ERROR: can't find clang-format in your path. Tried: $CLANG_COMMAND"
    exit 1
  fi

  if [ "$1" = "fix" ]; then
    echo "Running fix format."
    fix_format
  elif [ "$1" = "check" ]; then
    echo "Running check format."
    check_format
  else
    print_usage
    exit 1
  fi
else
  print_usage
  exit 1
fi

