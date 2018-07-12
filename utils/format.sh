#!/usr/bin/env bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/.."

if [ $(which clang-format) ]; then
  FILES=`find lib tests/unittests/ tools/ include examples -name \*.h -print -o -name \*.cpp -print`

  FARRAY=( $FILES ) # count the number of files to process
  echo  Formatting ${#FARRAY[@]} files

  echo "$FILES" | xargs -P8 -n1 clang-format -i
  echo "Done"
  exit
fi

echo "ERROR: can't find clang-format in your path."

