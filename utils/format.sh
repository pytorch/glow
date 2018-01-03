#!/usr/bin/env bash

if [ $(which clang-format) ]; then
  FILES=`find lib tests/unittests/ tools/ include examples -name \*.h -print -o -name \*.cpp -print`

  FARRAY=( $FILES ) # count the number of files to process
  echo  Formatting ${#FARRAY[@]} files

  for F in $FILES; do
    clang-format -i $F
    echo -n .
  done
  echo
  echo "Done"
  exit
fi

echo "ERROR: can't find clang-format in your path."

