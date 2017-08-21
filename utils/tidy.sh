#!/usr/bin/env bash

if [ $(which clang-tidy) ]; then
  FILES=`find src lib include tools unittests -name \*.h -print -o -name \*.cpp -print`

  FARRAY=( $FILES ) # count the number of files to process
  echo  Inspecting ${#FARRAY[@]} files

  clang-tidy $FILES -p ../build_/ $1
  echo
  echo "Done"
  exit
fi

echo "ERROR: can't find clang-tidy in your path."

