#!/bin/bash

FILES=$(find . -name "chkp_*")

for f in $FILES; do
  with_zip=${f##*chkp_}
  num=$(cut -d '_' -f 1 <<< "$with_zip")
  rem=$(($num % $1))
  if [ $rem -ne 0 ]; then
    rm $f
  fi
done
