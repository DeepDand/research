#!/bin/bash
pu1=64
pu2=49
for u1 in 4 8 16 32 64 128 256 512; do
  for u2 in 2 4 8 16 32 64 128 256; do
    echo $u1, $u2
    find utils/flags.py -type f -exec sed -i "" "s/1_units', $pu1,/1_units', $u1,/g" {} \;
    find utils/flags.py -type f -exec sed -i "" "s/2_units', $pu2,/2_units', $u2,/g" {} \;
    pu1=$u1
    pu2=$u2
    python -u autoencoder.py 2>&1 | tee result_$u1_$u2.txt
  done
done

