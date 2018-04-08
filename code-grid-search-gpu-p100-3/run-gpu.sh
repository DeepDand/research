#!/bin/bash
let pu1=4
let pu2=4
for u1 in 4096; do
  for u2 in 2 4 8 16; do
    echo $u1, $u2
    find utils/flags.py -type f -exec sed -i  "s/1_units', $pu1,/1_units', $u1,/g" {} \;
    find utils/flags.py -type f -exec sed -i  "s/2_units', $pu2,/2_units', $u2,/g" {} \;
    pu1=$u1
    pu2=$u2
    
    if [ "$pu1" -gt "$pu2" ]; then
      echo "Doing architecture [$u1, $u2]"
      CUDA_VISIBLE_DEVICES=2 python -u autoencoder.py 2>&1 | tee result_$u1'_'$u2.txt
    else
      echo "Do not do it"
    fi
  done
done

