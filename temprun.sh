#!/bin/bash

for i in 5 6 7 8 9
do
  cp -r 3d_$i"buttons" ~/
  python train.py -d ~/3d_$i"buttons" -s 3d_$i"buttons_out" -ne 5 -lr .0005 -ub -sr 51 -opt novograd -at -device cuda:1
  rm -r ~/3d_$i"buttons"
done
