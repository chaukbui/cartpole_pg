#!/bin/bash

rm -rf output

python main.py --eps_length 500 --train_eps 2000 --test_eps 50 --lr 0.01 --lr_dec 0.5 --render
