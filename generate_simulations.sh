#!/bin/bash

for i in {1..10}
do
    python  generate_simulations.py  $i 'test_sim/' configs/  0
done
