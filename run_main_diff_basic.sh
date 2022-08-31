#!/bin/bash

for i in {1..20}
do
    python  main_diff_config_basic.py  $i Results_tum/ configs/  0
done
