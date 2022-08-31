#!/bin/bash

for i in {1..20}
do
    python  main_diff_config_combined.py  $i Results_combined/ configs/  0
done
