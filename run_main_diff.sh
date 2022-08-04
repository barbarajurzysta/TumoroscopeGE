#!/bin/bash

for i in {1..10}
do
    python  main_diff_config_.py  $i Results/ configs/  0
done
