#!/bin/bash
for i in {1..20}
do
  python run_tumoroscope1000.py Results1000_t_my_rp_$i prostate_tumoroscope_input/config_selected_spots_any_mutations_prostate.json True True True True False $i
done
