#!/bin/bash
for i in {1..20}
do
  python run_tumoroscope1000_only_tum_est_rp.py Results1000_t_only_tum_est_rp_$i prostate_tumoroscope_input/config_selected_spots_any_mutations_prostate.json True True True True False $i
done
