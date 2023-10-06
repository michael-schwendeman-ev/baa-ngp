#!/bin/bash

# scene="CTTBLO004_ROI_1_transient_DS4"
# # exp_name="baangp_noise5e2_noba_noc2f_scaled"
# exp_name="baangp_nonoise_noba_noc2f_unscaled"
# data_root="/workspaces/data"

scene="synthetic_spherical20_noise5e2_DS4"
exp_name="baa-ngp-baseline-at"
data_root="/workspaces/data/schwendy"

python3 -u baangp/train_baangp.py   --scene $scene \
                                    --data-root $data_root \
                                    --save-dir $exp_name \
                                    --adjust-pose 
                                    # --c2f 0.1 0.5
