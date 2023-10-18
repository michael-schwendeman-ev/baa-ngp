#!/bin/bash

# scene="CTTBLO004_ROI_1_transient_DS4"
# # exp_name="baangp_noise5e2_noba_noc2f_scaled"
# exp_name="baangp_nonoise_noba_noc2f_unscaled"
# data_root="/workspaces/data"

scene="synthetic_spherical20_50pixelsnoise_DS4"
exp_name="baa-ngp-at-fullpose-aabb-c2f-testnewcode"
data_root="/workspaces/data/schwendy"

python3 -u baangp/train_baangp.py   --scene $scene \
                                    --data-root $data_root \
                                    --save-dir $exp_name \
                                    --adjustment-type "full" \
                                    --c2f 0.1 0.5 \
                                    --bounding-box-buffer 5.0 \
                                    --learning-rates "1.0e-2" "1.0e-4" "3.0e-3" "3.0e-5" \
                                    --regularizations "1.0e-6" "0.0" \
                                    --resolutions 400.0 400.0 100.0 0.5 \
                                    --max-steps 10000 \
                                    --noise 0.0 0.0 0.0 0.0 0.0 0.0

                                    
