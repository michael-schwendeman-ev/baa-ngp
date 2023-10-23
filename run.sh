#!/bin/bash

# scene="CTTBLO004_ROI_1_transient_DS4"
# # exp_name="baangp_noise5e2_noba_noc2f_scaled"
# exp_name="baangp_nonoise_noba_noc2f_unscaled"
# data_root="/workspaces/data"

# scene="synthetic_spherical20_nonoise_goodtestimages_DS4"
# exp_name="baa-ngp-10pixelnoise-noat-check"
# data_root="/workspaces/data/schwendy"

python3 -u baangp/train_baangp.py   --scene "synthetic_spherical20_nonoisefinal_DS4" \
                                    --data-root "/workspaces/data/schwendy" \
                                    --save-dir "baa-ngp-noat" \
                                    --adjustment-type "none" \
                                    --c2f 0.1 0.5 \
                                    --bounding-box-buffer 10.0 \
                                    --learning-rates "1.0e-2" "1.0e-4" "3.0e-3" "3.0e-5" \
                                    --regularizations "1.0e-6" "0.0" \
                                    --resolutions 400.0 400.0 100.0 0.5 \
                                    --max-steps 10000 \
                                    --noise 0.0 0.0 0.0 0.0 0.0 0.0
                                    # --noise 0.087 0.087 0.087 0.0 0.0 0.0
                                    # --noise 0.00066 0.00066 0.00066 0.0 0.0 0.0

                                    

# --scene synthetic_spherical20_nonoise_goodtestimages_DS4 --data-root /workspaces/data/schwendy --save-dir baa-ngp-5degnoise-fullat --adjustment-type full --c2f 0.1 0.5 --bounding-box-buffer 5.0 --learning-rates 1.0e-2 1.0e-4 3.0e-3 3.0e-5 --regularizations 1.0e-6 0.0 --resolutions 400.0 400.0 100.0 0.5 --max-steps 10000 --noise 0.087 0.087 0.087 0.0 0.0 0.0
# --scene synthetic_spherical20_nonoise_goodtestimages_DS4 --data-root /workspaces/data/schwendy --save-dir baa-ngp-5degnoise-fullat-noc2f --adjustment-type full --bounding-box-buffer 5.0 --learning-rates 1.0e-2 1.0e-4 3.0e-3 3.0e-5 --regularizations 1.0e-6 0.0 --resolutions 400.0 400.0 100.0 0.5 --max-steps 10000 --noise 0.087 0.087 0.087 0.0 0.0 0.0