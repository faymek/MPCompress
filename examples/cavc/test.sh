python examples/cavc/run_eval_cavc.py \
    --config examples/cavc/config/eval_base.yaml examples/cavc/config/eval_cavc.yaml \
    --checkpoint "" \
    --task uvg_val_rec \
    --head "" \
    --quality 1.0 \
    --cuda --recon 2 --real \
    --output_dir exp/eval_uvg_val_cavc_lpips_bpp0.1_1204
