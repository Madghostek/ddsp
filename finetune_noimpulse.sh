# run me from root directory !!!


 python3 ddsp/finetune.py --dataset_path ~/datasets/echo_75/groove \
                         --output_path finetune_outputs/groove_75_noimpulse \
                         --initial_model ./ArtistProtectModels/SingleEchoes/DDSP/groove_clean.pkl \
                         --num_epochs 30 \
                         --no-impulse
                         
