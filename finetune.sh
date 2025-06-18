# run me from root directory !!!

#python3 ddsp/finetune.py --dataset_path ~/datasets/echo_75/groove \
#                        --output_path finetune_outputs/groove_75_full_dataset \
#                        --initial_model ./ArtistProtectModels/SingleEchoes/DDSP/groove_clean.pkl \
#                        --num_epochs 30

 python3 ddsp/finetune.py --dataset_path ~/datasets/echo_75/groove \
                         --dataset_size 10 \
                         --output_path finetune_outputs/groove_75_10_samples_test \
                         --initial_model ./ArtistProtectModels/SingleEchoes/DDSP/groove_clean.pkl \
                         --num_epochs 30
                         
