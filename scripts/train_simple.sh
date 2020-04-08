# Example on MPISintel Final and Clean, with L1Loss on FlowNet2 model
python main.py --model FlowNet2 --loss=L1Loss --optimizer=Adam --optimizer_lr=1e-4 \
--number_gpus 4 --gpu_ids 0 2 3 4 \
--crop_size 768 1280 --batch_size 2 \
--training_dataset Nba2k --training_dataset_dstype 'train'  --training_dataset_img1_dirname '2k_mesh_rasterized' \
--validation_dataset Nba2k --validation_dataset_dstype 'val' --validation_dataset_img1_dirname '2k_mesh_rasterized' \
--validation_frequency 10 --validation_n_batches 10 --render_validation --save_flow --inference_visualize
