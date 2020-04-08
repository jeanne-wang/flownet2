# Example on MPISintel Final and Clean, with L1Loss on FlowNet2 model
python main.py --model FlowNet2 --loss=L1Loss --optimizer=Adam --optimizer_lr=1e-4 \
--training_dataset Nba2k --training_dataset_dstype 'train' --training_dataset_img1_dirname '2k_mesh_rasterized' \
--validation_dataset Nba2k --validation_dataset_dstype 'val' --validation_dataset_img1_dirname '2k_mesh_rasterized' \
--inference_dataset Nba2k --inference_dataset_dstype 'val' --inference_dataset_img1_dirname '2k_mesh_rasterized' \
--crop_size 512 512 --batch_size 4 \
--number_gpus 2 --gpu_ids 0 1 \
--validation_frequency 10 \
--render_validation --save_flow --inference_visualize \
--inference_n_batches 40 --inference_batch_size 1 \
--save './work_simple'


