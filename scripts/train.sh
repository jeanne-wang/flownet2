# Example on MPISintel Final and Clean, with L1Loss on FlowNet2 model
python main.py --model FlowNet2 --loss=L1Loss --optimizer=Adam --optimizer_lr=1e-5 \
--training_dataset Nba2k --training_dataset_dstype 'train'  \
--validation_dataset Nba2k --validation_dataset_dstype 'val' \
--inference_dataset Nba2k --inference_dataset_dstype 'val' \
--crop_size 768 1280 --batch_size 1 \
--number_gpus 4 --gpu_ids 0 1 2 3 \
--validation_frequency 5 \
--render_validation --save_flow --inference_visualize \
--inference_n_batches 40 --inference_batch_size 1 \
--save './work_lr_1e_5'
