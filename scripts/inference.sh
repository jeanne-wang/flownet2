# Example on MPISintel Final and Clean, with L1Loss on FlowNet2 model
python main.py --inference --model FlowNet2 --save_flow --inference_visualize \
--inference_dataset Nba2k --inference_dataset_dstype 'train' \
--number_gpus 1 --gpu_ids 0 \
--resume './work/FlowNet2_model_best.pth.tar' \
--inference_n_batches 10 --inference_batch_size 1 \
--save './train_results_game_to_rastered'
