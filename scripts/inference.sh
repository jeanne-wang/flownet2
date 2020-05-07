# Example on MPISintel Final and Clean, with L1Loss on FlowNet2 model
python main.py --inference --model FlowNet2 --save_flow --inference_visualize \
--inference_dataset Nba2k --inference_dataset_dstype 'val' \
--number_gpus 1 --gpu_ids 1 \
--resume './work_simple_1280/FlowNet2_model_best.pth.tar' \
--inference_n_batches 10 --inference_batch_size 1 \
--save './val_results_rastered_to_rastered'
