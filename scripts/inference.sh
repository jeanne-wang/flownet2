# Example on MPISintel Final and Clean, with L1Loss on FlowNet2 model
python main.py --inference --model FlowNet2 --save_flow --inference_visualize \
--inference_dataset Nba2k_players_images --inference_dataset_root '/projects/grail/xiaojwan/nba2k_players_flow' \
--inference_dataset_dstype 'val' \
--inference_dataset_img1_dirname '/projects/grail/xiaojwan/2k_frames_masked_players' \
--inference_dataset_img2_dirname '/projects/grail/xiaojwan/2k_players_mesh_blender_est_camera' \
--number_gpus 1 --gpu_ids 1 \
--resume './pretrained_models/FlowNet2_checkpoint.pth.tar' \
--inference_n_batches 10 --inference_batch_size 1 \
--save './val_results_pretrained_flownet2_model'
