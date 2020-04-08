# Example on MPISintel Final and Clean, with L1Loss on FlowNet2 model
python main.py --batch_size 8 --model FlowNet2 --loss=L1Loss --optimizer=Adam --optimizer_lr=1e-4 \
--training_dataset Nba2k --training_dataset_dstype 'train'  \
--validation_dataset Nba2k --validation_dataset_dstype 'val'
