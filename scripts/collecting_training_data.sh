python src/preprocess/saving_training_data.py \
    ++batch_dims=[8,150] \
    ++waymax_conf.drop_remainder=True \
    ++waymax_conf.path="${WOMD_TRAIN_PATH}" \
    ++data_conf.path_to_processed_map_route="${PRE_PROCESS_TRAIN_PATH}" \
    ++metric_conf.intention_label_path="${INTENTION_TRAIN_PATH}" \
    ++save_path="${TRAINING_DATA_PATH}"

# You should drop the last batch here. Since the overall training data is extremely large (~487,000), it do not affect the performance.
# In our paper the number of data we used is 486,375 (collected using one GPUs, if you using 8 gpus 150 each, result in 483,575)
# You can use multiple GPUs to speed up the process but it may results fewer training data.
