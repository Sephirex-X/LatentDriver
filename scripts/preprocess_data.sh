# for vaildation
python src/preprocess/preprocess_data.py \
    ++batch_dims=[1,1000] \
    ++waymax_conf.path="${WOMD_VAL_PATH}" \
    ++waymax_conf.drop_remainder=False \
    ++data_conf.path_to_processed_map_route="${PRE_PROCESS_VAL_PATH}" \
    ++metric_conf.intention_label_path="${INTENTION_VAL_PATH}"
# # for training
python src/preprocess/preprocess_data.py \
    ++batch_dims=[1,1000] \
    ++waymax_conf.path="${WOMD_TRAIN_PATH}" \
    ++waymax_conf.drop_remainder=False \
    ++data_conf.path_to_processed_map_route="${PRE_PROCESS_TRAIN_PATH}" \
    ++metric_conf.intention_label_path="${INTENTION_TRAIN_PATH}"
#     tips: batch_dims=[1,2000] means batch size is 2000 and used GPU  is 1, you MUST set GPU=1 here to make 'drop_remainder=False' validated
#      It may take some time to preprocess the data

