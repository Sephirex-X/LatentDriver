# set your path here
export WAYMO_DATASET_PATH="/root/waymo_data"
export ROOT_PATH="/root/ltdriver_data"
export WOMD_VAL_PATH="${WAYMO_DATASET_PATH}/waymo_open_dataset_motion_v_1_1_0/uncompressed/tf_example/validation/validation_tfexample.tfrecord@150"
export WOMD_TRAIN_PATH="${WAYMO_DATASET_PATH}/waymo_open_dataset_motion_v_1_1_0/uncompressed/tf_example/training/training_tfexample.tfrecord@1000"

export PRE_PROCESS_VAL_PATH="${ROOT_PATH}/val_preprocessed_path"
export PRE_PROCESS_TRAIN_PATH="${ROOT_PATH}/train_preprocessed_path"

export INTENTION_VAL_PATH="${ROOT_PATH}/val_intention_label"
export INTENTION_TRAIN_PATH="${ROOT_PATH}/train_intention_label"

export TRAINING_DATA_PATH="${ROOT_PATH}/train_data"