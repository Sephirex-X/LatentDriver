## Waymo Dataset Preparation
**Step 1:** Download [Waymo Open Motion Dataset v1.1.0](https://waymo.com/open/) (we use the `tf.Example protos` form dataset), and organize the data as follows: 
```bash
waymo_open_dataset_motion_v_1_1_0/
├── uncompressed
│   ├── tf_example
│   │   │──training
│   │   │──validation
│   │   │──testing

```
**Step 2:** Setup your path by environment variables in `scripts/set_env.sh` and run:
```shell
source scripts/set_env.sh
```
**Step 3:** Preprocess the dataset and collect training data (This process will take several hours to complete based on your setting). You need to dump map and route first, together with intention label for metric calculation. After that, you can collect the training data. Using the following script to do so: 
```shell
# dump map and route
sh scripts/preprocess_data.sh
# collect data for training
sh scripts/collecting_training_data.sh
```
Then, the processed data will be saved to `${TRAINING_DATA_PATH}` and organized as follows:
```bash
${TRAINING_DATA_PATH}
├── val_preprocessed_path (2.2G)
├── train_preprocessed_path (25G)
├── val_intention_label (174M)
├── train_intention_label (1.9G)
├── train_data (196G)

```