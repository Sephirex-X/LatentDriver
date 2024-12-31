## Train the model
Since we used pre-trained BERT for faster coverage, you first need to place the pre-trained BERT model in `checkpoints/` first (you can also use your own by training a BERT-based planner, e.g. planT). You can find our pre-trained BERT model [here](https://huggingface.co/Sephirex-x/LatentDriver/tree/main).

To train latentdriver, run:
```bash
python train.py method=latentdriver \
    ++data_path=${TRAINING_DATA_PATH} \
    ++exp_name=your_exp_name \
    ++version=latentdriver \
    ++method.max_epochs=10 \
    ++method.train_batch_size=2500 \
    ++method.learning_rate=2.0e-4 \
    ++load_num_workers=32 \
    ++method.num_of_decoder=3 \
    ++method.max_len=2 \
    ++method.est_layer=0 \
    ++method.pretrain_enc=checkpoints/pretrained_bert.pth.tar
```

You can train planT using:
```bash
python train.py method=planT \
        ++data_path=${TRAINING_DATA_PATH} \
        ++exp_name=your_exp_name \
        ++load_num_workers=32 \
        ++version=planT
```

## Close-loop evaluation on Waymax
Place the weights under `checkpoints/`, using the following command to run close-loop evaluation:

**For LatentDriver:**
```bash
# reproducing the reusults in Tab.1
python simulate.py method=latentdriver \
        ++waymax_conf.path="${WOMD_VAL_PATH}" \
        ++data_conf.path_to_processed_map_route="${PRE_PROCESS_VAL_PATH}" \
        ++metric_conf.intention_label_path="${INTENTION_VAL_PATH}" \
        ++batch_dims=[7,125] \
        ++method.num_of_decoder=3 \
        ++method.ckpt_path='checkpoints/lantentdriver_t2_J3.ckpt' \
        # other agents are controlled by IDM
        ++ego_control_setting.npc_policy_type=idm

# for ablation study (In ablation study, we set J=4, while in Tab.1 we set J=3)
python simulate.py method=latentdriver \
        ++waymax_conf.path="${WOMD_VAL_PATH}" \
        ++data_conf.path_to_processed_map_route="${PRE_PROCESS_VAL_PATH}" \
        ++metric_conf.intention_label_path="${INTENTION_VAL_PATH}" \
        ++batch_dims=[7,150] \
        ++method.num_of_decoder=4 \
        ++method.ckpt_path='checkpoints/lantentdriver_t2_J4.ckpt'
```
**For planT:**
```bash
python simulate.py method=planT \
        ++waymax_conf.path="${WOMD_VAL_PATH}" \
        ++data_conf.path_to_processed_map_route="${PRE_PROCESS_VAL_PATH}" \
        ++metric_conf.intention_label_path="${INTENTION_VAL_PATH}" \
        ++batch_dims=[7,125] \
        ++ego_control_setting.npc_policy_type=idm \
        ++method.ckpt_path='checkpoints/planT.ckpt'
```

**For Easychauffeur-PPO:**
```bash
python simulate.py method=easychauffeur \
        ++waymax_conf.path="${WOMD_VAL_PATH}" \
        ++data_conf.path_to_processed_map_route="${PRE_PROCESS_VAL_PATH}" \
        ++metric_conf.intention_label_path="${INTENTION_VAL_PATH}" \
        ++batch_dims=[7,125] \
        ++ego_control_setting.npc_policy_type=idm \
        ++method.ckpt_path='checkpoints/easychauffeur_policy_best.pth.tar'
```
**Note:** Set `ego_control_setting.npc_policy_type=idm` to eval under reactive agents, and set `ego_control_setting.npc_policy_type=expert` to eval under non-reavtive agents. You may need to lower eval batch size since simulate IDM agents consumes more memory.