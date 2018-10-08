#!/bin/bash
echo "PID $$";

SEED=0;
while [[ "$#" > 0 ]]; do case $1 in
  --out_dir) OUT_DIR="$2"; shift;;
  --seed) SEED="$2"; shift;;
  --base_dir) BASE_DIR="$2"; shift;;
  *) echo "Unknown parameter passed: $1"; exit 1;;
esac; shift; done

case BASE_DIR in
    ('enduro') ENV='EnduroNoFrameskip-v4' ;;
    ('pong') ENV='PongNoFrameskip-v4' ;;
    ('catcher') ENV='PLECatcher-v0' ;;
    (*) echo "Unknown base directory $BASE_DIR"; exit 1;;
esac;

mkdir workshop/$BASE_DIR/$OUT_DIR;
echo "Placing results in workshop/$BASE_DIR/$OUT_DIR"



base_command="python train_airl.py --env $ENV --seed $SEED --trajectories_file results/$BASE_DIR/8_trajectories_cache.pkl --n_iter 100 $MODIFICATION";

batched="--ppo_itrs_in_batch 4 " 
state_only="--state_only ";
encoded="--encoder results/$BASE_DIR/encoder.pkl --reward_type mlp ";

echo $base_command;
eval "$base_command > workshop/$BASE_DIR/$OUT_DIR/basic.txt"&
eval "$base_command $state_only > workshop/$BASE_DIR/$OUT_DIR/state_only.txt";
eval "$base_command $encoded > workshop/$BASE_DIR/$OUT_DIR/encoded.txt"&
eval "$base_command $encoded $state_only > workshop/$BASE_DIR/$OUT_DIR/encoded_state_only.txt";

eval "$base_command $batched > workshop/$BASE_DIR/$OUT_DIR/batched.txt"&
eval "$base_command $batched $state_only > workshop/$BASE_DIR/$OUT_DIR/batched_state_only.txt";
eval "$base_command $batched $encoded > workshop/$BASE_DIR/$OUT_DIR/batched_encoded.txt"&
eval "$base_command $batched $state_only $encoded > workshop/$BASE_DIR/$OUT_DIR/batched_encoded_state_only.txt";