#!/bin/bash
echo "PID $$";

SEED=0;
while [[ "$#" > 0 ]]; do case $1 in
  --out_dir) OUT_DIR="$2"; shift;;
  --seed) SEED="$2"; shift;;
  --base_dir) BASE_DIR="$2"; shift;;
  *) echo "Unknown parameter passed: $1"; exit 1;;
esac; shift; done

echo "Base directory arg: $BASE_DIR";

case $BASE_DIR in
    ('enduro') ENV='EnduroNoFrameskip-v4' ;;
    ('pong') ENV='PongNoFrameskip-v4' ;;
    ('catcher') ENV='PLECatcher-v0' ;;
    (*) echo "Unknown base directory $BASE_DIR"; exit 1;;
esac;

mkdir workshop/$BASE_DIR/$OUT_DIR;
mkdir workshop/$BASE_DIR/$OUT_DIR/batched_state_only;
mkdir workshop/$BASE_DIR/$OUT_DIR/batched_encoded_state_only;
mkdir workshop/$BASE_DIR/$OUT_DIR/batched_state_action;
mkdir workshop/$BASE_DIR/$OUT_DIR/batched_encoded_state_action;
echo "Placing results in workshop/$BASE_DIR/$OUT_DIR"


for i in 1 2 4 8 16 64; do
    echo $i;
    base_command="pasuspender -- python train_airl.py --env $ENV --seed $SEED --trajectories_file results/$BASE_DIR/8_trajectories_cache.pkl --n_iter 1 --discriminator_itrs 1000 --policy_update_freq 1 --ppo_itrs_in_batch $i --init_location results/$BASE_DIR/expert.pkl --ablation train_discriminator";
    out_file="buffer_size_$i.txt;";
    
    encoded="--encoder results/$BASE_DIR/encoder.pkl --reward_type mlp ";
    
    echo "state only";
    eval "$base_command --state_only &> workshop/$BASE_DIR/$OUT_DIR/batched_state_only/$out_file";
    echo "state only encoded";
    eval "$base_command --state_only $encoded &> workshop/$BASE_DIR/$OUT_DIR/batched_encoded_state_only/$out_file";
    echo "just batched";
    eval "$base_command &> workshop/$BASE_DIR/$OUT_DIR/batched_state_action/$out_file";
    echo "encoded";
    eval "$base_command $encoded &> workshop/$BASE_DIR/$OUT_DIR/batched_encoded_state_action/$out_file";
done
