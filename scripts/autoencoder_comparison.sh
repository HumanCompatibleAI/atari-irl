#!/bin/bash
echo "PID $$";

SEED=0;
while [[ "$#" > 0 ]]; do case $1 in
  --out_dir) OUT_DIR="$2"; shift;;
  --seed) SEED="$2"; shift;;
  --env) ENVSHORT="$2"; shift;;
  *) echo "Unknown parameter passed: $1"; exit 1;;
esac; shift; done

ENCODER='pixel_class';
case $ENVSHORT in
    ('enduro') ENV='EnduroNoFrameskip-v4' ;;
    ('pong') ENV='PongNoFrameskip-v4'; ENCODER='score_trimmed';;
    ('catcher') ENV='PLECatcher-v0' ;;
    (*) echo "Unknown environment $ENV_SHORT"; exit 1;;
esac;

mkdir workshop/autoencoder/$OUT_DIR;
echo "Placing results in workshop/autoencoder/$OUT_DIR";

eval "python train_ae.py --env $ENV --seed $SEED --encoder_type $ENCODER > workshop/autoencoder/$OUT_DIR/$ENVSHORT\_$ENCODER.txt&";
ENCODER='non_pixel_class'; # score trimming for pong is handled by the train_ae script
eval "python train_ae.py --env $ENV --seed $SEED --encoder_type $ENCODER > workshop/autoencoder/$OUT_DIR/$ENVSHORT\_$ENCODER.txt&";