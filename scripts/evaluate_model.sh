#!/usr/bin/env bash
#
# Copyright (c) 2020 MaybeShewill-CV, Inc. All Rights Reserved
#
# Author: MaybeShewill-CV
# Date: 2021/04/30
#
# evaluate model

# ------------ 设置常量 ----------------
parameters=4
input_para_nums=$#
para1=$1
para2=$2
para3=$3
para4=$4

# ------------ 帮助函数 ----------------
function usage() {
  echo "examples: "
  echo "        CUDA_VISIBLE_DEVICES="" python3.5 tools/evaluate_model.py --batch_size 1 --net resnet
  --dataset ilsvrc_2012 --weights_path ./model/resnet_ilsvrc_2012/resnet_val_acc\=0.4631.ckpt-88 ~/image-classification-tensorflow"
  exit 1
}

# ------------ 主函数 ------------------
function main() {
if [ ${input_para_nums} != ${parameters} ];
then
  usage
else
  cd "${para4}" || exit 1
  export PYTHONPATH="${para4}":$PYTHONPATH
  CUDA_VISIBLE_DEVICES="" python3.5 ./tools/evaluate_model.py --batch_size 1 --net "${para1}" --dataset "${para2}" --weights_path "${para3}"
fi
}

main