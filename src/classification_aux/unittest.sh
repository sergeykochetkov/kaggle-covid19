#!/bin/bash

./unittest_train.sh

./generate_pseudo_label.sh checkpoints false

if [ $? -eq 0 ]
then
  echo OK, Tests Passed
else
  echo Fail
fi