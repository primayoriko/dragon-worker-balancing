#!/bin/bash
x=1
while [ $x -le 500 ]
do
  kubectl logs dragon-tf-operator -n kube-system > log.txt
  x=$(( $x + 1 ))
  sleep 30
done