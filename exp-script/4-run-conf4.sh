#!/bin/bash

kubectl create -f ../example/experiment/conf-4-job1.yaml
sleep 3
kubectl create -f ../example/experiment/conf-4-job2.yaml
sleep 3
kubectl create -f ../example/experiment/conf-4-job3.yaml
kubectl get pods -w