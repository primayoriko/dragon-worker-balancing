#!/bin/bash

kubectl create -f ../example/test-acc/job1.yaml
sleep 3
kubectl create -f ../example/test-acc/job2.yaml
kubectl get pods -w
