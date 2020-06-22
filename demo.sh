#!/bin/bash
kubectl create -f examples/job1.yaml
kubectl create -f examples/job2.yaml
kubectl create -f examples/job3.yaml
kubectl create -f examples/job4.yaml
kubectl get pods -w