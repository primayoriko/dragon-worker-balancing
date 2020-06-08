#!/bin/bash
kubectl create -f examples/job1.yaml
kubectl create -f examples/job2.yaml
kubectl create -f examples/job3.yaml
kubectl create -f examples/job4.yaml
kubectl create -f examples/job5.yaml
kubectl create -f examples/job6.yaml
kubectl create -f examples/job7.yaml
kubectl create -f examples/job8.yaml
kubectl create -f examples/job9.yaml
kubectl create -f examples/job10.yaml
kubectl create -f examples/job11.yaml
kubectl create -f examples/job12.yaml
kubectl get pods -w
