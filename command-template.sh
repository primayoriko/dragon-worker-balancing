#!/bin/bash

# gcloud projects create PROJECT_ID
# gcloud config set project PROJECT_ID
gcloud config set project stei-rpl-13518146

gcloud services enable artifactregistry.googleapis.com container.googleapis.com

# gcloud config set compute/zone COMPUTE_ZONE
# gcloud config set compute/region COMPUTE_REGION
gcloud config set compute/region us-central1

# gcloud container clusters create hello-cluster --num-nodes=1

gcloud beta container --project "stei-rpl-13518146" clusters create "dragon-cluster" --zone "us-central1-a" --no-enable-basic-auth --cluster-version "1.19.16-gke.6100" --release-channel "None" --machine-type "e2-medium" --image-type "COS_CONTAINERD" --disk-type "pd-standard" --disk-size "45" --metadata disable-legacy-endpoints=true --scopes "https://www.googleapis.com/auth/devstorage.read_only","https://www.googleapis.com/auth/logging.write","https://www.googleapis.com/auth/monitoring","https://www.googleapis.com/auth/servicecontrol","https://www.googleapis.com/auth/service.management.readonly","https://www.googleapis.com/auth/trace.append" --max-pods-per-node "110" --num-nodes "1" --logging=SYSTEM,WORKLOAD --monitoring=SYSTEM --enable-ip-alias --network "projects/stei-rpl-13518146/global/networks/default" --subnetwork "projects/stei-rpl-13518146/regions/us-central1/subnetworks/default" --no-enable-intra-node-visibility --default-max-pods-per-node "110" --no-enable-master-authorized-networks --addons HorizontalPodAutoscaling,HttpLoadBalancing,GcePersistentDiskCsiDriver --enable-autoupgrade --enable-autorepair --max-surge-upgrade 0 --max-unavailable-upgrade 1 --enable-shielded-nodes --node-locations "us-central1-a"

# gcloud container clusters get-credentials CLUSTER_NAME
gcloud container clusters get-credentials "dragon-cluster" --zone "us-central1-a"

# docker build -t REGION-docker.pkg.dev/${PROJECT_ID}/hello-repo/hello-app:v2 .
# docker push REGION-docker.pkg.dev/${PROJECT_ID}/hello-repo/hello-app:v2

gcloud auth configure-docker us-central1-docker.pkg.dev

docker build -t us-central1-docker.pkg.dev/stei-rpl-13518146/ta/dragon-scaledown:latest .
docker push us-central1-docker.pkg.dev/stei-rpl-13518146/ta/dragon-scaledown:latest

kubectl create -f crd/v1.yaml

# kubectl create -f dragon/v1.yaml
kubectl create -f dragon/v1-scaledown.yaml
kubectl delete -f dragon/v1-scaledown.yaml

kubectl create -f example/1.yaml
kubectl delete -f example/1.yaml

# curl -X POST \
# -H "Authorization: Bearer "$(gcloud auth application-default print-access-token) \
# -H "Content-Type: application/json; charset=utf-8" \
# -d @permission.json \
# "https://cloudresourcemanager.googleapis.com/v1/projects/stei-rpl-13518146:testIamPermissions"

kubectl describe pods dragon-tf-operator -n kube-system

kubectl get pods --all-namespaces

kubectl get nodes

kubectl describe nodes gke-dragon-cluster-default-pool-61e78eff-2mhh

kubectl logs dragon-tf-operator -n kube-system > logs.txt
