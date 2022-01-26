package main

import (
	tfv1 "github.com/NTHU-LSALAB/DRAGON/pkg/apis/tensorflow/v1"
	"github.com/NTHU-LSALAB/DRAGON/pkg/controller.v1/DRAGON/cluster"
)

type JobQueue []*TrainingJob

type TrainingJob struct {
	*tfv1.TFJob
	// tfv1.TFReplicaType is string
	ReplicasPlacementPlan map[tfv1.TFReplicaType]*JobPlacementPlan
	ReplicaRequest        map[tfv1.TFReplicaType]*cluster.PodRequest
}

type PodRequest struct {
	CpuReq    int64
	MemReq    int64
	GpuReq    int64
	GpuMemReq int64
}

type JobsPlacementPlan map[*TrainingJob]*JobPlacementPlan

// the key is the node name,
// where the value is the node info
type JobPlacementPlan map[string]*NodeResPlacePlan

// The key is worker id & the value is the worker (pods) info
type NodeResPlacePlan map[string]*WorkerResources

type WorkerResources struct {
	// the key is ResourceName => ResourceId,
	// and the value is the amount that used of that resources
	Workers  map[string]string
	Critical bool
}

type NodeResources map[string]*NodeResource

type NodeResource struct {
	CpuTotal int64
	MemTotal int64
	GpuTotal int
	// GpuMemTotal in bytes
	GpuMemTotal int64
	CpuFree     int64
	MemFree     int64
	/* Available GPU calculate */
	// Total GPU count - Pods using nvidia.com/gpu
	GpuFreeCount int
	// GPUs available usage (1.0 - SharePod usage)
	// GPUID to integer index mapping
	GpuFree map[string]*GPUInfo
}
