package main

import (
	"fmt"
	"github.com/NTHU-LSALAB/DRAGON/cmd/DRAGON/app/options"
	tfv1 "github.com/NTHU-LSALAB/DRAGON/pkg/apis/tensorflow/v1"
	tfjobclientset "github.com/NTHU-LSALAB/DRAGON/pkg/client/clientset/versioned"
	"github.com/NTHU-LSALAB/DRAGON/pkg/controller.v1/DRAGON/cluster"
	kubesharev1 "github.com/NTHU-LSALAB/KubeShare/pkg/apis/kubeshare/v1"
	kubeshareclientset "github.com/NTHU-LSALAB/KubeShare/pkg/client/clientset/versioned"
)

var (
	kubeClientSet      kubeclientset.Interface
	tfJobClientSet     tfjobclientset.Interface
	kubeshareClientSet kubeshareclientset.Interface
	lastActionTime     metav1.Time = metav1.Now()
	option             *options.ServerOption
)

func isEnoughResources(job *TrainingJob, node *cluster.NodeResource, isSupportKubeShare bool) bool {
	request := job.ReplicaRequest[tfv1.TFReplicaTypeWorker]

	if node.CpuFree < request.CpuReq || node.MemFree < request.MemReq {
		return false
	}

	if isSupportKubeShare {
		if request.GpuReq > 0 {
			hasFreeGPU := false
			for _, gpu := range node.GpuFree {
				if gpu.GPUFreeReq >= request.GpuReq && gpu.GPUFreeMem >= request.GpuMemReq {
					hasFreeGPU = true
					break
				}
			}
			if !hasFreeGPU && node.GpuFreeCount <= 0 {
				return false
				//if node.GpuFreeCount <= 0 {
				//	return false
				//} else {
				//	node.GpuFreeCount--
				//	freeGPUID = kubesharev1.NewGPUID(5)
				//	node.GpuFree[freeGPUID] = &cluster.GPUInfo{
				//		GPUFreeReq: 1000,
				//		GPUFreeMem: node.GpuMemTotal,
				//	}
				//}
			}
		}
	} else {
		if request.GpuReq > 0 && node.GpuFreeCount < int(request.GpuReq/1000) {
			return false
		}
	}

	fmt.Println("found job with enough resources!")

	return true
}

func getNodeNameOfJobsPSNode(jobs *JobQueue) []string {
	n := len(*jobs)
	PSNames := make([]string, n)

	for i, job := range *jobs {
		for nodeName, _ := range *(job.ReplicasPlacementPlan[tfv1.TFReplicaTypePS]) {
			log.Infof("==== PS of job in index %d, located in Node %s ====", i, nodeName)
			PSNames[i] = nodeName
		}
	}

	return PSNames
}

func ScaleUp(runningQueue JobQueue, constNodeRes cluster.NodeResources) (can bool, scaleUpTarget JobsPlacementPlan) {
	log.Infof("================= ScaleUp Start =================")
	defer log.Infof("================== ScaleUp End ==================")

	nodeRes := constNodeRes.DeepCopy()
	scaleUpTarget = make(JobsPlacementPlan)
	runningJobsNum := len(runningQueue)
	canBeScaledNum := 0
	canBeScaledJobs := make([]bool, runningJobsNum)
	jobsWorkerNum := make([]int32, runningJobsNum)
	PSNodeNames := getNodeNameOfJobsPSNode(&runningQueue)

	for i := 0; i < runningJobsNum; i++ {
		job := runningQueue[i]
		jobsWorkerNum[i] = int32(runningQueue[i].ReplicasPlacementPlan[tfv1.TFReplicaTypeWorker].Count())
		canBeScaledJobs[i] = jobsWorkerNum[i] < *(job.Spec.MaxInstances)

		if canBeScaledJobs[i] {
			canBeScaledNum++
		}
	}

	for canBeScaledNum != 0 {
		selectedJobIdx := -1
		for i := 0; i < runningJobsNum; i++ {
			if canBeScaledJobs[i] && (selectedJobIdx == -1 ||
				jobsWorkerNum[i] < jobsWorkerNum[selectedJobIdx]) {
				selectedJobIdx = i
			}
		}

		job := runningQueue[selectedJobIdx]
		request := job.ReplicaRequest[tfv1.TFReplicaTypeWorker]
		priorityNode := (*nodeRes)[PSNodeNames[selectedJobIdx]]

		var node *cluster.NodeResource = nil
		nodeName := ""

		if isEnoughResources(job, priorityNode, option.KubeShareSupport) {
			node, nodeName = priorityNode, PSNodeNames[selectedJobIdx]
		} else {
			for currNodeName, currNode := range *nodeRes {
				if currNode != priorityNode && isEnoughResources(job, currNode, option.KubeShareSupport) {
					node, nodeName = currNode, currNodeName
					break
				}
			}
		}

		jobsWorkerNum[selectedJobIdx]++
		if node == nil || jobsWorkerNum[selectedJobIdx] >= *(job.Spec.MaxInstances) {
			canBeScaledJobs[selectedJobIdx] = false
			canBeScaledNum--
			continue
		}

		if !option.KubeShareSupport {
			node.CpuFree -= request.CpuReq
			node.MemFree -= request.MemReq
			if request.GpuReq > 0 {
				node.GpuFreeCount -= int(request.GpuReq / 1000)
			}

			if _, ok := scaleUpTarget[job]; !ok {
				scaleUpTarget[job] = job.ReplicasPlacementPlan[tfv1.TFReplicaTypeWorker].DeepCopy()
			}
			if _, ok := (*scaleUpTarget[job])[nodeName]; !ok {
				(*scaleUpTarget[job])[nodeName] = &NodeResPlacePlan{}
			}
			t := &WorkerResources{
				Workers:  map[string]string{},
				Critical: false,
			}
			(*(*scaleUpTarget[job])[nodeName])[NewWorkerID(5)] = t
			if request.GpuReq > 0 {
				(*t).Workers[cluster.ResourceNvidiaGPU] = fmt.Sprintf("%d", (request.GpuReq / 1000))
			}
		} else {
			hasFreeGPU, freeGPUID := false, ""
			if request.GpuReq > 0 {
				for id, gpu := range node.GpuFree {
					if gpu.GPUFreeReq >= request.GpuReq && gpu.GPUFreeMem >= request.GpuMemReq {
						hasFreeGPU, freeGPUID = true, id
						break
					}
				}
				if !hasFreeGPU {
					if node.GpuFreeCount <= 0 {
						stop = true
						break
					} else {
						node.GpuFreeCount--
						freeGPUID = kubesharev1.NewGPUID(5)
						node.GpuFree[freeGPUID] = &cluster.GPUInfo{
							GPUFreeReq: 1000,
							GPUFreeMem: node.GpuMemTotal,
						}
					}
				}
			}

			node.CpuFree -= request.CpuReq
			node.MemFree -= request.MemReq
			if request.GpuReq > 0 {
				node.GpuFree[freeGPUID].GPUFreeReq -= request.GpuReq
				node.GpuFree[freeGPUID].GPUFreeMem -= request.GpuMemReq
			}

			if _, ok := scaleUpTarget[job]; !ok {
				scaleUpTarget[job] = job.ReplicasPlacementPlan[tfv1.TFReplicaTypeWorker].DeepCopy()
			}
			if _, ok := (*scaleUpTarget[job])[nodeName]; !ok {
				(*scaleUpTarget[job])[nodeName] = &NodeResPlacePlan{}
			}
			t := &WorkerResources{
				Workers:  map[string]string{},
				Critical: false,
			}
			(*(*scaleUpTarget[job])[nodeName])[NewWorkerID(5)] = t
			if request.GpuReq > 0 {
				(*t).Workers[cluster.ResourceKubeShareGPU] = freeGPUID
			}
		}
	}

	return
}
