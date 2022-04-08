package main

import (
	"github.com/NTHU-LSALAB/DRAGON/cmd/DRAGON/app/options"
	tfv1 "github.com/NTHU-LSALAB/DRAGON/pkg/apis/tensorflow/v1"
	tfjobclientset "github.com/NTHU-LSALAB/DRAGON/pkg/client/clientset/versioned"
	"github.com/NTHU-LSALAB/DRAGON/pkg/controller.v1/DRAGON/cluster"
	kubeshareclientset "github.com/NTHU-LSALAB/KubeShare/pkg/client/clientset/versioned"
)

var (
	kubeClientSet      kubeclientset.Interface
	tfJobClientSet     tfjobclientset.Interface
	kubeshareClientSet kubeshareclientset.Interface
	lastActionTime     metav1.Time = metav1.Now()
	option             *options.ServerOption
)

// SortNodeFromJob sort node priority from job's placement plan,
// from the least important to the most
func SortNodeFromJob(job *TrainingJob) (sortedNodes []string) {
	/*
	 * current sorting algorithm is:
	 *   Parameter Server node is the most important!
	 */
	sortedNodes = make([]string, 0, len(*job.ReplicasPlacementPlan[tfv1.TFReplicaTypeWorker]))
	isFound := false

	// TODO: optimize this var
	PSNodeName := func() string {
		for n := range *job.ReplicasPlacementPlan[tfv1.TFReplicaTypePS] {
			return n
		}
		return ""
	}()

	for name := range *job.ReplicasPlacementPlan[tfv1.TFReplicaTypeWorker] {
		if name != PSNodeName {
			sortedNodes = append(sortedNodes, name)
		} else {
			isFound = true
		}
	}
	if isFound {
		sortedNodes = append(sortedNodes, PSNodeName)
	}
	return
}

// ScaleDown scale down other jobs let high priority job runs.
// ScaleDown is only called if high priority job exists.
func ScaleDown(highPriorityJob *cluster.PodRequests, runningQueue JobQueue, constNodeRes cluster.NodeResources) (can bool, scaleDownTarget JobsPlacementPlan, highPriorityJobPlacementPlan *[]*JobPlacementPlan) {
	log.Infof("================= ScaleDown Start with %d priority jobs and %d running jobs in queue =================", len(*highPriorityJob), len(runningQueue))
	defer log.Infof("================== ScaleDown End ==================")

	// Don't modify original one
	nodeRes := *constNodeRes.DeepCopy()
	scaleDownTarget = make(JobsPlacementPlan)
	can = false

	runningJobsNum := len(runningQueue)
	canBeScaledNum := 0
	canBeScaledJobs := make([]bool, runningJobsNum)
	jobsWorkerNum := make([]int32, runningJobsNum)
	jobsOrderedNodesName := make([][]string, runningJobsNum)
	//PSNodeNames := getNodeNameOfJobsPSNode(&runningQueue)

	for i := 0; i < runningJobsNum; i++ {
		job := runningQueue[i]
		jobsOrderedNodesName[i] = SortNodeFromJob(job)
		jobsWorkerNum[i] = int32(runningQueue[i].ReplicasPlacementPlan[tfv1.TFReplicaTypeWorker].Count())
		canBeScaledJobs[i] = jobsWorkerNum[i] > *(job.Spec.MinInstances)

		if canBeScaledJobs[i] {
			canBeScaledNum++
		}

		scaleLogStr := "cannot be scaled"
		if canBeScaledJobs[i] {
			scaleLogStr = "can be scaled"
		}
		log.Infof("======== Job [%d]: curr worker %d, %s =======", i, jobsWorkerNum[i], scaleLogStr)
	}

	log.Infof("======== ScaleDown initially has %d jobs can be scaled up =======", canBeScaledNum)

	for canBeScaledNum != 0 {
		ok, tmp := ScheduleJob(&([]*cluster.PodRequests{highPriorityJob}), nodeRes)
		if ok[0] == len(*highPriorityJob) {
			log.Infof("========= Scale Down successful! =========")

			highPriorityJobPlacementPlan = tmp
			can = true

			break
		}

		selectedJobIdx := -1
		for i := 0; i < runningJobsNum; i++ {
			if canBeScaledJobs[i] && (selectedJobIdx == -1 ||
				jobsWorkerNum[i] > jobsWorkerNum[selectedJobIdx]) {
				selectedJobIdx = i
			}
		}

		log.Infof("======== Job [%d] selected =======", selectedJobIdx)

		job := runningQueue[selectedJobIdx]
		jobReq := job.ReplicaRequest[tfv1.TFReplicaTypeWorker]
		isSuccess := false

		// better to store `SortNodeFromJob(job)` in array? e.g => sortedNode[job]
		for _, nodeName := range jobsOrderedNodesName[selectedJobIdx] {
			log.Infof("======== Node [%s] for job [%d] selected =======", nodeName, selectedJobIdx)

			stopFlg := false
			plan := (*job.ReplicasPlacementPlan[tfv1.TFReplicaTypeWorker])[nodeName]

			for workerID, worker := range *plan {
				log.Infof("======== Worker [%s] in node [%s] for job [%d] selected =======", workerID, nodeName, selectedJobIdx)

				if worker.Critical {
					log.Infof("======== Worker [%s] in node [%s] for job [%d] is CRITICAL!! =======", workerID, nodeName, selectedJobIdx)
					continue
				}

				res := nodeRes[nodeName]
				res.CpuFree += jobReq.CpuReq
				res.MemFree += jobReq.MemReq

				if option.KubeShareSupport { // kubeshare/gpu
					if gpuid, ok := (*worker).Workers[cluster.ResourceKubeShareGPU]; ok {
						res.GpuFree[gpuid].GPUFreeReq += jobReq.GpuReq
						res.GpuFree[gpuid].GPUFreeMem += jobReq.GpuMemReq
					}
				} else { // nvidia.com/gpu
					if _, ok := (*worker).Workers[cluster.ResourceNvidiaGPU]; ok {
						res.GpuFreeCount += int(jobReq.GpuReq / 1000)
					}
				}

				// log.Infof("************************************ DEBUG ************************************")
				// nodeRes.PrintMe()
				// log.Infof("************************************ DEBUG ************************************")

				// make a temporary copy. apply to origin only if can scale down
				if _, ok := scaleDownTarget[job]; !ok {
					scaleDownTarget[job] = job.ReplicasPlacementPlan[tfv1.TFReplicaTypeWorker].DeepCopy()
				}

				delete(*(*scaleDownTarget[job])[nodeName], workerID)

				log.Infof("======== Worker [%s] in node [%s] for job [%d] successfully selected to terminated =======", workerID, nodeName, selectedJobIdx)

				stopFlg = true
				break
			}

			if stopFlg {
				isSuccess = true
				break
			}
		}

		jobsWorkerNum[selectedJobIdx]--
		if !isSuccess || jobsWorkerNum[selectedJobIdx] <= *(job.Spec.MinInstances) {
			if !isSuccess {
				log.Infof("======== Job [%d] unsuccessful to scaled down =======", selectedJobIdx)
			} else {
				log.Infof("======== Worker for job [%d] already reached minimum limit =======", selectedJobIdx)
			}

			canBeScaledJobs[selectedJobIdx] = false
			canBeScaledNum--
		}

		//for currNodeName, currNode := range *(job.ReplicasPlacementPlan[tfv1.TFReplicaTypeWorker]) {
		//	if currNodeName != priorityNodeName {
		//		nodeName, node = currNodeName, currNode
		//	}
		//}
		//
		//if node == nil {
		//	nodeName, node = priorityNodeName, (*job.ReplicasPlacementPlan[tfv1.TFReplicaTypeWorker])[priorityNodeName]
		//}
	}

	if !can {
		ok, tmp := ScheduleJob(&([]*cluster.PodRequests{highPriorityJob}), nodeRes)
		if ok[0] == len(*highPriorityJob) {
			log.Infof("========= Scale Down successful! =========")

			highPriorityJobPlacementPlan = tmp
			can = true
		}
	}

	for i, job := range runningQueue {
		if _, ok := scaleDownTarget[job]; ok {
			for name, node := range *scaleDownTarget[job] {
				log.Infof("Job [%d] in node [%s] has %d worker", i, name, len(*node))
			}
		}
	}

	return
}

func ScheduleJob(requestsGroups *[]*cluster.PodRequests, constNodeRes cluster.NodeResources) (okNum []int, placementPlansPtr *[]*JobPlacementPlan) {
	return
}
