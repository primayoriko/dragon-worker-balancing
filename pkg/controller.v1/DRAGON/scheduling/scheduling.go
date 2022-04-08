package scheduling

import (
	"bytes"
	"fmt"
	"math"
	"math/rand"
	"strconv"
	"sync"
	"time"
	"unsafe"

	"github.com/NTHU-LSALAB/DRAGON/cmd/DRAGON/app/options"
	common "github.com/NTHU-LSALAB/DRAGON/pkg/apis/common/v1"
	tfv1 "github.com/NTHU-LSALAB/DRAGON/pkg/apis/tensorflow/v1"
	tfjobclientset "github.com/NTHU-LSALAB/DRAGON/pkg/client/clientset/versioned"
	"github.com/NTHU-LSALAB/DRAGON/pkg/controller.v1/DRAGON/cluster"
	kubesharev1 "github.com/NTHU-LSALAB/KubeShare/pkg/apis/kubeshare/v1"
	kubeshareclientset "github.com/NTHU-LSALAB/KubeShare/pkg/client/clientset/versioned"
	log "github.com/sirupsen/logrus"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	kubeclientset "k8s.io/client-go/kubernetes"
)

var (
	kubeClientSet      kubeclientset.Interface
	tfJobClientSet     tfjobclientset.Interface
	kubeshareClientSet kubeshareclientset.Interface
	lastActionTime     metav1.Time = metav1.Now()
	option             *options.ServerOption
	//BaseTime     	   metav1.Time = metav1.Now()
)

func InitClientSets(itk kubeclientset.Interface, itt tfjobclientset.Interface, itm kubeshareclientset.Interface, op *options.ServerOption) {
	kubeClientSet, tfJobClientSet, kubeshareClientSet, option = itk, itt, itm, op
}

/* ------------------- struct JobQueue start ------------------- */

type JobQueue []*TrainingJob

func (this *JobQueue) PrintMe(whoami string) {
	log.Infof("============ %s ============", whoami)
	if this != nil {
		for i, _ := range *this {
			//log.Infof("%d: %s/%s", i, j.Namespace, j.Name)
			log.Infof("%d", i)
		}
	}
	log.Infof("====================================")
}

func (this *JobQueue) Add(job *TrainingJob) {
	*this = append(*this, job)
}

func (this *JobQueue) Remove(job *TrainingJob) error {
	ns, name := job.ObjectMeta.Namespace, job.ObjectMeta.Name
	for i, j := range *this {
		if j.ObjectMeta.Namespace == ns && j.ObjectMeta.Name == name {
			*this = append((*this)[:i], (*this)[i+1:]...)
			return nil
		}
	}
	return fmt.Errorf("Error when removing job: %s/%s from queue, the job is not in queue", job.ObjectMeta.Namespace, job.ObjectMeta.Name)
}

/* ------------------- struct JobQueue end ------------------- */

/* ------------------- struct TrainingJob start ------------------- */

// TODO: add dominant value field
type TrainingJob struct {
	*tfv1.TFJob
	ReplicasPlacementPlan map[tfv1.TFReplicaType]*JobPlacementPlan
	ReplicaRequest        map[tfv1.TFReplicaType]*cluster.PodRequest
	//Namespace string
	//Name string
}

// TODO: add dominant value initialization
func NewTrainingJob(tfjob *tfv1.TFJob) *TrainingJob {
	replicaReq := make(map[tfv1.TFReplicaType]*cluster.PodRequest)
	for replica, replicaSpec := range tfjob.Spec.TFReplicaSpecs {
		replicaReq[replica] = GetPodRequestsFromTFJobReplica(replicaSpec)
	}
	newJob := &TrainingJob{
		TFJob:                 tfjob.DeepCopy(),
		ReplicasPlacementPlan: make(map[tfv1.TFReplicaType]*JobPlacementPlan),
		ReplicaRequest:        replicaReq,
	}
	return newJob
}

func (this *TrainingJob) UpdateTFJobTime() error {
	// if option.KubeShareSupport {
	// 	panic("FFFFFFFFFFUUUUUUUUUUUUUUCCCCCCCCCCCCCCCKKKKKKKKKKKKKKKKKK")
	// }
	log.Infof("UpdateTFJobTime: updating tfjob time status")
	oldJob, err := tfJobClientSet.KubeflowV1().TFJobs(this.Namespace).Get(this.Name, metav1.GetOptions{})
	if err != nil {
		log.Errorf("UpdateTFJobTime when get old TFJob error: %s", err.Error())
		return err
	}
	newJob := oldJob.DeepCopy()
	newJob.Status.EnqueueTime = this.Status.EnqueueTime.DeepCopy()
	newJob.Status.StartRunTime = this.Status.StartRunTime.DeepCopy()
	newJob.Status.FinishedTime = this.Status.FinishedTime.DeepCopy()
	updatedJob, err := tfJobClientSet.KubeflowV1().TFJobs(this.Namespace).UpdateStatus(newJob)
	if err != nil {
		log.Errorf("UpdateTFJobTime when update TFJob error: %s", err.Error())
		return err
	}
	this.TFJob = updatedJob
	return nil
}

func (this *TrainingJob) GetPodRequests(rt tfv1.TFReplicaType) *cluster.PodRequests {
	requests := make(cluster.PodRequests, 0)
	for name, replica := range this.Spec.TFReplicaSpecs {
		if name != rt {
			continue
		}
		for i := int32(0); i < *replica.Replicas; i++ {
			requests = append(requests, this.ReplicaRequest[name])
		}
	}
	return &requests
}

// minta res sebanyak minimum worker , jadi array yg tiap2 nya resource worker (pod)
func (this *TrainingJob) GetMinInstanceWorkerPodRequests() *cluster.PodRequests {
	requests := make(cluster.PodRequests, 0)
	for name := range this.Spec.TFReplicaSpecs {
		if name != tfv1.TFReplicaTypeWorker {
			continue
		}
		for i := int32(0); i < *this.Spec.MinInstances; i++ {
			requests = append(requests, this.ReplicaRequest[name])
		}
	}
	return &requests
}

/* ------------------- struct TrainingJob end ------------------- */

/* ------------------- struct JobsPlacementPlan start ------------------- */

// Job NS/Name => Job's Placement Plan
type JobsPlacementPlan map[*TrainingJob]*JobPlacementPlan

func (this *JobsPlacementPlan) DeepCopy() *JobsPlacementPlan {
	js := make(JobsPlacementPlan, len(*this))
	for jskey, jsval := range *this {
		js[jskey] = jsval.DeepCopy()
	}
	return &js
}

func (this *JobsPlacementPlan) PrintMe() {
	log.Infof("============ Jobs Placement Plan ============")
	if this != nil {
		for job, jobPlacementPlan := range *this {
			log.Infof("Job: %s/%s", job.Namespace, job.Name)
			jobPlacementPlan.PrintMe()
		}
	}
	log.Infof("=============================================")
}

/* ------------------- struct JobsPlacementPlan end ------------------- */

/* ------------------- struct JobPlacementPlan start ------------------- */

// NodeName => Node Placement Resource
type JobPlacementPlan map[string]*NodeResPlacePlan

func (this *JobPlacementPlan) DeepCopy() *JobPlacementPlan {
	j := make(JobPlacementPlan, len(*this))
	for jkey, jval := range *this {
		j[jkey] = jval.DeepCopy()
	}
	return &j
}

func (this *JobPlacementPlan) PrintMe() {
	if this != nil {
		for nodeName, placementPlan := range *this {
			log.Infof("  %s:", nodeName)
			placementPlan.PrintMe("    ")
		}
	}
}

func (this *JobPlacementPlan) Count() (sum int) {
	sum = 0
	if this != nil {
		for _, workers := range *this {
			sum += len(*workers)
		}
	}
	return
}

// GetAllWorkersID returns all workers ID storing in Pod's Label
// which combine with format NodeName-[devices id-device id...]
// device
/*func (this *JobPlacementPlan) GetAllWorkers() *NodeResPlacePlan {
	p := make(NodeResPlacePlan)
	for _, noderes := range *this {
		for workerid, worker := range *noderes {
			p[workerid] = worker
		}
	}
	return &p
}*/

/* ------------------- struct JobPlacementPlan end ------------------- */

/* ------------------- struct NodeResPlacePlan start ------------------- */

// Worker ID => Worker Resources
type NodeResPlacePlan map[string]*WorkerResources

func (this *NodeResPlacePlan) DeepCopy() *NodeResPlacePlan {
	n := make(NodeResPlacePlan, len(*this))
	for key, val := range *this {
		n[key] = val.DeepCopy()
	}
	return &n
}

func (this *NodeResPlacePlan) PrintMe(prefix string) {
	if this != nil {
		var buf bytes.Buffer
		for n, nval := range *this {
			buf.Reset()
			for t, id := range nval.Workers {
				buf.WriteString(t)
				buf.WriteString(":")
				buf.WriteString(id)
				buf.WriteString(" ")
			}
			log.Infof("%s%s: %s", prefix, n, buf.String())
		}
	}
}

/* ------------------- struct NodeResPlacePlan end ------------------- */

/* ------------------- struct WorkerResources start ------------------- */

type WorkerResources struct {
	// ResourceName => ResourceId
	Workers  map[string]string
	Critical bool
}

func (this *WorkerResources) DeepCopy() *WorkerResources {
	w := WorkerResources{
		Workers:  make(map[string]string, len(this.Workers)),
		Critical: this.Critical,
	}
	for key, val := range this.Workers {
		w.Workers[key] = val
	}
	return &w
}

/* ------------------- struct WorkerResources end ------------------- */

func GetPodRequestsFromTFJobReplica(replica *common.ReplicaSpec) *cluster.PodRequest {
	return GetPodRequestsFromPodTemplate(&replica.Template)
}

func GetPodRequestsFromPodTemplate(template *corev1.PodTemplateSpec) *cluster.PodRequest {
	tmp := cluster.PodRequest{
		CpuReq:    0,
		MemReq:    0,
		GpuReq:    0,
		GpuMemReq: 0,
	}

	for _, container := range template.Spec.Containers {
		tmp.CpuReq += container.Resources.Requests.Cpu().MilliValue()
		tmp.MemReq += container.Resources.Requests.Memory().MilliValue()
	}

	if option.KubeShareSupport {
		if gpureq, gpureqok := template.ObjectMeta.Annotations[kubesharev1.KubeShareResourceGPURequest]; gpureqok && gpureq != "" {
			gpureqf, err := strconv.ParseFloat(gpureq, 64)
			if err != nil {
				log.Errorf("Cannot parse nvidia gpu request, pod: %s/%s, gpu req: %s", template.Namespace, template.Name, template.ObjectMeta.Annotations[kubesharev1.KubeShareResourceGPURequest])
				return nil
			}
			gpureqi := int64(math.Ceil(gpureqf * (float64)(1000.0)))
			tmp.GpuReq += gpureqi
		}
		if gpumem, gpumemok := template.ObjectMeta.Annotations[kubesharev1.KubeShareResourceGPUMemory]; gpumemok && gpumem != "" {
			gpumemi, err := strconv.ParseInt(gpumem, 10, 64)
			if err != nil {
				log.Errorf("Cannot parse nvidia gpu memory, pod: %s/%s, gpu req: %s", template.Namespace, template.Name, template.ObjectMeta.Annotations[kubesharev1.KubeShareResourceGPUMemory])
				return nil
			}
			tmp.GpuMemReq += gpumemi
		}
	} else {
		for _, container := range template.Spec.Containers {
			var gpuNum resource.Quantity
			gpuNum.Add(container.Resources.Limits[kubesharev1.ResourceNVIDIAGPU])
			tmp.GpuReq += gpuNum.MilliValue()
		}
	}
	return &tmp
}

func SchedulingAlgorithm(
	waitingQueue *JobQueue,
	runningQueue *JobQueue,
	highPrioritySharePodsQueue *[]*kubesharev1.SharePod,
	highPrioritySharePodsQueueMutex *sync.Mutex,
	nodeRes cluster.NodeResources,
) {
	//log.Errorf("================ Scheduling Algo Enter ===================")
	//defer log.Errorf("================ Scheduling Algo Exit ===================")

	// check if high priority job exists
	var pendingResource *cluster.PodRequest = nil
	var pendingSharePod *kubesharev1.SharePod

	highPrioritySharePodsQueueMutex.Lock()
	for _, pod := range *highPrioritySharePodsQueue {
		if val, ok := pod.ObjectMeta.Annotations["lsalab.nthu/priority"]; ok && val == "high" && pod.Spec.NodeName == "" {
			pendingResource = GetPodRequestsFromPodTemplate(&corev1.PodTemplateSpec{
				ObjectMeta: pod.ObjectMeta,
				Spec:       pod.Spec,
			})
			pendingSharePod = pod
			log.Infof("Found a SharePod need to be scheduled: %s/%s", pod.ObjectMeta.Namespace, pod.ObjectMeta.Name)
			break
		}
	}
	highPrioritySharePodsQueueMutex.Unlock()

	/*
	 * Scheduling Phase 1
	 * Determine if there is a high priority job:
	 * 1. Other pending Pod
	 * 2. A job in waiting queue is waiting over 60 seconds (avoid starvation)
	 * other jobs must waiting until cluster have free resource for high
	 * priority job to be scheduled.
	 *
	 * If there is high priority job, try to schedule it through scale down.
	 * ScaleDown is only called if high priority job exists.
	 */
	//log.Errorf("================ Scheduling Algo P1 Enter ===================")

	var highPriorityJob *cluster.PodRequests = nil
	// var highPriorityTrainingJob *TrainingJob = nil

	// High priority job first
	if pendingResource != nil {
		highPriorityJob = &cluster.PodRequests{pendingResource}
	} else if now := metav1.Now(); len(*waitingQueue) > 0 {
		// Job that waiting over 1 min first
		// jobs in waitingQueue, the older the more front
		waitingTime := now.Sub((*waitingQueue)[0].Status.EnqueueTime.Time).Seconds()
		if waitingTime >= 30.0 {
			log.Infof("************** PRIME: there is job waiting more than threshold time [%f]", waitingTime)
			// TODO: need to find out, is this only by worker pod or all pod (including PS pod)
			highPriorityJob = (*waitingQueue)[0].GetMinInstanceWorkerPodRequests()
			// highPriorityTrainingJob = (*waitingQueue)[0]
		}
	}

	var scaleDownFlag bool = false

	if highPriorityJob != nil {
		log.Infof("************** PRIME: high priority job found, try to scaledown")
		ok, scaleDownPlan, _ := ScaleDown(highPriorityJob, *runningQueue, nodeRes)
		if ok {
			log.Infof("************** PRIME: scaledown success")
			scaleDownFlag = true
			for job, plan := range scaleDownPlan {
				job.ReplicasPlacementPlan[tfv1.TFReplicaTypeWorker] = plan
			}
		} else {
			log.Infof("************** PRIME: scaledown failed")
		}
		lastActionTime = metav1.Now()
	}
	//log.Errorf("================ Scheduling Algo P2 Enter ===================")

	/*
	 * Scheduling Phase 2
	 * If no high priority job, or there is scale down that going to be performed,
	 * select a job can be scheduled from waiting queue.
	 */
	if highPriorityJob == nil || scaleDownFlag {
		if pendingResource != nil {
			log.Infof("************** PRIME: there is job waiting more than threshold time [%f]", waitingTime)
			ok, placementPlans := ScheduleJob(
				&([]*cluster.PodRequests{
					highPriorityJob,
				}),
				nodeRes,
			)
			// If high priority job can be scheduled, schedule it here... LOL
			if ok[0] >= 1 {
				var nodeName string
				var worker *WorkerResources
				for n, p := range *(*placementPlans)[0] {
					nodeName = n
					for _, w := range *p {
						worker = w
						break
					}
					break
				}
				latestSharePod, err := kubeshareClientSet.KubeshareV1().SharePods(pendingSharePod.Namespace).Get(pendingSharePod.Name, metav1.GetOptions{})
				if err == nil && latestSharePod.ObjectMeta.UID == pendingSharePod.ObjectMeta.UID {
					latestSharePod.Spec.NodeName = nodeName
					if (*highPriorityJob)[0].GpuReq > 0 {
						if latestSharePod.Annotations == nil {
							latestSharePod.Annotations = map[string]string{}
						}
						latestSharePod.Annotations[kubesharev1.KubeShareResourceGPUID] = (*worker).Workers[cluster.ResourceKubeShareGPU]
					}
					_, errr := kubeshareClientSet.KubeshareV1().SharePods(latestSharePod.Namespace).Update(latestSharePod)
					if errr != nil {
						log.Errorf("Error when update SharePod: %s", errr)
					} else {
						/* SharePod Schedule successfully */
						lastActionTime = metav1.Now()
						// delete SharePod from queue
						highPrioritySharePodsQueueMutex.Lock()
						for i, p := range *highPrioritySharePodsQueue {
							if p.ObjectMeta.UID == pendingSharePod.ObjectMeta.UID {
								*highPrioritySharePodsQueue = append((*highPrioritySharePodsQueue)[:i], (*highPrioritySharePodsQueue)[i+1:]...)
								break
							}
						}
						highPrioritySharePodsQueueMutex.Unlock()
					}
				}
				//else {
				//	log.Errorf("Error when schedule SharePod %s/%s, err: %s", pendingSharePod.ObjectMeta.Namespace, pendingSharePod.ObjectMeta.Name, err)
				//}
			}
			//else {
			//	log.Infof("No resource for SharePod %s/%s", pendingSharePod.Namespace, pendingSharePod.Name)
			//}
		} else {
			i := 0
			for _, job := range *waitingQueue {
				log.Infof("************** PRIME: schedule job [%d] in waiting queue", i)
				ok, placementPlans := ScheduleJob(
					&([]*cluster.PodRequests{
						job.GetPodRequests(tfv1.TFReplicaTypePS),
						job.GetPodRequests(tfv1.TFReplicaTypeWorker),
					}),
					nodeRes,
				)

				log.Infof("************** ERICYEH: OK NUM: %d", ok)
				req := [2]int {
					int(*job.Spec.TFReplicaSpecs[tfv1.TFReplicaTypePS].Replicas),
					int(*job.Spec.MinInstances),
				}
				log.Infof("************** PRIMA: needed ok: %d", req)
				log.Infof("************** ERICYEH: gpu: %d", job.ReplicaRequest[tfv1.TFReplicaTypeWorker].GpuReq)
				if ok[0] >= int(*job.Spec.TFReplicaSpecs[tfv1.TFReplicaTypePS].Replicas) && ok[1] >= int(*job.Spec.MinInstances) {
					log.Infof("************** PRIME: job [%d] in waiting queue can be scheduled", i)

					job.ReplicasPlacementPlan[tfv1.TFReplicaTypePS] = (*placementPlans)[0]
					for _, plan := range *job.ReplicasPlacementPlan[tfv1.TFReplicaTypePS] {
						for _, worker := range *plan {
							worker.Critical = true
						}
					}

					job.ReplicasPlacementPlan[tfv1.TFReplicaTypeWorker] = (*placementPlans)[1]
					// TODO: uncomment if want to make 1st worker critical
					//flg := false
					//for _, plan := range *job.ReplicasPlacementPlan[tfv1.TFReplicaTypeWorker] {
					//	for _, worker := range *plan {
					//		worker.Critical = true
					//		flg = true
					//		break
					//	}
					//	if flg {
					//		break
					//	}
					//}

					waitingQueue.Remove(job)
					runningQueue.Add(job)
					now := metav1.Now()
					job.Status.StartRunTime = &now

					lastActionTime = metav1.Now()
					break
				}
				log.Infof("************** PRIME: job [%d] can't be scheduled", i)
				i += 1
			}
		}
	}

	//log.Errorf("================ Scheduling Algo P3 Enter ===================")

	/*
	 * Scheduling Phase 3
	 * If no any action over 60 secs, try to scale up workers of jobs in
	 * running queue.
	 */
	//log.Infof("===== check ScaleUp at timestamp [%f] =============",
	//	metav1.Now().Sub(BaseTime.Time).Seconds())
	//log.Infof("===== check ScaleUp at timestamp lastActionTime [%f] =============",
	//	metav1.Now().Sub(lastActionTime.Time).Seconds())
	if now := metav1.Now(); now.Sub(lastActionTime.Time).Seconds() >= 60.0 {
		ok, placementPlan := ScaleUp(*runningQueue, nodeRes)
		if ok {
			for job, plan := range placementPlan {
				job.ReplicasPlacementPlan[tfv1.TFReplicaTypeWorker] = plan
			}
		}
		lastActionTime = metav1.Now()
	}
}

// TODO: Add process to update dominant value
// ScheduleJob returns:
// * okNum: the max number of worker can be scheduled,
// * placementPlan: placement plan of workers,
// * PSPlace: nodeName of parameter server.
func ScheduleJob(requestsGroups *[]*cluster.PodRequests, constNodeRes cluster.NodeResources) (okNum []int, placementPlansPtr *[]*JobPlacementPlan) {
	log.Infof("================= ScheduleJob Start =================")
	defer log.Infof("================== ScheduleJob End ==================")

	groupNum := len(*requestsGroups)
	placementPlans := make([]*JobPlacementPlan, groupNum)
	for k := range placementPlans {
		placementPlans[k] = &JobPlacementPlan{}
	}
	placementPlansPtr = &placementPlans
	okNum = make([]int, groupNum)
	for i := range okNum {
		okNum[i] = 0
	}

	nodeRes := *constNodeRes.DeepCopy()

	maxSlot, maxSlotNode := 0, ""

	// Test one node can contain all requests
	for nodeName, node := range nodeRes {
		tmps := make([]*NodeResPlacePlan, groupNum)
		for k := range tmps {
			tmps[k] = &NodeResPlacePlan{}
		}

		stop := false
		oneNodeOk := true
		for groupIdx, requests := range *requestsGroups {
			for _, request := range *requests {
				if node.CpuFree < request.CpuReq || node.MemFree < request.MemReq {
					oneNodeOk = false
					stop = true
					log.Infof("Break in cpu or mem request")
					break
				}

				if option.KubeShareSupport { // kubeshare/gpu
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
								oneNodeOk = false
								stop = true
								log.Infof("Break in gpu request 1")
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

					t := &WorkerResources{
						Workers:  map[string]string{},
						Critical: false,
					}
					(*tmps[groupIdx])[NewWorkerID(5)] = t
					if request.GpuReq > 0 {
						(*t).Workers[cluster.ResourceKubeShareGPU] = freeGPUID
					}
				} else { // nvidia.com/gpu
					if request.GpuReq > 0 {
						if node.GpuFreeCount < int(request.GpuReq/1000) {
							oneNodeOk = false
							stop = true
							log.Infof("Break in nvidia.com/gpu request")
							break
						}
					}

					node.CpuFree -= request.CpuReq
					node.MemFree -= request.MemReq
					if request.GpuReq > 0 {
						node.GpuFreeCount -= int(request.GpuReq / 1000)
					}

					t := &WorkerResources{
						Workers:  map[string]string{},
						Critical: false,
					}
					(*tmps[groupIdx])[NewWorkerID(5)] = t
					if request.GpuReq > 0 {
						(*t).Workers[cluster.ResourceNvidiaGPU] = fmt.Sprintf("%d", (request.GpuReq / 1000))
					}
					// log.Infof("*****************ERICYEH 1*********************: %d, %v", request.GpuReq, t.Workers)
				}
			}
			if stop {
				break
			}
		}

		if oneNodeOk {
			for i, val := range tmps {
				okNum[i] = len(*val)
			}
			for groupIdx, val := range tmps {
				(*placementPlans[groupIdx])[nodeName] = val
			}
			log.Infof("There is one node %s can contain all requests, ok num: %d", nodeName, okNum)
			return
		}

		tmpNum := func() (max int) {
			tmp := make([]int, groupNum)
			for groupIdx, val := range tmps {
				tmp[groupIdx] += len(*val)
			}
			max = 0
			for _, val := range tmp {
				if val > max {
					max = val
				}
			}
			return
		}()
		if tmpNum > maxSlot {
			maxSlot, maxSlotNode = tmpNum, nodeName
		}
	}

	if maxSlot == 0 {
		return
	}

	// worker cross node
	nodeRes = *constNodeRes.DeepCopy()
	sortedNodes := SortNodeFromNodeRes(nodeRes, maxSlotNode)
	groupIdx := 0
	requestsIdx := 0

	for _, nodeName := range sortedNodes {
		node := nodeRes[nodeName]

		tmps := make([]*NodeResPlacePlan, groupNum)
		for k := range tmps {
			tmps[k] = &NodeResPlacePlan{}
		}

		stop := false
		for ; groupIdx < len(*requestsGroups); groupIdx++ {
			requests := (*requestsGroups)[groupIdx]

			for ; requestsIdx < len(*requests); requestsIdx++ {
				request := (*requests)[requestsIdx]

				if node.CpuFree < request.CpuReq || node.MemFree < request.MemReq {
					stop = true
					break
				}

				if option.KubeShareSupport { // kubeshare/gpu
					hasFreeGPU, freeGPUID := false, ""
					if request.GpuReq > 0 {
						for id, gpu := range node.GpuFree {
							if gpu.GPUFreeReq >= request.GpuReq && gpu.GPUFreeMem >= request.GpuMemReq {
								hasFreeGPU, freeGPUID = true, id
								log.Infof("Break in cpu or mem request 2")
								break
							}
						}
						if !hasFreeGPU {
							if node.GpuFreeCount <= 0 {
								stop = true
								log.Infof("Break in gpu request")
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

					t := &WorkerResources{
						Workers:  map[string]string{},
						Critical: false,
					}
					(*tmps[groupIdx])[NewWorkerID(5)] = t
					if request.GpuReq > 0 {
						(*t).Workers[cluster.ResourceKubeShareGPU] = freeGPUID
					}
				} else { // nvidia.com/gpu
					if request.GpuReq > 0 {
						if node.GpuFreeCount < int(request.GpuReq/1000) {
							stop = true
							log.Infof("Break in nvidia.com/gpu request")
							break
						}
					}

					node.CpuFree -= request.CpuReq
					node.MemFree -= request.MemReq
					if request.GpuReq > 0 {
						node.GpuFreeCount -= int(request.GpuReq / 1000)
					}

					t := &WorkerResources{
						Workers:  map[string]string{},
						Critical: false,
					}
					(*tmps[groupIdx])[NewWorkerID(5)] = t
					if request.GpuReq > 0 {
						(*t).Workers[cluster.ResourceNvidiaGPU] = fmt.Sprintf("%d", (request.GpuReq / 1000))
					}
				}
				okNum[groupIdx]++
			}
			if stop {
				break
			}
			requestsIdx = 0
		}
		for groupIdx, tmp := range tmps {
			if len(*tmp) > 0 {
				(*placementPlans[groupIdx])[nodeName] = tmp
			}
		}
	}

	return
}

// O( NM + MDR + NMDR )
// O( NM + NMDR )
// O( NMDR )
// TODO: update as in the report
// ScaleDown scale down other jobs let high priority job runs.
// ScaleDown is only called if high priority job exists.
//func ScaleDown(highPriorityJob *cluster.PodRequests, runningQueue JobQueue, constNodeRes cluster.NodeResources) (can bool, scaleDownTarget JobsPlacementPlan, highPriorityJobPlacementPlan *[]*JobPlacementPlan) {
//	log.Infof("================= ScaleDown Start =================")
//	defer log.Infof("================== ScaleDown End ==================")
//
//	// Don't modify original one
//	nodeRes := *constNodeRes.DeepCopy()
//	scaleDownTarget = make(JobsPlacementPlan)
//	can = false
//
//	// Run over running jobs to free resources
//	for _, runJob := range runningQueue {
//		i := int32(0)
//		maxDeleteCount := int32(runJob.ReplicasPlacementPlan[tfv1.TFReplicaTypeWorker].Count()) - *(runJob.Spec.MinInstances)
//		runJobReq := runJob.ReplicaRequest[tfv1.TFReplicaTypeWorker]
//		if maxDeleteCount < 0 {
//			log.Errorf("WHY running worker - min instances < 0 ???")
//		}
//		stop := false
//		// run over each worker but can't delete over max delete count
//		for _, nodeName := range SortNodeFromJob(runJob) {
//			plan := (*runJob.ReplicasPlacementPlan[tfv1.TFReplicaTypeWorker])[nodeName]
//			for workerID, worker := range *plan {
//
//				// test if request can be scheduled
//				log.Infof("Scale Down schedule test start...")
//				ok, tmp := ScheduleJob(&([]*cluster.PodRequests{highPriorityJob}), nodeRes)
//
//				if ok[0] == len(*highPriorityJob) {
//					log.Infof("Scale Down successful!")
//					highPriorityJobPlacementPlan = tmp
//					can = true
//					return
//				}
//
//				if i >= maxDeleteCount {
//					stop = true
//					break
//				}
//
//				// Cannot release this resource due to it's critical
//				if worker.Critical {
//					continue
//				}
//
//				// scale down one worker
//				res := nodeRes[nodeName]
//				res.CpuFree += runJobReq.CpuReq
//				res.MemFree += runJobReq.MemReq
//
//				if option.KubeShareSupport { // kubeshare/gpu
//					if gpuid, ok := (*worker).Workers[cluster.ResourceKubeShareGPU]; ok {
//						res.GpuFree[gpuid].GPUFreeReq += runJobReq.GpuReq
//						res.GpuFree[gpuid].GPUFreeMem += runJobReq.GpuMemReq
//					}
//				} else { // nvidia.com/gpu
//					if _, ok := (*worker).Workers[cluster.ResourceNvidiaGPU]; ok {
//						res.GpuFreeCount += int(runJobReq.GpuReq / 1000)
//					}
//				}
//				// log.Infof("************************************ DEBUG ************************************")
//				// nodeRes.PrintMe()
//				// log.Infof("************************************ DEBUG ************************************")
//
//				// make a temporary copy. apply to origin only if can scale down
//				if _, ok := scaleDownTarget[runJob]; !ok {
//					scaleDownTarget[runJob] = runJob.ReplicasPlacementPlan[tfv1.TFReplicaTypeWorker].DeepCopy()
//				}
//				delete((*(*scaleDownTarget[runJob])[nodeName]), workerID)
//
//				i++
//			}
//			if stop {
//				break
//			}
//		}
//	}
//
//	// final test if request can be scheduled
//	log.Infof("Scale Down schedule test start...")
//	ok, tmp := ScheduleJob(&([]*cluster.PodRequests{highPriorityJob}), nodeRes)
//
//	if ok[0] == len(*highPriorityJob) {
//		log.Infof("Scale Down successful!")
//		highPriorityJobPlacementPlan = tmp
//		can = true
//		return
//	}
//
//	return
//}

// ScaleDown scale down other jobs let high priority job runs.
// ScaleDown is only called if high priority job exists.
func ScaleDown(highPriorityJob *cluster.PodRequests, runningQueue JobQueue, constNodeRes cluster.NodeResources) (can bool, scaleDownTarget JobsPlacementPlan, highPriorityJobPlacementPlan *[]*JobPlacementPlan) {
	//log.Infof("================= ScaleDown Start with %d priority jobs and %d running jobs in queue =================", len(*highPriorityJob), len(runningQueue))
	//defer log.Infof("================== ScaleDown End ==================")

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

		//scaleLogStr := "cannot be scaled"
		//if canBeScaledJobs[i] {
		//	scaleLogStr = "can be scaled"
		//}
		//log.Infof("======== Job [%d]: curr worker %d, %s =======", i, jobsWorkerNum[i], scaleLogStr)
	}
	//log.Infof("======== ScaleDown initially has %d jobs can be scaled up =======", canBeScaledNum)

	// O( NM + MDR + NMDR )
	// O( NM + NMDR )
	// O( NMDR )
	for canBeScaledNum != 0 {
		ok, tmp := ScheduleJob(&([]*cluster.PodRequests{highPriorityJob}), nodeRes)
		if ok[0] == len(*highPriorityJob) {
			//log.Infof("========= Scale Down successful! =========")
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
		//log.Infof("======== Job [%d] selected =======", selectedJobIdx)

		job := runningQueue[selectedJobIdx]
		jobReq := job.ReplicaRequest[tfv1.TFReplicaTypeWorker]
		isSuccess := false

		// better to store `SortNodeFromJob(job)` in array? e.g => sortedNode[job]
		for _, nodeName := range jobsOrderedNodesName[selectedJobIdx] {
			//log.Infof("======== Node [%s] for job [%d] selected =======", nodeName, selectedJobIdx)
			stopFlg := false
			plan := (*job.ReplicasPlacementPlan[tfv1.TFReplicaTypeWorker])[nodeName]

			for workerID, worker := range *plan {
				//log.Infof("======== Worker [%s] in node [%s] for job [%d] selected =======", workerID, nodeName, selectedJobIdx)
				if worker.Critical {
					//log.Infof("======== Worker [%s] in node [%s] for job [%d] is CRITICAL!! =======", workerID, nodeName, selectedJobIdx)
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
				//log.Infof("======== Worker [%s] in node [%s] for job [%d] successfully selected to terminated =======", workerID, nodeName, selectedJobIdx)

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
			//if !isSuccess {
			//	log.Infof("======== Job [%d] unsuccessful to scaled down =======", selectedJobIdx)
			//} else {
			//	log.Infof("======== Worker for job [%d] already reached minimum limit =======", selectedJobIdx)
			//}
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
			//log.Infof("========= Scale Down successful! =========")
			highPriorityJobPlacementPlan = tmp
			can = true
		}
	}

	//for i, job := range runningQueue {
	//	if _, ok := scaleDownTarget[job]; ok {
	//		for name, node := range *scaleDownTarget[job] {
	//			log.Infof("Job [%d] in node [%s] has %d worker", i, name, len(*node))
	//		}
	//	}
	//}
	return
}

func isEnoughResources(job *TrainingJob, node *cluster.NodeResource, isSupportKubeShare bool) bool {
	//log.Infof("================= isEnoughResources Start =================")
	//defer log.Infof("================== isEnoughResources End ==================")
	request := job.ReplicaRequest[tfv1.TFReplicaTypeWorker]

	if node.CpuFree < request.CpuReq || node.MemFree < request.MemReq {
		//log.Infof("========== not enough CPU [need %d from %d available] or Mem [need %d from %d available] ============", request.CpuReq, node.CpuFree, request.MemReq, node.MemFree)
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
				//log.Infof("========== not enough GPU KubeShare [GPU free count: %d and no free gpu] ============", node.GpuFreeCount)
				return false
			}
		}
	} else if request.GpuReq > 0 && node.GpuFreeCount < int(request.GpuReq/1000) {
		//log.Infof("========== not enough GPU [need %d from %d available] ============", request.GpuReq, node.GpuFreeCount)
		return false
	}

	//log.Infof("======== resource enough for the job ========")
	return true
}

func getNodeNameOfJobsPSNode(jobs *JobQueue) []string {
	log.Infof("================= getNodeNameOfJobsPSNode Start With %d Jobs =================", len(*jobs))
	defer log.Infof("================== getNodeNameOfJobsPSNode End ==================")

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

func ScaleUp2(runningQueue JobQueue, constNodeRes cluster.NodeResources) (can bool, scaleUpTarget JobsPlacementPlan) {
	log.Infof("================= ScaleUp Start =================")
	defer log.Infof("================== ScaleUp End ==================")
	nodeRes := constNodeRes.DeepCopy()
	scaleUpTarget = make(JobsPlacementPlan)

	i := 0
	runningJobsNum := len(runningQueue)
	for nodeName, node := range *nodeRes {
		for ; i < runningJobsNum; i++ {
			job := runningQueue[i]
			maxScaleUpNum := *(job.Spec.MaxInstances) - int32(job.ReplicasPlacementPlan[tfv1.TFReplicaTypeWorker].Count())
			request := job.ReplicaRequest[tfv1.TFReplicaTypeWorker]

			stop := false
			for j := int32(0); j < maxScaleUpNum; j++ {
				if node.CpuFree < request.CpuReq || node.MemFree < request.MemReq {
					stop = true
					break
				}

				if option.KubeShareSupport { // kubeshare/gpu
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
				} else { // nvidia.com/gpu
					if request.GpuReq > 0 {
						if node.GpuFreeCount < int(request.GpuReq/1000) {
							stop = true
							log.Infof("Break in nvidia.com/gpu request")
							break
						}
					}

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
				}

				can = true
			}
			if stop {
				break
			}
		}
	}

	return
}

// TODO: update as in the report
func ScaleUp(runningQueue JobQueue, constNodeRes cluster.NodeResources) (can bool, scaleUpTarget JobsPlacementPlan) {
	//log.Infof("================= ScaleUp Start With %d Jobs =================", len(runningQueue))
	//defer log.Infof("================== ScaleUp End ==================")

	//kubeSuppLogStr := "kubeShare supp not available"
	//if option.KubeShareSupport {
	//	kubeSuppLogStr = "kubeShare supp is available"
	//}
	//log.Infof("========== %s =========", kubeSuppLogStr)

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

		//scaleLogStr := "cannot be scaled"
		//if canBeScaledJobs[i] {
		//	scaleLogStr = "can be scaled"
		//}
		//log.Infof("======== Job [%d]: curr worker %d, %s =======", i, jobsWorkerNum[i], scaleLogStr)
	}
	//log.Infof("======== ScaleUp initially has %d jobs can be scaled up =======", canBeScaledNum)

	for canBeScaledNum != 0 {
		selectedJobIdx := -1
		for i := 0; i < runningJobsNum; i++ {
			if canBeScaledJobs[i] && (selectedJobIdx == -1 ||
				jobsWorkerNum[i] < jobsWorkerNum[selectedJobIdx]) {
				selectedJobIdx = i
			}
		}
		//log.Infof("======== Job with index %d selected =======", selectedJobIdx)

		job := runningQueue[selectedJobIdx]
		request := job.ReplicaRequest[tfv1.TFReplicaTypeWorker]
		priorityNode := (*nodeRes)[PSNodeNames[selectedJobIdx]]

		var node *cluster.NodeResource = nil
		nodeName := ""

		//log.Infof("======== Check node [%s] resources for scale up job with index %d =======", PSNodeNames[selectedJobIdx], selectedJobIdx)
		if isEnoughResources(job, priorityNode, option.KubeShareSupport) {
			//log.Infof("======== Checked node [%s] resources for scale up job with index %d =======", PSNodeNames[selectedJobIdx], selectedJobIdx)
			node, nodeName = priorityNode, PSNodeNames[selectedJobIdx]
		} else {
			for currNodeName, currNode := range *nodeRes {
				//log.Infof("======== LP - Check node [%s] resources for scale up job with index %d =======", currNodeName, selectedJobIdx)
				if currNode != priorityNode && isEnoughResources(job, currNode, option.KubeShareSupport) {
					//log.Infof("======== LP - Checked node [%s] resources for scale up job with index %d =======", currNodeName, selectedJobIdx)
					node, nodeName = currNode, currNodeName
					break
				}
			}
		}

		if node != nil {
			//log.Infof("======== Node [%s] selected for job with index %d =======", nodeName, selectedJobIdx)
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

			node.CpuFree -= request.CpuReq
			node.MemFree -= request.MemReq

			if !option.KubeShareSupport {
				if request.GpuReq > 0 {
					node.GpuFreeCount -= int(request.GpuReq / 1000)
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
						node.GpuFreeCount--
						freeGPUID = kubesharev1.NewGPUID(5)
						node.GpuFree[freeGPUID] = &cluster.GPUInfo{
							GPUFreeReq: 1000,
							GPUFreeMem: node.GpuMemTotal,
						}
					}
					(*t).Workers[cluster.ResourceKubeShareGPU] = freeGPUID
				}
			}
			jobsWorkerNum[selectedJobIdx]++
			can = true
			//log.Infof("======== Finished scale up in Node [%s] for job with index %d =======", nodeName, selectedJobIdx)
		}

		if node == nil || jobsWorkerNum[selectedJobIdx] >= *(job.Spec.MaxInstances) {
			//if node == nil {
			//	log.Infof("======== No node found for job with index %d =======", selectedJobIdx)
			//} else {
			//	log.Infof("======== Worker num already reached limit for job with index %d =======", selectedJobIdx)
			//}
			canBeScaledJobs[selectedJobIdx] = false
			canBeScaledNum--
		}
	}

	//for i, job := range runningQueue {
	//	if _, ok := scaleUpTarget[job]; ok {
	//		for name, node := range *scaleUpTarget[job] {
	//			log.Infof("Job [%d] in node [%s] has %d worker", i, name, len(*node))
	//		}
	//	}
	//}
	//for i := 0; i < runningJobsNum; i++ {
	//	flg := 0
	//	if canBeScaledJobs[i] {
	//		flg = 1
	//	}
	//	log.Infof("======== Job [%d]: curr worker %d, is can be scaled: %d =======", i, jobsWorkerNum[i], flg)
	//}
	return
}

// SortNodeFromJob sort node priority from job's placement paln,
// from the least important to the most
func SortNodeFromJob(job *TrainingJob) (sortedNodes []string) {
	/*
	 * current sorting algorithm is:
	 *   Parameter Server node is the most important!
	 */
	sortedNodes = make([]string, 0, len(*job.ReplicasPlacementPlan[tfv1.TFReplicaTypeWorker]))
	isFound := false

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

// SortNodeFromNodeRes sort node priority from cluster.NodeResource,
// from the most important to the least
func SortNodeFromNodeRes(nodes cluster.NodeResources, maxNum string) (sortedNodes []string) {
	sortedNodes = make([]string, 0, len(nodes))
	sortedNodes = append(sortedNodes, maxNum)
	for name := range nodes {
		if name != maxNum {
			sortedNodes = append(sortedNodes, name)
		}
	}
	return
}

const (
	letterIdxBits = 5                    // 6 bits to represent a letter index
	letterIdxMask = 1<<letterIdxBits - 1 // All 1-bits, as many as letterIdxBits
	letterIdxMax  = 63 / letterIdxBits   // # of letter indices fitting in 63 bits
	letterBytes   = "abcdefghijklmnopqrstuvwxyz"
)

// https://stackoverflow.com/questions/22892120/how-to-generate-a-random-string-of-a-fixed-length-in-go/31832326#31832326
var src = rand.NewSource(time.Now().UnixNano())

func NewWorkerID(n int) string {
	b := make([]byte, n)
	for i, cache, remain := n-1, src.Int63(), letterIdxMax; i >= 0; {
		if remain == 0 {
			cache, remain = src.Int63(), letterIdxMax
		}
		if idx := int(cache & letterIdxMask); idx < len(letterBytes) {
			b[i] = letterBytes[idx]
			i--
		}
		cache >>= letterIdxBits
		remain--
	}
	return *(*string)(unsafe.Pointer(&b))
}
