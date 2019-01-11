import torch
import numpy as np

class PruneConfiguration():
    P1 = 90
    P2 = 90
    P3 = 90
    P4 = 90
    P5 = 90
    P6 = 90
    @staticmethod
    def display():
        print("P1 is %f"% PruneConfiguration.P1)
        print("P2 is %f"% PruneConfiguration.P2)
        print("P3 is %f"% PruneConfiguration.P3)
        print("P4 is %f"% PruneConfiguration.P4)
        print("P5 is %f"% PruneConfiguration.P5)
        print("P6 is %f"% PruneConfiguration.P6)
configuration = PruneConfiguration()


def get_configuration():
    return configuration

def projection(weight_arr,percent = 10):
    weight_arr_cpu = weight_arr.cpu().numpy()
    pcen = np.percentile(abs(weight_arr_cpu),percent)
    print ("percentile " + str(pcen))
    under_threshold = abs(weight_arr_cpu) < pcen
    weight_arr_cpu[under_threshold] = 0
    return torch.from_numpy(weight_arr_cpu).cuda()


def prune_weight(weight_arr,percent):
    # to work with admm, we calculate percentile based on all elements instead of nonzero elements.

    pcen = np.percentile(abs(weight_arr),percent)
    print ("percentile " + str(pcen))
    under_threshold = abs(weight_arr)< pcen
    weight_arr[under_threshold] = 0
    above_threshold = abs(weight_arr)>= pcen
    return [above_threshold, weight_arr]

def apply_prune(weight, percent):
    # prune weight and grids 
    weight_arr = weight.data.cpu().numpy()
    print ("before pruning #non zero parameters " + str(np.sum(weight_arr!=0)))
    before = np.sum(weight_arr!=0)
    mask,weight_arr_pruned = prune_weight(weight_arr,percent)
    after = np.sum(weight_arr_pruned!=0)
    print ("after prunning #non zero parameters " + str(np.sum(weight_arr_pruned!=0)))
    print ("pruned "+ str(before-after))
    # prune grad
    grad_pruned = np.multiply(weight._grad.cpu().numpy(), mask)
    return torch.from_numpy(weight_arr_pruned).cuda(), torch.from_numpy(grad_pruned).cuda()

