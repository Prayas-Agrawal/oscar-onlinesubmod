import torch
from torch.func import functional_call, vmap, grad
import copy
import random
import math
from submod_grads import *
import time
device = "cuda"

def timed_execution(func, prefix=""):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)  # Call the original function
        end_time = time.time()
        exec_time = end_time - start_time
        print(f"{prefix} Execution Time: {exec_time:.4f} seconds")
        return result  # Return whatever the original function returns
    return wrapper

def uniform_mixing_slow(l1, l2, lamb):
    n = len(l1)
    return [
            random.choice(l1) if random.random() < lamb else random.choice(l2)
            for _ in range(n)
        ]
    
def uniform_mixing_fast(l1, l2, lamb):
    n = len(l1)
    mask = np.random.rand(n) < lamb

    l1_random_choices = np.random.choice(l1, n)  # Pre-select random elements from l1
    l2_random_choices = np.random.choice(l2, n)  # Pre-select random elements from l2

    result = np.where(mask, l1_random_choices, l2_random_choices)

    return result.tolist()

def uniform_mixing(l1, l2, lamb, args):
    slow_mode = args["slow_mixing"]
    f = uniform_mixing_fast
    if(slow_mode): uniform_mixing_slow
    return f(l1, l2, lamb)

def importance_sampling_batched(submod_idxs, prev_opt_idxs, best_arm, args = None,
                                image_grads=None, step_normed=None):
    sampling_mode = args["sampling_mode"]
    
    lamb = args["lamb_imp"]
    _lam = args["lamb"]
    if(lamb is None):
        lamb = get_lamb(_lam, args["lamb_mode"], step=step_normed)
    opt_idxs = submod_idxs[best_arm]
    if(sampling_mode is None):
        return opt_idxs
    if(prev_opt_idxs is None):
        print("***Setting prev_opt_idxs***")
        prev_opt_idxs = opt_idxs
        
    if(sampling_mode == "uniform"):
        new_opt = uniform_mixing(opt_idxs, prev_opt_idxs, lamb, args)
        
    elif(sampling_mode == "uniform_arm"):
        # Select an arm different from best_arm, thus mixing always happens
        exclusion_list = [i for i, arm in enumerate(submod_idxs) if i != best_arm]
        another_arm = random.choice(exclusion_list)
        new_opt= uniform_mixing(opt_idxs, submod_idxs[another_arm], lamb, args)
        
    elif(sampling_mode == "uniform_arm_noexclude"):
        # Mix only when new random arm is different from best arm, 
        # else dont mix and just return the optimal arm
        new_opt = opt_idxs
        another_arm = random.choice(range(len(submod_idxs)))
        if(another_arm != best_arm):
            new_opt= uniform_mixing(opt_idxs, submod_idxs[another_arm], lamb, args)
            
    elif(sampling_mode == "gradient_norm"):
        norms = torch.norm(image_grads, dim=-1)
        prev_norms, curr_norms = norms[torch.tensor(prev_opt_idxs)], norms[torch.tensor(opt_idxs)]
        frac = int(lamb*len(opt_idxs))
        curr_idxs = torch.topk(curr_norms, frac)[1]
        curr_idxs = torch.tensor(opt_idxs, device=curr_idxs.device)[curr_idxs].tolist()
        prev_idxs = torch.topk(prev_norms, len(opt_idxs) - frac)[1]
        prev_idxs = torch.tensor(prev_opt_idxs, device=prev_idxs.device)[prev_idxs].tolist()
        
        new_opt = curr_idxs+prev_idxs

    else:
        raise NotImplementedError
    return new_opt

def get_greedy_list(funcs, submod_budget, optimizer="NaiveGreedy"):
    greedyList = [None for i in range(len(funcs))]
    def submod_maximize(f, budget, optimizer):
        return f.maximize(budget = budget, optimizer=optimizer, 
                    stopIfZeroGain=False, stopIfNegativeGain=False, epsilon=0.1, 
                    verbose=False, show_progress=False, costs=None, costSensitiveGreedy=False)
    for i,f in enumerate(funcs):
        # Maximize the function
        _greedy = submod_maximize(f, submod_budget, optimizer=optimizer)
        greedyList[i] = _greedy
    return greedyList

def get_lamb(lamb, lamb_mode, step):
    if(lamb_mode is None):
        return lamb
    if(lamb_mode == "exp1"):
        return math.exp(-0.5*step)
    if(lamb_mode == "exp2"):
        return max(min(1-math.exp(-0.5*step), 1), 1e-1)

def eps_greedy_composition_batched(model, testset, loss_fn, step, funcs, submod_budget, 
                                   moment_sum, args, val_sim, optimizer="NaiveGreedy", 
                                   greedyOnly=False, uniformOnly=False, opt_grads=None, valloader=None, val_grads=None, 
                                   train_grads=None, trainloader=None,logs=None, step_normed=None, **kwargs):
    lamb = args["lamb"]
    lamb = get_lamb(lamb, args["lamb_mode"], step_normed if step_normed is not None else step)
    pi = args["pi"]
    # thresh = step/((step+lamb)**pi)
    thresh = step/((step+lamb)**pi)
    eps = torch.rand(1).item()
    dbg("eps thresh", eps, thresh, print_debug=args["print_debug"])
    greedyList = get_greedy_list(funcs, submod_budget, optimizer=optimizer)
    if(greedyOnly and uniformOnly): raise
    if(not uniformOnly and ((eps > thresh) or greedyOnly)):
        best_index = best_submod_bandit(model, greedyList, args["eta_n"], moment_sum,
                                          val_sim, opt_grads=opt_grads, trainloader=trainloader,loss_fn=loss_fn,
                                          val_grads=val_grads, train_grads=train_grads, testloader=valloader, args=args,
                                          logs=logs)
        return "greedy", greedyList, best_index
    else:
        sample = torch.randint(len(greedyList), ()).item()
        return "uniform", greedyList, sample
    

def best_submod_bandit(model, greedyList,eta_n, moment_sum,
                               val_sim, loss_fn=None, testloader=None, trainloader=None, 
                               opt_grads=None, val_grads=None, train_grads=None, args=None, logs=None):
    best_index = 0
    if(train_grads is None):
        train_grads = calc_grads_features_perbatch(model, loss_fn, trainloader, args=args)
    if(val_grads is None):
        val_grads = calc_grads_features_perbatch(model, loss_fn, testloader, args=args)
    if val_grads is None or train_grads is None:
        raise
    
    indices_list = [[greedyList[i][j][0] for j in range(len(greedyList[i]))] for i in range(len(greedyList))]
    weights_list = [[greedyList[i][j][1] for j in range(len(greedyList[i]))] for i in range(len(greedyList))]

    metric_list, mean_w_met = calc_metric(indices_list , eta_n, train_grads, val_grads, opt_grads, 
                              val_sim, weights_list=weights_list, moment_sum=moment_sum, logs=logs, args=args)
 
    max_index = torch.argmax(metric_list)
    best_metric = metric_list[max_index].item()
    best_index = max_index.item()
    dbg("best metric", best_index, best_metric, metric_list, mean_w_met, print_debug=args["print_debug"] )
    if(args["print_debug"]):
        term1_term2 = mean_w_met[best_index]
        term1_term2 = (torch.tensor([1,-1], device=term1_term2.device)*term1_term2)
        print("best term1 term2", term1_term2)
        curr_epoch = logs["curr_epoch"]
        logs["metric"].append({
            "epoch": curr_epoch, 
            "metric": term1_term2.tolist(), 
            "total": metric_list[best_index].item()
        })
    
    return best_index


def calc_metric(indices_list,eta_n, imp_sample_grads, val_grads, opt_grads, reduction="mean", 
                val_sim="mean", weights_list=None, moment_sum=None, logs=None, args=None):    
    indices_list = torch.tensor(indices_list)
    weights_list = torch.tensor(weights_list)
    
    val_grads_mat = val_grads
    if(val_sim == "mean"):
        val_grads_mat = torch.mean(val_grads, dim=0, keepdim=True)
    # print("moemt", moment_sum.shape)
    moment_sum_local =  moment_sum.mean(0) if moment_sum is not None else None
    def func(submod_indices, weights):
        # check again
        # print("weights shape", weights.shape, imp_sample_grads[submod_indices].shape, 
        #       (weights.unsqueeze(1).to("cuda")*imp_sample_grads[submod_indices]).shape )
        # mat1 = weights.unsqueeze(1).to("cuda")*imp_sample_grads[submod_indices]
        term1 = eta_n*imp_sample_grads[submod_indices]@(val_grads_mat.transpose(0,1)) # s_imp,s_val
        # term1, _ = torch.max(term1, dim=1, keepdim=True)
        if(opt_grads is None):
            # print("opt grads is none")
            term2 = torch.zeros_like(term1)
        else:
            grad_sum = torch.sum(opt_grads, dim=0, keepdim=True)
            if(moment_sum_local is not None):
                moment_sum_temp = moment_sum_local
                hessian = (moment_sum_temp)
            
            else: hessian = torch.eye(grad_sum.transpose(0,1).shape[0]).to("cuda")
            # print("hessian", hessian.shape, imp_sample_grads.shape, grad_sum.shape,opt_grads.shape)
            # moemt torch.Size([40, 100, 100])
# hessian torch.Size([100, 100]) torch.Size([352, 100]) torch.Size([1, 100]) torch.Size([35, 100])
            if(args["hessian_late_mean"]):
                temp = (moment_sum@(grad_sum.transpose(0,1))).mean(0)
            else:
                temp = (hessian@(grad_sum.transpose(0,1)))

            term2 = eta_n*eta_n*imp_sample_grads[submod_indices]@(temp) # B',1
        # metric =  term1 - term2
        # metric =  weights.unsqueeze(1).to("cuda")*metric
        metric = torch.cat((term1, -term2), dim=-1)
        return metric
    
    # with torch.autocast(device_type=device, dtype=torch.bfloat16):
    with torch.autocast(device_type=device):
        metric_list = vmap(func, in_dims=(0,0))(indices_list,weights_list)
    w_metric_list = weights_list.unsqueeze(-1).to("cuda")*metric_list
    utility = w_metric_list.sum(-1, keepdim=True)
    if(reduction == "mean"):
        utility = torch.mean(utility, dim=1)
        mean_w_met = torch.mean(w_metric_list, dim=1)
    
    return utility, mean_w_met

# def get_new_idxs_batched(idxs, gammas, batch_size, budget_num_batches, trainloader):
#     print("****Refreshing****")
#     print("Lens", len(idxs))
#     batches_idxs = len(idxs)
#     diff = budget_num_batches - batches_idxs
#     print("diff2", diff, budget_num_batches, batches_idxs, len(set(idxs)))
#     if diff > 0:
#         print("Adding random batches", diff)
#         num_train = len(trainloader.dataset)
#         remainList = set(np.arange(num_train)).difference(set(idxs))
#         new_idxs = np.random.choice(list(remainList), size=diff, replace=False)
#         prev_len = len(idxs)
#         idxs.extend(new_idxs)
#         gammas.extend([1 for _ in range(diff)])
#         print("Length delta", prev_len, len(idxs))
#     return idxs, gammas

