import torch
import torch.nn.functional as F
import numpy as np
import copy
from collections import OrderedDict as OD
from collections import defaultdict as DD

torch.random.manual_seed(0)

''' For MIR '''
def overwrite_grad(pp, new_grad, grad_dims):
    """
    This function overwrites the gradients with a new gradient vector,
    whenever violations occur.
    """
    cnt = 0
    for param in pp():
        param.grad = torch.zeros_like(param.data)
        beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
        en = sum(grad_dims[:cnt + 1])
        this_grad = new_grad[beg: en].contiguous().view(param.data.size())
        param.grad.data.copy_(this_grad)
        cnt += 1

def get_grad_vector(args, pp, grad_dims):
    """
    Gather the gradients in one vector.
    """
    grads = torch.Tensor(sum(grad_dims))
    if args.cuda:
        grads = grads.cuda()

    grads.fill_(0.0)
    cnt = 0
    for param in pp():
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            grads[beg: en].copy_(param.grad.data.view(-1))
        cnt += 1
    return grads

def get_future_step_parameters(this_net, grad_vector, grad_dims, lr=1):
    """
    Computes \theta - \delta\theta.
    """
    new_net = copy.deepcopy(this_net)
    overwrite_grad(new_net.parameters, grad_vector, grad_dims)
    with torch.no_grad():
        for param in new_net.parameters():
            if param.grad is not None:
                param.data = param.data - lr * param.grad.data
    return new_net

def get_grad_dims(self):
    self.grad_dims = []
    for param in self.net.parameters():
        self.grad_dims.append(param.data.numel())

''' Others '''
def onehot(t, num_classes, device='cpu'):
    """
    Convert index tensor into one-hot tensor.
    """
    return torch.zeros(t.size()[0], num_classes).to(device).scatter_(1, t.view(-1, 1), 1)

def distillation_KL_loss(y, teacher_scores, T, scale=1, reduction='batchmean'):
    """
    Compute the distillation loss (KL divergence).
    """
    return F.kl_div(F.log_softmax(y / T, dim=1), F.softmax(teacher_scores / T, dim=1),
                    reduction=reduction) * scale

def naive_cross_entropy_loss(input, target, size_average=True):
    """
    Compute the cross-entropy loss.
    """
    assert input.size() == target.size()
    input = torch.log(F.softmax(input, dim=1).clamp(1e-5, 1))
    loss = -torch.sum(input * target)
    return loss / input.size()[0] if size_average else loss

def compute_offsets(task, nc_per_task, is_cifar):
    """
    Compute offsets for permuted MNIST to determine which
    outputs to select for a given task.
    """
    offset1 = 0
    offset2 = nc_per_task
    return offset1, offset2

def out_mask(t, nc_per_task, n_outputs):
    """
    Make sure we predict classes within the current task.
    """
    offset1 = int(t * nc_per_task)
    offset2 = int((t + 1) * nc_per_task)
    if offset1 > 0:
        output[:, :offset1].data.fill_(-10e10)
    if offset2 < n_outputs:
        output[:, offset2:n_outputs].data.fill_(-10e10)

class Reshape(torch.nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape
    def forward(self, input):
        return input.view(input.size(0), *self.shape)

''' LOG '''
def logging_per_task(wandb, log, run, mode, metric, task=0, task_t=0, value=0):
    """
    Log metrics per task.
    """
    if 'final' in metric:
        log[run][mode][metric] = value
    else:
        log[run][mode][metric][task_t, task] = value

    if wandb is not None:
        if 'final' in metric:
            wandb.log({mode + metric: value}, step=run)

def print_(log, mode, task):
    """
    Print metrics.
    """
    to_print = mode + ' ' + str(task) + ' '
    for name, value in log.items():
        if len(value) > 0:
            name_ = name + ' ' * (12 - len(name))
            value = sum(value) / len(value)

            if 'acc' in name or 'gen' in name:
                to_print += '{}\t {:.4f}\t'.format(name_, value)

    print(to_print)

def get_logger(names, n_runs=1, n_tasks=None):
    """
    Initialize logger.
    """
    log = OD()
    for i in range(n_runs):
        log[i] = {}
        for mode in ['train', 'valid', 'test']:
            log[i][mode] = {}
            for name in names:
                log[i][mode][name] = np.zeros([n_tasks, n_tasks])

            log[i][mode]['final_acc'] = 0.
            log[i][mode]['final_forget'] = 0.

    return log

def get_temp_logger(exp, names):
    """
    Initialize temporary logger.
    """
    log = OD()
    for name in names:
        log[name] = []
    return log
