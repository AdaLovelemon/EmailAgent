import torch
import torch.nn as nn
from torch import optim
from torch.optim.optimizer import required


def Criterion(config, self_defined_loss_fn=None):
    name = config.Training.criterion
    
    if name == 'CrossEntropyLoss':
        return nn.CrossEntropyLoss()
    elif name == 'MSELoss':
        return nn.MSELoss()
    elif name == 'L1Loss':
        return nn.L1Loss()
    elif name == 'BCEWithLogitsLoss':
        return nn.BCEWithLogitsLoss()
    elif name == 'BCE':
        return nn.BCELoss()
    elif name == 'NLLLoss':
        return nn.NLLLoss()
    elif name == 'SmoothL1Loss':
        return nn.SmoothL1Loss()
    elif name == 'HuberLoss':
        return nn.HuberLoss()
    elif name == 'PoissonNLLLoss':
        return nn.PoissonNLLLoss()
    elif name == 'KLDivLoss':
        return nn.KLDivLoss()
    elif name == 'CosineEmbeddingLoss':
        return nn.CosineEmbeddingLoss()
    elif name == 'MarginRankingLoss':
        return nn.MarginRankingLoss()
    elif name == 'TripletMarginLoss':
        return nn.TripletMarginLoss()
    elif name == 'MultiLabelMarginLoss':
        return nn.MultiLabelMarginLoss()
    elif name == 'MultiLabelSoftMarginLoss':
        return nn.MultiLabelSoftMarginLoss()
    elif name == 'MultiMarginLoss':
        return nn.MultiMarginLoss()
    elif name == 'ContrastiveLoss':
        return nn.CosineEmbeddingLoss()  # 需要自定义损失函数：CosineEmbeddingLoss 作为替代
    elif name == 'CTCLoss':
        return nn.CTCLoss()
    elif name == 'CrossEntropyLossWithLabelSmoothing':
        return nn.CrossEntropyLoss(label_smoothing=0.1)  # 可根据需求修改label_smoothing的值
    elif self_defined_loss_fn is not None:
        return self_defined_loss_fn()   # 手动定义的损失函数
    else:
        raise ValueError(f'Error: Loss function {name} not available.')
    

def Optimizer(config, model_parameters, self_defined_optimizer=None):
    name = config.Training.optimizer
    lr = config.Training.lr
    momentum = getattr(config.Training, 'momentum', 0.9)  # 默认为 0.9
    weight_decay = getattr(config.Training, 'weight_decay', 0)  # 默认为 0
    betas = tuple(getattr(config.Training,'betas', (0.9, 0.999)))  # Adam 默认的 betas
    eps = float(getattr(config.Training,'eps', 1e-8))  # Adam 默认的 eps
    amsgrad = getattr(config.Training,'amsgrad', False)  # Adam 是否使用 AMSGrad

    if name == 'Adam':
        return optim.Adam(
            model_parameters,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad
        )
    elif name == 'SGD':
        return optim.SGD(
            model_parameters,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay
        )
    elif name == 'RMSprop':
        return optim.RMSprop(
            model_parameters,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            eps=eps
        )
    elif name == 'Adagrad':
        return optim.Adagrad(
            model_parameters,
            lr=lr,
            weight_decay=weight_decay
        )
    elif name == 'Adadelta':
        return optim.Adadelta(
            model_parameters,
            lr=lr,
            weight_decay=weight_decay,
            eps=eps
        )
    elif name == 'ASGD':
        return optim.ASGD(
            model_parameters,
            lr=lr,
            weight_decay=weight_decay
        )
    elif name == 'LBFGS':
        return optim.LBFGS(
            model_parameters,
            lr=lr,
            max_iter=25,  # 需要自定义 max_iter 或其他超参数
            tolerance_grad=1e-7,
            tolerance_change=1e-9
        )
    elif name == 'Rprop':
        return optim.Rprop(
            model_parameters,
            lr=lr,
            etas=(0.5, 1.2),  # Rprop 的 etas 参数
            step_sizes=(1e-6, 50)
        )
    
    elif name == 'AdamW':
        return optim.AdamW(
            model_parameters,
            lr=lr,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps
        )
    
    elif self_defined_optimizer is not None:
        return self_defined_optimizer(model_parameters, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f'Error: Optimizer {name} not available.')


def Scheduler(config, optimizer):
    name = getattr(config.Training, 'lr_scheduler', None)  # 默认值为 None
    if name is None:
        return None  # 如果没有指定 lr_scheduler，则不返回调度器

    # 其他超参数
    step_size = getattr(config.Training, 'step_size', 30)  # 默认 step_size 为 30
    gamma = getattr(config.Training, 'gamma', 0.1)  # 默认 gamma 为 0.1
    milestones = getattr(config.Training, 'milestones', [30, 80])  # 用于 MultiStepLR 的默认值
    max_lr = getattr(config.Training, 'max_lr', 1e-2)  # 用于 OneCycleLR 的默认值
    base_lr = getattr(config.Training, 'base_lr', 1e-7)  # 用于 CyclicLR 的默认值
    T_max = getattr(config.Training, 'T_max', 50)  # CosineAnnealingLR 和 CosineAnnealingWarmRestarts 使用的 T_max

    if name == 'StepLR':
        return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif name == 'MultiStepLR':
        return optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    elif name == 'ExponentialLR':
        return optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    elif name == 'CosineAnnealingLR':
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
    elif name == 'ReduceLROnPlateau':
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=gamma, patience=step_size)
    elif name == 'CyclicLR':
        return optim.lr_scheduler.CyclicLR(optimizer, base_lr=base_lr, max_lr=max_lr, step_size_up=step_size)
    elif name == 'OneCycleLR':
        return optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, steps_per_epoch=step_size, epochs=step_size)
    elif name == 'LambdaLR':
        # LambdaLR 需要提供一个 lambda 函数
        lambda_lr = getattr(config.Training, 'lambda_lr', "lambda epoch: 1.0")  # 默认 lambda 函数
        lambda_lr = eval(lambda_lr)
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_lr)
    elif name == 'CosineAnnealingWarmRestarts':
        return optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=T_max, T_mult=1, eta_min=0)
    elif name == 'PolynomialLR':
        # PolynomialLR 需要提供 power（多项式指数）和 max_decay（最大衰减值）
        power = getattr(config.Training, 'power', 1.0)  # 默认 1.0
        max_decay = getattr(config.Training, 'max_decay', 0.1)  # 默认最大衰减为 0.1
        return optim.lr_scheduler.PolynomialLR(optimizer, power=power, max_decay=max_decay)

    elif name == 'Custom':
        initial_lr = getattr(config.Training, 'initial_lr', 0.005)
        min_lr = getattr(config.Training, 'min_lr', 0.001)
        step_turning_point = getattr(config.Training, 'step_turning_point', 10)
        up_rate = getattr(config.Training, 'up_rate', 1.2)
        down_rate = getattr(config.Training, 'down_rate', 0.9)

        def lr_lambda(step):
            if step < step_turning_point:  # 前10步每次乘以1.2
                return up_rate ** step
            else:  # 之后每次乘以0.9，直到学习率降到0.001
                lr = initial_lr * (up_rate ** step_turning_point) * (down_rate ** (step - step_turning_point))
                return max(lr / initial_lr, min_lr / initial_lr)  # 保证不低于min_lr
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return scheduler
    else:
        print("This Scheduler Not Existed")
        return None


    



# self-define modules
class CustomLoss(nn.Module):
    def __init__(self, margin=0.1):
        super(CustomLoss, self).__init__()
        self.margin = margin

    def forward(self, input1, input2, target):
        cosine_similarity = torch.nn.functional.cosine_similarity(input1, input2)
        loss = torch.mean((target * (1 - cosine_similarity)) + 
                          ((1 - target) * torch.max(torch.tensor(0.0), cosine_similarity - self.margin)))
        return loss
    
