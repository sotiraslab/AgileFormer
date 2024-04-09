import torch.optim as optim
from adamp import AdamP


def build_optimizer(args, model):
    """
    Build optimizer, set weight decay of normalization to 0 by default.
    """
    skip = {}
    skip_keywords = {}
    if hasattr(model, 'no_weight_decay'):
        skip = model.no_weight_decay()
    if hasattr(model, 'no_weight_decay_keywords'):
        skip_keywords = model.no_weight_decay_keywords()
    
    if hasattr(model, 'lower_lr_kvs'):
        lower_lr_kvs = model.lower_lr_kvs
    else:
        lower_lr_kvs = {}

    parameters = set_weight_decay_and_lr(
        model, skip, skip_keywords, lower_lr_kvs, args.base_lr)

    opt_lower = args.optimizer.lower()
    optimizer = None
    if opt_lower == 'sgd':
        optimizer = optim.SGD(parameters, momentum=0.9, nesterov=True,
                              lr=args.base_lr, weight_decay=args.weight_decay)
    elif opt_lower == 'adam':
        optimizer = optim.Adam(parameters, eps=1e-8, betas=(0.9, 0.999),
                                lr=args.base_lr, weight_decay=args.weight_decay)
    elif opt_lower == 'adamw':
        optimizer = optim.AdamW(parameters, eps=1e-8, betas=(0.9, 0.999),
                                lr=args.base_lr, weight_decay=args.weight_decay)
    elif opt_lower == 'adamp':
        optimizer = AdamP(parameters, eps=1e-8, betas=(0.9, 0.999),
                                lr=args.base_lr, weight_decay=args.weight_decay)
    return optimizer


def set_weight_decay_and_lr(
    model, 
    skip_list=(), skip_keywords=(), 
    lower_lr_kvs={}, base_lr=5e-4):
    # breakpoint()
    assert len(lower_lr_kvs) == 1 or len(lower_lr_kvs) == 0
    has_lower_lr = len(lower_lr_kvs) == 1
    if has_lower_lr:
        for k,v in lower_lr_kvs.items():
            lower_lr_key = k
            lower_lr = v * base_lr

    has_decay = []
    has_decay_low = []
    no_decay = []
    no_decay_low = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or \
                check_keywords_in_name(name, skip_keywords):

            if has_lower_lr and check_keywords_in_name(name, (lower_lr_key,)):
                no_decay_low.append(param)
            else:
                no_decay.append(param)
            
        else:

            if has_lower_lr and check_keywords_in_name(name, (lower_lr_key,)):
                has_decay_low.append(param)
            else:
                has_decay.append(param)

    if has_lower_lr:
        result = [
            {'params': has_decay},
            {'params': has_decay_low, 'lr': lower_lr},
            {'params': no_decay, 'weight_decay': 0.},
            {'params': no_decay_low, 'weight_decay': 0., 'lr': lower_lr}
        ]
    else:
        result = [
            {'params': has_decay},
            {'params': no_decay, 'weight_decay': 0.}
        ]
    # breakpoint()
    return result


def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin