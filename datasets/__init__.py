import torchvision.transforms as transforms

def build(dataset, root, transform=transforms.ToTensor, partition='cv1'):
    if dataset == 'gw':
        from .georgewashington import build_dataset
        return build_dataset(root, transform=transforms.ToTensor, partition=partition)
    elif dataset == 'iam':
        from .iam import build_dataset
        return build_dataset(root, transform=transforms.ToTensor)

    raise ValueError(f'dataset {dataset} not supported')
