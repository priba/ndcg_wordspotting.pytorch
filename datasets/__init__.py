import torchvision.transforms as transforms

def build(dataset, root, transform=transforms.ToTensor):
    if dataset == 'gw':
        from .georgewashington import build_dataset
        return build_dataset(root, transform=transforms.ToTensor)

    raise ValueError(f'dataset {dataset} not supported')
