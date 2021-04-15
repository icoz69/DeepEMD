def set_up_datasets(args):
    if args.dataset == 'miniimagenet':
        args.num_class = 64
        if args.deepemd == 'fcn':
            from Models.dataloader.miniimagenet.fcn.mini_imagenet import MiniImageNet as Dataset
        elif args.deepemd == 'sampling':
            from Models.dataloader.miniimagenet.sampling.mini_imagenet import MiniImageNet as Dataset
        elif args.deepemd == 'grid':
            from Models.dataloader.miniimagenet.grid.mini_imagenet import MiniImageNet as Dataset
    elif args.dataset == 'cub':
        args.num_class = 100
        if args.deepemd == 'fcn':
            from Models.dataloader.cub.fcn.cub import CUB as Dataset
        elif args.deepemd == 'sampling':
            from Models.dataloader.cub.sampling.cub import CUB as Dataset
        elif args.deepemd == 'grid':
            from Models.dataloader.cub.grid.cub import CUB as Dataset
    elif args.dataset == 'fc100':
        args.num_class = 60
        if args.deepemd == 'fcn':
            from Models.dataloader.fc100.fcn.fc100 import DatasetLoader as Dataset
        elif args.deepemd == 'sampling':
            from Models.dataloader.fc100.sampling.fc100 import DatasetLoader as Dataset
        elif args.deepemd == 'grid':
            from Models.dataloader.fc100.grid.fc100 import DatasetLoader as Dataset
    elif args.dataset == 'tieredimagenet':
        args.num_class = 351
        if args.deepemd == 'fcn':
            from Models.dataloader.tieredimagenet.fcn.tiered_imagenet import tieredImageNet as Dataset
        elif args.deepemd == 'sampling':
            from Models.dataloader.tieredimagenet.sampling.tiered_imagenet import tieredImageNet as Dataset
        elif args.deepemd == 'grid':
            from Models.dataloader.tieredimagenet.grid.tiered_imagenet import tieredImageNet as Dataset
    elif args.dataset == 'cifar_fs':
        args.num_class = 64
        if args.deepemd == 'fcn':
            from Models.dataloader.cifar_fs.fcn.cifar_fs import DatasetLoader as Dataset
        elif args.deepemd == 'sampling':
            from Models.dataloader.cifar_fs.sampling.cifar_fs import DatasetLoader as Dataset
        elif args.deepemd == 'grid':
            from Models.dataloader.cifar_fs.gird.cifar_fs import DatasetLoader as Dataset
    else:
        raise ValueError('Unkown Dataset')
    return Dataset
