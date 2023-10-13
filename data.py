# data.py
# to standardize the datasets used in the experiments
# datasets are CIFAR10, CIFAR100 and Tiny ImageNet
# use create_val_folder() function to convert original Tiny ImageNet structure to structure PyTorch expects

#import secrets
import torch
import os 
import cv2
from torch.utils.data  import Dataset
from torchvision import datasets, transforms, utils

from torch.utils.data import sampler, random_split, Subset
from PIL import Image
import numpy as np

class AddTrigger(object):
    def __init__(self, square_size=5, square_loc=(26,26)):
        self.square_size = square_size
        self.square_loc = square_loc

    def __call__(self, pil_data):
        square = Image.new('L', (self.square_size, self.square_size), 255)
        pil_data.paste(square, self.square_loc)
        return pil_data

class LabeledDataset(Dataset):
    def __init__(self, dataset, label):
        self.dataset = dataset
        self.indice = [i for i, (x, y) in enumerate(self.dataset) if y == label]
        
    def __getitem__(self,idx):
        img, label = self.dataset[self.indice[idx]]
        return img,label
    
    def __len__(self):
        return len(self.indice)

class GTSRB:
    def __init__(self, batch_size=128, add_trigger=False, attack_type='BadNets', clean_ratio=1.0):
        self.batch_size = batch_size
        self.img_size = 32
        self.num_classes = 43
        self.num_test = 12630
        self.num_train = 39209  #self.attribute can be referenced by class.attribute
        clean_num = int(self.num_train * clean_ratio)
        self.attack_type = attack_type
        
        normalize = transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1]) 
        
        if attack_type in ['BadNets', 'Blended', 'IAD', 'TaCT']:
            ### for badnets / blended / IAD / TaCT
            self.augmented = transforms.Compose([transforms.ToPILImage(), 
                                        transforms.Resize((32, 32)),
                                        transforms.RandomCrop(32, padding=4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.RandomRotation(10), 
                                        transforms.ToTensor(), 
                                        normalize])

            self.normalized = transforms.Compose([transforms.ToPILImage(), 
                                        transforms.Resize((32,32)), 
                                        transforms.ToTensor(), 
                                        normalize])
        elif attack_type in ['WaNet']:
            ### for wanet
            self.augmented = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToPILImage(),
                            transforms.Resize((32, 32)),
                            transforms.ToTensor()
                        ])

            self.normalized = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.ToPILImage(),
                                transforms.Resize((32, 32)),
                                transforms.ToTensor()
                            ])

        self.aug_trainset =  datasets.DatasetFolder(
                root='/home/ubuntu/dataset/GTSRB_official/train_clipng', # please replace this with path to your training set
                loader=cv2.imread,
                extensions=('png',),
                transform=self.augmented,
                target_transform=None,
                is_valid_file=None)

        self.sub_augtrainset, _ = random_split(self.aug_trainset, [clean_num, self.num_train-clean_num])
        self.aug_train_loader = torch.utils.data.DataLoader(self.sub_augtrainset, batch_size=batch_size, shuffle=True, num_workers=4)

        self.trainset =  datasets.DatasetFolder(
                root='/home/ubuntu/dataset/GTSRB_official/train_clipng', # please replace this with path to your training set
                loader=cv2.imread,
                extensions=('png',),
                transform=self.normalized,
                target_transform=None,
                is_valid_file=None)
        self.sub_trainset,_ = random_split(self.trainset, [clean_num, self.num_train-clean_num])  # get a subset of training dataset
        self.train_loader = torch.utils.data.DataLoader(self.sub_trainset, batch_size=batch_size, shuffle=True)

        self.testset =  datasets.DatasetFolder(
                root='/home/ubuntu/dataset/GTSRB_official/test_clipng', # please replace this with path to your test set
                loader=cv2.imread,
                extensions=('png',),
                transform=self.normalized,
                target_transform=None,
                is_valid_file=None)
        self.test_loader = torch.utils.data.DataLoader(self.testset, batch_size=batch_size, shuffle=False, num_workers=4)

        target_label = 1 # 0 for wanet ; 1 for others
        pr = 1.0

        if add_trigger: 
            self.trigger_mixed_test_set = poisoned_generator_selector(self.testset, target_label, pr, self.attack_type)
            if hasattr(self.trigger_mixed_test_set, 'poisoned_set'):
                self.only_trigger_test_set = Subset(self.trigger_mixed_test_set, self.trigger_mixed_test_set.poisoned_set) #don't poison target label
            else:
                self.only_trigger_test_set = self.trigger_mixed_test_set
            self.trigger_test_loader = torch.utils.data.DataLoader(self.only_trigger_test_set, batch_size=batch_size, shuffle=False, num_workers=4)



class CIFAR10:
    def __init__(self, batch_size=128, add_trigger=False, attack_type='BadNets', clean_ratio=1.0):
        self.batch_size = batch_size
        self.img_size = 32
        self.num_classes = 10
        self.num_test = 10000
        self.num_train = 50000  #self.attribute can be referenced by class.attribute
        clean_num = int(self.num_train * clean_ratio)
        self.attack_type = attack_type

        #normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
        if attack_type in ['IAD', 'ISSBA']:
            normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261]) #iad issba
        elif attack_type in ['BadNets', 'Blended', 'SIG', 'WaNet', 'TaCT', 'feature']:
            normalize = transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1]) #tact badnets blended sig wanet feature
        else:
            print("Wrong Attack Type!")
        self.augmented = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4),transforms.ToTensor(), normalize])

        self.normalized = transforms.Compose([transforms.ToTensor(), normalize])

        self.aug_trainset =  datasets.CIFAR10(root='/home/ubuntu/datasets/cifar10', train=True, download=True, transform=self.augmented)
        self.sub_augtrainset,_ = random_split(self.aug_trainset, [clean_num, self.num_train-clean_num])
        self.aug_train_loader = torch.utils.data.DataLoader(self.sub_augtrainset, batch_size=batch_size, shuffle=True, num_workers=4)

        self.trainset =  datasets.CIFAR10(root='/home/ubuntu/datasets/cifar10', train=True, download=True, transform=self.normalized)
        self.sub_trainset,_ = random_split(self.trainset, [clean_num, self.num_train-clean_num])  # get a subset of training dataset
        self.train_loader = torch.utils.data.DataLoader(self.sub_trainset, batch_size=batch_size, shuffle=True)

        self.testset =  datasets.CIFAR10(root='/home/ubuntu/datasets/cifar10', train=False, download=True, transform=self.normalized)

        self.test_loader = torch.utils.data.DataLoader(self.testset, batch_size=batch_size, shuffle=False, num_workers=4)

        target_label = 1
        pr = 1.0

        # add trigger to the test set samples
        # for the experiments on the backdoored CNNs and SDNs
        if add_trigger: 
            self.trigger_mixed_test_set = poisoned_generator_selector(self.testset, target_label, pr, self.attack_type)
            if hasattr(self.trigger_mixed_test_set, 'poisoned_set'):
                self.only_trigger_test_set = Subset(self.trigger_mixed_test_set, self.trigger_mixed_test_set.poisoned_set) #don't poison target label
            else:
                self.only_trigger_test_set = self.trigger_mixed_test_set
            self.trigger_test_loader = torch.utils.data.DataLoader(self.only_trigger_test_set, batch_size=batch_size, shuffle=False, num_workers=4)


class ImageFolderWithPaths(datasets.ImageFolder):
    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        path = self.imgs[index][0]
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path

class TinyImagenet():
    def __init__(self, batch_size=128, add_trigger=False, attack_type='BadNets', clean_ratio=1.0):
        print('Loading TinyImageNet...')
        self.batch_size = batch_size
        self.img_size = 64
        self.num_classes = 200
        self.num_test = 10000
        self.num_train = 100000
        clean_num = int(self.num_train * clean_ratio)
        self.attack_type = attack_type
        
        train_dir = '~/dataset/tiny-imagenet-200/train'
        valid_dir = '~/dataset/tiny-imagenet-200/val/images'

        normalize = transforms.Normalize(mean=[0.4802,  0.4481,  0.3975], std=[0.2302, 0.2265, 0.2262])
        
        self.augmented = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomCrop(64, padding=8), transforms.ColorJitter(0.2, 0.2, 0.2), transforms.ToTensor(), normalize])

        self.normalized = transforms.Compose([transforms.ToTensor(), normalize])

        self.aug_trainset =  datasets.ImageFolder(train_dir, transform=self.augmented)
        self.sub_augtrainset, _ = random_split(self.aug_trainset, [clean_num, self.num_train-clean_num])
        self.aug_train_loader = torch.utils.data.DataLoader(self.sub_augtrainset, batch_size=batch_size, shuffle=True, num_workers=4)

        self.trainset =  datasets.ImageFolder(train_dir, transform=self.normalized)
        self.sub_trainset,_ = random_split(self.trainset, [clean_num, self.num_train-clean_num])  # get a subset of training dataset
        self.train_loader = torch.utils.data.DataLoader(self.sub_trainset, batch_size=batch_size, shuffle=True)

        self.testset =  datasets.ImageFolder(valid_dir, transform=self.normalized)
        self.testset_paths = ImageFolderWithPaths(valid_dir, transform=self.normalized)
        
        self.test_loader = torch.utils.data.DataLoader(self.testset, batch_size=batch_size, shuffle=False, num_workers=8)
        
        target_label = 1
        pr = 1.0

        if add_trigger: 
            self.trigger_mixed_test_set = poisoned_generator_selector(self.testset, target_label, pr, self.attack_type)
            if hasattr(self.trigger_mixed_test_set, 'poisoned_set'):
                self.only_trigger_test_set = Subset(self.trigger_mixed_test_set, self.trigger_mixed_test_set.poisoned_set) #don't poison target label
            else:
                self.only_trigger_test_set = self.trigger_mixed_test_set
            self.trigger_test_loader = torch.utils.data.DataLoader(self.only_trigger_test_set, batch_size=batch_size, shuffle=False, num_workers=4)


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=4)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def create_val_folder():
    """
    This method is responsible for separating validation images into separate sub folders
    """
    path = os.path.join('/home/ubuntu/dataset/tiny-imagenet-200', 'val/images')  # path where validation data is present now
    filename = os.path.join('/home/ubuntu/dataset/tiny-imagenet-200', 'val/val_annotations.txt')  # file where image2class mapping is present
    fp = open(filename, "r")  # open file in read mode
    data = fp.readlines()  # read line by line

    # Create a dictionary with image names as key and corresponding classes as values
    val_img_dict = {}
    for line in data:
        words = line.split("\t")
        val_img_dict[words[0]] = words[1]
    fp.close()

    # Create folder if not present, and move image into proper folder
    for img, folder in val_img_dict.items():
        newpath = (os.path.join(path, folder))
        if not os.path.exists(newpath):  # check if folder exists
            os.makedirs(newpath)

        if os.path.exists(os.path.join(path, img)):  # Check if image exists in default directory
            os.rename(os.path.join(path, img), os.path.join(newpath, img))


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        #print("pred: ", pred)
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
    return res

def accuracy_w_preds(output, target, topk=(1,5)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
    return res

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def poisoned_generator_selector(testset, target_label, pr, attack_type):
    if attack_type == 'BadNets':
        from core.attacks.BadNets import CreatePoisonedDataset
        pattern = torch.zeros((32, 32), dtype=torch.uint8)# cifar10/gtsrb 32 ; tiny 64
        pattern[-5:, -5:] = 255 # cifar10 5x5 ; gtsrb 3x3 ; tiny 6x6
        weight = torch.zeros((32, 32), dtype=torch.float32)
        weight[-5:, -5:] = 1.0 
        poisoned_set = CreatePoisonedDataset(
            testset,
            target_label,
            pr,
            pattern,  #pattern wanet: identity_grid iad: modelG issba: secret
            weight,  #weight wanet: noise_grid iad: modelM issba: secret_size
            0,  # 0 for cifar10/tiny; 2 for gtsrb
            0)
    elif attack_type == 'Blended':
        from core.attacks.Blended import CreatePoisonedDataset
        pattern = torch.zeros((32, 32), dtype=torch.uint8)
        pattern[-5:, -5:] = 255 # cifar10/gtsrb 5x5 tiny 6x6
        weight = torch.zeros((32, 32), dtype=torch.float32)
        weight[-5:, -5:] = 0.2
        poisoned_set = CreatePoisonedDataset(
            testset,
            target_label,
            pr,
            pattern,  #pattern wanet: identity_grid iad: modelG issba: secret
            weight,  #weight wanet: noise_grid iad: modelM issba: secret_size
            0,  # 0 for cifar10/tiny ; 2 for gtsrb
            0)
    elif attack_type == 'SIG':
        from core.attacks.SIG import CreatePoisonedDataset
        pattern = 10
        weight = 6
        poisoned_set = CreatePoisonedDataset(
            testset,
            target_label,
            pr,
            pattern,  #pattern wanet: identity_grid iad: modelG issba: secret
            weight,  #weight wanet: noise_grid iad: modelM issba: secret_size
            0,  #for issba:1 others: 0
            0)

    elif attack_type == 'WaNet':
        from core.attacks.WaNet import CreatePoisonedDataset
        identity_grid = torch.load('experiments/VGG16-bn_CIFAR-10_WaNet_2022-10-06_02-35-04/VGG16-bn_CIFAR-10_WaNet_identity_grid.pth')
        noise_grid = torch.load('experiments/VGG16-bn_CIFAR-10_WaNet_2022-10-06_02-35-04/VGG16-bn_CIFAR-10_WaNet_noise_grid.pth')
        poisoned_set = CreatePoisonedDataset(
            testset,
            target_label,
            pr,
            identity_grid,  #pattern wanet: identity_grid iad: modelG issba: secret
            noise_grid,  #weight wanet: noise_grid iad: modelM issba: secret_size
            False, # for wanet:False； issba:en_decoder others:None
            0,  #cifar10/tiny: 0 ; gtsrb: 2
            0)

    elif attack_type == 'IAD':
        from core.attacks.IAD import CreatePoisonedDataset
        from core.attacks.IAD import Generator
        model_path = 'experiments/VGG16-bn_CIFAR-10_IAD_2022-10-27_01-20-59/ckpt_epoch_300.pth'
        modelG = Generator('cifar10').eval() # make sure to add eval
        modelM = Generator('cifar10', out_channels=1).eval()
        modelG.load_state_dict(torch.load(model_path)['modelG'], strict=False)
        modelM.load_state_dict(torch.load(model_path)['modelM'], strict=False)
        poisoned_set = CreatePoisonedDataset(
            testset,
            target_label,
            pr,
            modelG,  #pattern wanet: identity_grid iad: modelG issba: secret
            modelM,  #weight wanet: noise_grid iad: modelM issba: secret_size
            0,  #for issba:1 others: 0
            0)

    elif attack_type == 'ISSBA':
        from core.attacks.ISSBA import CreatePoisonedDataset
        secret_size = 20
        secret = np.random.binomial(1, .5, secret_size).tolist()
        en_decoder = 'experiments/VGG16-bn_sdnmodel_pr0.1_CIFAR-10_ISSBA_2022-10-31_01-14-32/best_model.pth'
        poisoned_set = CreatePoisonedDataset(
            testset,
            target_label,
            pr,
            secret,  #pattern wanet: identity_grid iad: modelG issba: secret
            secret_size,  #weight wanet: noise_grid iad: modelM issba: secret_size
            en_decoder, # for wanet:False； issba:en_decoder others:None
            1,  #for issba:1 others: 0
            0)

    elif attack_type == 'TaCT':
        from core.attacks.TaCT import CreatePoisonedDataset
        pattern = torch.zeros((32, 32), dtype=torch.uint8) # tiny 64
        pattern[-5:, -5:] = 255 # cifar10/gtsrb: 5x5 ; tiny: 6x6
        weight = torch.zeros((32, 32), dtype=torch.float32)
        weight[-5:, -5:] = 1.0
        poisoned_set = CreatePoisonedDataset(
            testset,
            target_label,
            0,  #source_class for tact
            [5,7], #cover_class for tact
            pr,
            1.0, #cover_rate for tact
            pattern,  #pattern wanet: identity_grid iad: modelG issba: secret
            weight,  #weight wanet: noise_grid iad: modelM issba: secret_size
            0,  # cifar10/tiny: 0 ; gtsrb: 2
            0)
    elif attack_type == 'feature':
        test_bad_dataset = torch.load('experiments/feature_bd/test_1.pth', map_location='cpu')
        #print(type(test_bad_dataset[0][0]))
        poisoned_set = list(zip(test_bad_dataset[0], test_bad_dataset[1]))
        indice = torch.load('experiments/feature_bd/target0_indice.pth')
        poisoned_set = [data for i, data in enumerate(poisoned_set) if i not in indice]
    else:
        print("Wrong Attack Type!")

    return poisoned_set


if __name__ == '__main__':
    create_val_folder()
