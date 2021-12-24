
import torch
import torch.utils.data as data

from itertools import chain
from pathlib import Path
import glob
import os
from munch import Munch
import numpy as np
import random
from torch.utils.data.sampler import WeightedRandomSampler
from PIL import Image
from torchvision import transforms

def listdir(dname):
    fnames = list(chain(*[list(Path(dname).rglob('*.' + ext))
                          for ext in ['png', 'jpg', 'jpeg', 'JPG']]))
    return fnames


class InputFetcher:
    def __init__(self, loader, loader_ref=None, latent_dim=16, mode=''):
        self.loader = loader
        self.loader_ref = loader_ref
        self.latent_dim = latent_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mode = mode

    def _fetch_inputs(self):
        try:
            x, y,x_ref, x_ref2, y_ref = next(self.iter)
        except (AttributeError, StopIteration):
            self.iter = iter(self.loader)
            x, y,x_ref, x_ref2, y_ref = next(self.iter)
        return x, y,x_ref, x_ref2, y_ref

    def _fetch_refs(self):
        try:
            x, x2, y = next(self.iter_ref)
        except (AttributeError, StopIteration):
            self.iter_ref = iter(self.loader_ref)
            x, x2, y = next(self.iter_ref)
        return x, x2, y

    def __next__(self):
        x, y,x_ref, x_ref2, y_ref = self._fetch_inputs()
        if self.mode == 'train':
            z_trg = torch.randn(x.size(0), self.latent_dim)
            z_trg2 = torch.randn(x.size(0), self.latent_dim)
            inputs = Munch(x_src=x, y_src=y, y_ref=y_ref,
                           x_ref=x_ref, x_ref2=x_ref2,
                           z_trg=z_trg, z_trg2=z_trg2)
        elif self.mode == 'val':
            X, Y,x_ref,_, y_ref = self._fetch_inputs()
            inputs = Munch(x_src=X, y_src=Y,
                           x_ref=x_ref, y_ref=y_ref)
        elif self.mode == 'test':
            inputs = Munch(x=x, y=y)
        else:
            raise NotImplementedError

        return Munch({k: v.to(self.device)
                      for k, v in inputs.items()})


class CasiaTrain(data.Dataset):
    def __init__(self,root,image_size,transform=None):
        self.root=root
        self.subjects=62
        self.angles=['000','018','036','054','072','090','108','126','144','162','180']
        self.states=['nm-01','nm-02','nm-03','nm-04','nm-05','nm-06','bg-01','bg-02','cl-01','cl-02']
        self.n_angles=len(self.angles)
        self.n_states=len(self.states)
        self.image_size=image_size
        if transform is None:
            self.transform=transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.ToTensor()
            ])
        self.dataset,self.dataset_ref=self.make_dataset()
    
    def make_dataset(self):
        data=[]
        data_ref={}
        for i in range(1,self.subjects):
            data_ref[i]=[]
            for j in range(0,self.n_angles):
                id='%03d'% i
                image_list=glob.glob(os.path.join(self.root,id,'*','*-'+self.angles[j]+'*'))
                id_list=[i]*len(image_list)
                label_list=[j]*len(image_list)
                data.extend(list(zip(image_list,id_list,label_list)))
                data_ref[i].extend(list(zip(image_list,label_list)))
        return data,data_ref

    def __getitem__(self,idx):
        x_src_name=self.dataset[idx][0]
        id=self.dataset[idx][1]
        img=Image.open(x_src_name).convert('RGB')
        x_src_img=self.transform(img)
        while True:
            x_ref1,x_ref2=random.sample(self.dataset_ref[id],2)
            if x_ref1[0]==x_src_name or x_ref2[0]==x_src_name:
                continue
            if not x_ref1[1]==x_ref2[1]:
                continue
            break
        x_src_label=torch.tensor(self.dataset[idx][2],dtype=torch.long)
        x_ref1_img=Image.open(x_ref1[0]).convert('RGB')
        x_ref2_img=Image.open(x_ref2[0]).convert('RGB')
        label=torch.tensor(x_ref1[1],dtype=torch.long)
        #print(x_src_name,x_src_label,x_ref1[0],x_ref2[0],label)
        x_ref1_img=self.transform(x_ref1_img)
        x_ref2_img=self.transform(x_ref2_img)
        return x_src_img,x_src_label,x_ref1_img,x_ref2_img,label

    def __len__(self):
        return len(self.dataset)

class CasiaVal(data.Dataset):
    def __init__(self,root,image_size,transform=None):
        self.root=root
        self.subjects=62
        self.angles=['000','018','036','054','072','090','108','126','144','162','180']
        self.states=['nm-01','nm-02','nm-03','nm-04','nm-05','nm-06','bg-01','bg-02','cl-01','cl-02']
        self.n_angles=len(self.angles)
        self.n_states=len(self.states)
        self.image_size=image_size
        if transform is None:
            self.transform=transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.ToTensor()
            ])
        self.dataset,self.dataset_ref=self.make_dataset()
    
    def make_dataset(self):
        data=[]
        data_ref={}
        for i in range(63,self.subjects+63):
            data_ref[i]=[]
            for j in range(0,self.n_angles):
                id='%03d'% i
                image_list=glob.glob(os.path.join(self.root,id,'*','*-'+self.angles[j]+'*'))
                id_list=[i]*len(image_list)
                label_list=[j]*len(image_list)
                data.extend(list(zip(image_list,id_list,label_list)))
                data_ref[i].extend(list(zip(image_list,label_list)))
        return data,data_ref

    def __getitem__(self,idx):
        x_src_name=self.dataset[idx][0]
        id=self.dataset[idx][1]
        img=Image.open(x_src_name).convert('RGB')
        x_src_img=self.transform(img)
        while True:
            x_ref1,x_ref2=random.sample(self.dataset_ref[id],2)
            if x_ref1[0]==x_src_name or x_ref2[0]==x_src_name:
                continue
            if not x_ref1[1]==x_ref2[1]:
                continue
            break
        x_src_label=torch.tensor(self.dataset[idx][2],dtype=torch.long)
        x_ref1_img=Image.open(x_ref1[0]).convert('RGB')
        x_ref2_img=Image.open(x_ref2[0]).convert('RGB')
        label=torch.tensor(x_ref1[1],dtype=torch.long)
        #print(x_src_name,x_src_label,x_ref1[0],x_ref2[0],label)
        x_ref1_img=self.transform(x_ref1_img)
        x_ref2_img=self.transform(x_ref2_img)
        return x_src_img,x_src_label,x_ref1_img,x_ref2_img,label

    def __len__(self):
        return 50

class CasiaTest(data.Dataset):
    def __init__(self,root,image_size,transform=None):
        self.root=root
        self.subjects=62
        self.angles=['000','018','036','054','072','090','108','126','144','162','180']
        self.states=['nm-01','nm-02','nm-03','nm-04','nm-05','nm-06','bg-01','bg-02','cl-01','cl-02']
        self.n_angles=len(self.angles)
        self.n_states=len(self.states)
        self.image_size=image_size
        if transform is None:
            self.transform=transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.ToTensor()
            ])
        self.dataset=self.make_dataset()
    
    def make_dataset(self):
        data=[]
        data_ref={}
        for i in range(63,self.subjects+63):
            data_ref[i]=[]
            for j in range(0,self.n_angles):
                for k in range(0,self.n_states):
                    id='%03d'% i
                    image_list=glob.glob(os.path.join(self.root,id,self.states[k],'*-'+self.angles[j]+'*'))
                    for l in range(0,self.n_angles):
                        ref_img=glob.glob(os.path.join(self.root,id,self.states[k],'*-'+self.angles[l]+'*'))
                        data.extend([image_list,id,k,j,ref_img,l])
        return data

    def __getitem__(self,idx):
        x_src_name=self.dataset[idx][0]
        id=self.dataset[idx][1]
        img=Image.open(x_src_name).convert('RGB')
        x_src_img=self.transform(img)
        x_src_cond=torch.tensor(self.dataset[idx][2],dtype=torch.long)
        x_src_angle=torch.tensor(self.dataset[idx][3],dtype=torch.long)
        x_ref_name=self.dataset[idx][4]
        img=Image.open(x_ref_name).convert('RGB')
        x_ref_img=self.transform(img)
        y_ref=torch.tensor(self.dataset[idx][5],dtype=torch.int)
        return x_src_img,id,k,j,x_ref_img,y_ref

    def __len__(self):
        return len(self.dataset)



def get_train_loader(root,img_size,batch_size,num_workers):
    print("Preparing Dataloader for training ")
    dataset=CasiaTrain(root,img_size)
    return data.DataLoader(dataset=dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers,pin_memory=True)


def get_val_loader(root,img_size,batch_size,num_workers,shuffle=True):
    print("Preparing Dataloader for Validation Sampling ")
    dataset=CasiaVal(root,img_size)
    return data.DataLoader(dataset=dataset,batch_size=batch_size,shuffle=shuffle,num_workers=num_workers,pin_memory=True)

def get_test_loader(root,img_size,batch_size,num_workers,shuffle=False):
    print("Preparing Dataloader for Testing generation ")
    dataset=CasiaTest(root,img_size)
    return data.DataLoader(dataset=dataset,batch_size=batch_size,shuffle=shuffle,num_workers=num_workers,pin_memory=True)
    

casia=CasiaTrain('/mnt/sda2/intern/GEI',64)

#print(casia.dataset_ref[1])
loader=data.DataLoader(casia,batch_size=2)
a,b,c,d,e=next(iter(loader))
