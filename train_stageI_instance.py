# -*- coding: utf-8 -*-
from __future__  import print_function,division

import os
import time
import argparse
import torch
import torch.nn as nn 
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets,transforms
from load_text import load_data
from rank_loss import ImageSelector,TextSelector
from loss import TripletLoss

from model import Merge_image_text
from test_acc import test
import utils

parser = argparse.ArgumentParser(description='Training arguments')
parser.add_argument('--save_path',type=str,default='./flickr30k-56-stage1')  # the word num of a sequence is no more than 56
parser.add_argument('--datasets',type=str,default='/data/reid/flickr30k/Dual-path/')
parser.add_argument('--batch_size',type=int,default=32,help='batch_size') 
parser.add_argument('--learning_rate',type=float,default=0.001,help = 'FC parms learning rate')
parser.add_argument('--epochs',type=int,default=120,help='The number of epochs to train')
parser.add_argument('--stage',type=str,default='I',choices=['I','II'],help='which stage is on')

arg = parser.parse_args()

torch.manual_seed(1)
torch.cuda.manual_seed_all(1)

#Make saving directory
save_dir_path = arg.save_path
os.makedirs(save_dir_path,exist_ok=True)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

# -------------------------------------Train Function--------------------------------------
def train(model,criterion,optimizer,scheduler,dataloder, image_dataset, text_loader,num_epochs,device,stage):
    start_time = time.time()

    # Logger instance
    logger = utils.Logger(save_dir_path)
    logger.info('-'*10)
    logger.info(vars(arg))
    logger.info('Stage: '+stage)

    print("################################### Train stage I ######################################")
    for epoch in range(num_epochs):
        logger.info('Epoch {}/{}'.format(epoch+1,num_epochs))

        model.train()
        scheduler.step()

        ##Training 
        running_loss = 0.0
        running_text_loss = 0.0
        batch_num = 0
        img_cor = torch.zeros(1).squeeze().cuda()
        total = torch.zeros(1).squeeze().cuda()
        txt_cor = torch.zeros(1).squeeze().cuda()
        txt_total = torch.zeros(1).squeeze().cuda()
    
        for (inputs,labels),(text_inputs,text_labels) in zip(dataloder,text_loader):
            batch_num += 1
            inputs = inputs.to(device)
            labels = labels.to(device)
            text_inputs = text_inputs.to(device)
            text_labels = text_labels.to(device,dtype=torch.int64)

            outputs,text_outs = model(inputs,text_inputs)

            ###Intance loss
            loss = criterion(outputs,labels)
            text_loss = criterion(text_outs,text_labels)
            optimizer.zero_grad()
            loss.backward()
            text_loss.backward()
            optimizer.step()
            running_loss += loss.item()*inputs.size(0)
            running_text_loss += text_loss.item()*text_inputs.size(0)

            #Accurate
            img_pre = torch.argmax(outputs,1)
            img_cor += (img_pre == labels).sum().float()
            total += len(labels)
            #print(img_pre, labels)
            
            txt_pre = torch.argmax(text_outs,1)
            txt_cor += (txt_pre == text_labels).sum().float()
            txt_total += len(text_labels)
            #print(txt_pre, text_labels)

            if batch_num % 10 == 0:
                logger.info('Train image epoch : {} [{}/{}]\t Image Loss:{:.6f}\t || Text Loss:{:.6f}'.format(epoch+1,batch_num*len(inputs),len(dataloder.dataset.imgs),
                running_loss/(batch_num*arg.batch_size),running_text_loss/(batch_num*arg.batch_size)))
                
        logger.info("Img_acc: {:.2f} \t Text_acc: {:.2f}".format((img_cor/total).cpu().detach().data.numpy(),(txt_cor/txt_total).cpu().detach().data.numpy()))
        logger.info('Epoch {}:Done!!!'.format(epoch+1))

        loss_val_runing = 0.0
        loss_val_runing_text = 0.0

        img_cor_val = torch.zeros(1).squeeze().cuda()
        txt_cor_val = torch.zeros(1).squeeze().cuda()
        if (epoch+1) % 10 == 0 or epoch+1 == num_epochs:
            ##Testing / Vlidating
            torch.cuda.empty_cache()
            model.mode = 'test'
            CMC,mAP = test(model,arg.datasets, 128)
            logger.info('Testing: Top1:{:.2f}% Top5:{:.2f}% Top10:{:.2f}% mAP:{:.2f}%'.format(CMC[0],CMC[4],CMC[9],mAP))
            print("=======================================================")
            model.mode = 'train'
        logger.info('-'*10)
    time_elapsed = time.time() - start_time
    logger.info('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60,time_elapsed%60
    ))
    #Save final model weithts
    utils.save_network(model,save_dir_path,'final')

if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    ##It is a classifier task for image 
    train_dataloder = utils.getDataloader(arg.datasets,arg.batch_size,'train',shuffle=True,augment=True)
    image_dataset = train_dataloder.dataset
    train_dataloder_text = load_data(arg.datasets,'train',arg.batch_size, image_dataset)

    model = Merge_image_text(num_class=len(train_dataloder.dataset.classes),mode = 'train')   #Stage II ,change to 'test',Stage I:'train'
    if torch.cuda.device_count() > 1:
        print("Let's use",torch.cuda.device_count(), "GPUs")
        model = nn.DataParallel(model,device_ids =[0,1,2])

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    ignore_param = list(map(id,model.image_feature.backbone.parameters()))
    base_param = filter(lambda p: id(p) not in ignore_param,model.parameters())
    optimizer = optim.SGD([
        {'params':base_param,'lr':0.001},
        {'params':model.image_feature.backbone.parameters(),'lr':0.00001},
    ],momentum=0.9,weight_decay = 5e-4,nesterov=True)

    scheduler = lr_scheduler.StepLR(optimizer,step_size=20,gamma=0.1)

    #---------------------Start training----------------------
    #Stage I
    train(model, criterion, optimizer, scheduler, train_dataloder, image_dataset, train_dataloder_text,arg.epochs,device,'I')
    ##Stage II
    # model.load_state_dict(torch.load('./model_test_save/data_test/net_final.pth'))
    # train_rank(model,criterion,optimizer,scheduler,train_dataloder,train_dataloder_text,arg.epochs,device,'II')
    torch.cuda.empty_cache()
