from collections import namedtuple
import os
import glob
import json
import shutil
import re
import sys
from importlib import import_module
from pathlib import Path

from utils import *

import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from datetime import datetime
from tqdm import tqdm

cur_date = datetime.today().strftime("%y%m%d")

def save_checkpoint(epoch, model, loss, f1, optimizer, saved_dir, scheduler, file_name):
    check_point = {'epoch': epoch,
                   'net': model.state_dict(),
                   'optimizer_state_dict': optimizer.state_dict(),
                   'loss': loss,
                   'f1' : f1}
    if scheduler:
        check_point['scheduler_state_dict'] = scheduler.state_dict()
    output_path = os.path.join(saved_dir, file_name)
    torch.save(check_point, output_path)


def load_checkpoint(checkpoint_path, model, optimizer, scheduler):
    # load model if resume_from is set
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['net'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch']
    start_loss = checkpoint['loss']
    prv_best_f1 = checkpoint['f1']

    return model, optimizer, scheduler, start_epoch, start_loss, prv_best_f1


def validation(epoch, num_epochs, model, val_loader, criterion, device):
    model.eval()

    sum_loss = 0
    running_loss = 0
    sum_f1 = 0
    running_f1 = 0
    total = len(val_loader)
    pbar = tqdm(enumerate(val_loader), total = total)

    with torch.no_grad():
        for step, input in pbar:
            img = input['img'].to(device)
            csv = input['csv'].to(device)
            label = input['label'].to(device)

            output = model(img, csv)
            loss = criterion(output, label)

            sum_loss += loss.item()
            running_loss = sum_loss / (step + 1)

            f1 = get_f1(label, output)
            sum_f1 += f1
            running_f1 = sum_f1 / (step + 1)

            description =  f'Epoch [{epoch+1}/{num_epochs}], Step [{step+1}/{total}]: ' 
            description += f'running Loss: {round(running_loss,4)}, f1: {round(running_f1,4)}'
            pbar.set_description(description)
    
    return running_loss, running_f1





def train(num_epochs, model, train_loader, val_loader, criterion, optimizer,
          save_dir, resume_from, checkpoint_path, device, scheduler = None, fp16 = False):
    print("Start Training...")
    start_epoch = 0
    
    best_loss = sys.maxsize
    best_f1 = 0
    
    if resume_from:
        model, optimizer, scheduler, start_epoch, best_loss, best_f1 = load_checkpoint(checkpoint_path, model, optimizer, scheduler)

    if fp16:
        print("Mixed Precision Is Applied")
        scaler = GradScaler()
    
    for epoch in range(start_epoch, num_epochs):
        model.train()

        sum_loss = None
        running_loss = None
        sum_f1 = None
        running_f1 = None
        total = len(train_loader)
        pbar = tqdm(enumerate(train_loader), total = total)
        
        for step, input in pbar:
            img = input['img'].to(device)
            csv = input['csv'].to(device)
            label = input['label'].to(device)

            optimizer.zero_grad()
            
            if fp16:
                with autocast():
                    output = model(img, csv)
                    loss = criterion(output, label)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                output = model(img, csv)
                loss = criterion(output, label)
                loss.backward()
                optimizer.step()
            
            if running_loss is None:
                sum_loss = loss.item()
                running_loss = loss.item()
            else:
                sum_loss += loss.item()
                running_loss = sum_loss / (step + 1)
            
            f1 = get_f1(label, output)

            if running_f1 is None:
                sum_f1 = f1
                running_f1 = f1
            else:
                sum_f1 += f1
                running_f1 = sum_f1 / (step + 1)
            
            description =  f'Epoch [{epoch+1}/{num_epochs}], Step [{step+1}/{total}]: ' 
            description += f'running Loss: {round(running_loss,4)}, f1: {round(running_f1,4)}'
            pbar.set_description(description)
        
        val_loss, val_f1 = validation(epoch, num_epochs, model, val_loader, criterion, device)

        if val_f1 > best_f1:
            print(f"Best performance at epoch: {epoch + 1}")
            print(f"Save model in {save_dir}")
            best_f1 = val_f1
            save_checkpoint(epoch, model, loss, val_f1, optimizer, save_dir, scheduler, file_name = f"{model.model_name}_{round(best_f1, 3)}_{cur_date}.pt")

        
        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_f1)
            else:
                scheduler.step()
            
def main():
    args = arg_parse()
    with open(args.cfg, 'r') as f:
        cfgs = json.load(f, object_hook=lambda d: namedtuple('x', d.keys())(*d.values()))

    fix_seed(cfgs.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_augmentation_module = getattr(import_module("augmentation"), cfgs.train_augmentation)
    train_augmentation = train_augmentation_module().transform
    
    val_augmentation_module = getattr(import_module("augmentation"), cfgs.val_augmentation)
    val_augmentation = val_augmentation_module().transform

    # dataset & data loader
    train_dataset_module = getattr(import_module("dataset"), cfgs.train_dataset.name)
    train_dataset = train_dataset_module(cfgs.data_root, **cfgs.train_dataset.args._asdict(), transform = train_augmentation)
    train_dataloader = DataLoader(train_dataset, **cfgs.train_dataloader.args._asdict())
    
    val_dataset_module = getattr(import_module("dataset"), cfgs.val_dataset.name)
    val_dataset = val_dataset_module(cfgs.data_root, **cfgs.val_dataset.args._asdict(), transform = val_augmentation)
    val_dataloader = DataLoader(val_dataset, **cfgs.val_dataloader.args._asdict())

    # model
    img_encoder_module = getattr(import_module("model"), cfgs.img_encoder)
    img_encoder = img_encoder_module()
    csv_encoder_module = getattr(import_module("model"), cfgs.csv_encoder)
    csv_encoder = csv_encoder_module()
    model_module = getattr(import_module("model"), cfgs.model)
    model = model_module(img_encoder, csv_encoder)

    criterion_module = getattr(import_module("criterion"), cfgs.criterion.name)
    criterion = criterion_module(**cfgs.criterion.args._asdict())

    optimizer_module = getattr(import_module("torch.optim"), cfgs.optimizer.name)
    optimizer = optimizer_module(model.parameters(), **cfgs.optimizer.args._asdict())

    try:
        if hasattr(import_module("scheduler"), cfgs.scheduler.name):
            scheduler_module = getattr(import_module("scheduler"), cfgs.scheduler.name)
            scheduler = scheduler_module(optimizer, **cfgs.scheduler.args._asdict())
        else:
            scheduler_module = getattr(import_module("torch.optim.lr_scheduler"), cfgs.scheduler.name)
            scheduler = scheduler_module(optimizer, **cfgs.scheduler.args._asdict())
    except AttributeError :
            print('There is no Scheduler!')
            scheduler = None
    
    save_dir = cfgs.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    shutil.copy(args.cfg, f"{save_dir}/config.json")

    train_args = {
        'num_epochs': cfgs.num_epochs, 
        'model': model, 
        'train_loader': train_dataloader, 
        'val_loader': val_dataloader, 
        'criterion': criterion, 
        'optimizer': optimizer, 
        'save_dir': save_dir, 
        'resume_from': cfgs.resume_from,
        'checkpoint_path': cfgs.checkpoint_path,
        'device': device,
        'scheduler': scheduler,
        'fp16': cfgs.fp16
    }
    train(**train_args)


if __name__ == "__main__":
    main()




