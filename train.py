import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import warnings
from time import time
from pathlib import Path
from config import config
import torch.optim as optim
from tqdm.auto import tqdm
from dataset import PreGeneratedRadonDataset, get_collate_fn
import logging
import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=UserWarning, module='skimage')

def setup_ddp():
    """Initializes the distributed process group."""
    dist.init_process_group(backend="nccl")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    return rank, local_rank, device

def cleanup_ddp():
    """Destroys the distributed process group."""
    dist.destroy_process_group()

class Trainer:
    def __init__(self, model, train_dataloader, val_dataloader, device, rank, optimizer, num_epochs=1, save_dir=None, save_interval=1, logger=None, logfile=None):
        self.model = model  # Note: model is already DDP-wrapped and on the correct device
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.rank = rank
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        # self.model.to(self.device) # This is done before DDP wrapping
        self.save_dir = save_dir
        self.save_interval = save_interval
        self.logger = logger
        self.logfile = logfile #csv file to log losses


    def train(self):
        for epoch in range(self.num_epochs):
            self.train_dataloader.sampler.set_epoch(epoch)
            train_loss = self.train_one_epoch(epoch)
            val_loss = self.validate_one_epoch(epoch)

            if self.rank == 0:
                if self.logger:
                    self.logger.info(f"Epoch {epoch+1}/{self.num_epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
                else:
                    print(f"Epoch {epoch+1}/{self.num_epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

                # Log to CSV file if specified
                if self.logfile:
                    with open(self.logfile, 'a') as f:
                        if epoch == 0:
                            f.write("epoch,train_loss,val_loss\n")
                        f.write(f"{epoch+1},{train_loss},{val_loss}\n")

                if self.save_dir and (epoch + 1) % self.save_interval == 0:
                    save_path = Path(self.save_dir) / f"model_epoch_{epoch+1}.pt"
                    # Save the unwrapped model's state_dict
                    torch.save(self.model.module.state_dict(), save_path)
                    if self.logger:
                        self.logger.info(f"Saved model checkpoint to {save_path}")
                    else:
                        print(f"Saved model checkpoint to {save_path}")

    def train_one_step(self, batch):
        pixel_values = batch['pixel_values'].to(self.device)
        pixel_mask = batch['pixel_mask'].to(self.device)
        labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch['labels']]

        outputs = self.model(
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            labels=labels
        )
        
        loss = outputs.loss
        self.optimizer.zero_grad()
        loss.backward() # DDP handles gradient synchronization
        self.optimizer.step()
        return loss.item()

    def validate_one_step(self, batch):
        pixel_values = batch['pixel_values'].to(self.device)
        pixel_mask = batch['pixel_mask'].to(self.device)
        labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch['labels']]

        outputs = self.model(
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            labels=labels
        )

        loss = outputs.loss
        return loss.item()

    def train_one_epoch(self,epoch):
        self.model.train()
        train_loss = 0.0
        
        # Only show progress bar on rank 0
        train_loop = self.train_dataloader
        if self.rank == 0:
            train_loop = tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}/{self.num_epochs} [Train]")
        
        for batch in train_loop:
            loss = self.train_one_step(batch)
            train_loss += loss
            
            if self.rank == 0:
                train_loop.set_postfix(loss=loss)

        # Average loss for this process
        avg_train_loss = train_loss / len(self.train_dataloader)
        
        # Aggregate losses from all processes
        loss_tensor = torch.tensor(avg_train_loss, device=self.device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        
        # Get the global average by dividing by world size
        global_avg_train_loss = loss_tensor.item() / dist.get_world_size()
        
        return global_avg_train_loss

    def validate_one_epoch(self,epoch):
        self.model.eval()
        val_loss = 0.0
        
        # Only show progress bar on rank 0 (optional, but consistent)
        val_loop = self.val_dataloader
        if self.rank == 0:
            val_loop = tqdm(self.val_dataloader, desc=f"Epoch {epoch+1}/{self.num_epochs} [Validate]")

        with torch.no_grad():
            for batch in val_loop:
                loss = self.validate_one_step(batch)
                val_loss += loss
        
        # Average loss for this process
        avg_val_loss = val_loss / len(self.val_dataloader)
        
        # Aggregate losses from all processes
        loss_tensor = torch.tensor(avg_val_loss, device=self.device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)

        # Get the global average by dividing by world size
        global_avg_val_loss = loss_tensor.item() / dist.get_world_size()

        return global_avg_val_loss

def main():
    rank, local_rank, device = setup_ddp()

    if rank == 0:
        print(f"Loading processor: {config.PROCESSOR_NAME}")
    processor = DetrImageProcessor.from_pretrained(config.PROCESSOR_NAME)

    collate_fn = get_collate_fn(processor)

    if rank == 0:
        print("Creating training dataset...")
    train_dir = config.DATA_DIR / "train"
    train_dataset = PreGeneratedRadonDataset(
        data_dir=train_dir,
        processor=processor,
        box_width=config.BOX_WIDTH_PX,  
        box_height=config.BOX_HEIGHT_PX,
        class_id=config.CLASS_ID
    )

    if rank == 0:
        print("Creating validation dataset...")
    val_dir = config.DATA_DIR / "val"
    val_dataset = PreGeneratedRadonDataset(
        data_dir=val_dir,
        processor=processor,
        box_width=config.BOX_WIDTH_PX,  
        box_height=config.BOX_HEIGHT_PX,
        class_id=config.CLASS_ID
    )

    if rank == 0:
        print("Creating distributed samplers and dataloaders...")
    
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)

    train_dataloader = DataLoader(
        train_dataset, 
        collate_fn=collate_fn, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False,  # Sampler handles shuffling
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        sampler=train_sampler
    )
    val_dataloader = DataLoader(
        val_dataset, 
        collate_fn=collate_fn, 
        batch_size=config.BATCH_SIZE,
        shuffle=False, # Sampler for val
        num_workers=config.NUM_WORKERS, 
        pin_memory=True,
        sampler=val_sampler
    )

    if rank == 0:
        print("\nData loading pipeline successfully created.")

    id2label = {0: 'peak'} 
    NUM_LABELS = len(id2label)

    # Hyperparameters
    LR = 1e-4
    LR_BACKBONE = 1e-5
    WEIGHT_DECAY = 1e-4
    NUM_EPOCHS = 500
    PRETRAINED_MODEL_PATH = config.PROCESSOR_NAME

    model = DetrForObjectDetection.from_pretrained(
        PRETRAINED_MODEL_PATH,
        revision="no_timm",
        num_labels=NUM_LABELS,
        ignore_mismatched_sizes=True 
    )
    
    #move model to device before wrapping
    model.to(device)
    
    model = DDP(model, device_ids=[local_rank])
    
    param_dicts = [
        {
            "params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]
        },
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": LR_BACKBONE,
        },
    ]
    optimizer = optim.AdamW(param_dicts, lr=LR, weight_decay=WEIGHT_DECAY)

    # Only create save_dir on rank 0
    save_dir = "model_checkpoints"
    if rank == 0:
        Path(save_dir).mkdir(exist_ok=True)
        
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        device=device,
        rank=rank,
        optimizer=optimizer,
        num_epochs=NUM_EPOCHS,
        save_dir=save_dir,
        save_interval=10,
        logger=logger if rank == 0 else None, # Only log on rank 0
        logfile="training_logs.csv" if rank == 0 else None # Only write logfile on rank 0
    )
    
    if rank == 0:
        print("Starting DDP training...")
        
    trainer.train()

    if rank == 0:
        print("Training complete.")
        
    cleanup_ddp()


if __name__ == "__main__":
    main()