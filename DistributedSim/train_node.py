import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
import numpy as np
import zipfile

import os
import copy

from .sim_config import *
from .gradient_strategy.gradient_strategy import *
from .wandb_logger import *
from .gradient_strategy.communicate import *

from .dataset.nanogpt.dataset import get_dataset

class TrainNode:
    '''
    Single node of distributed training process. Should be the same regardless of rank topology/architecture.
    '''
    def __init__(self, 
                 config: SimConfig,
                 device: torch.device,
                 rank: int):
        self.config = config
        self.device = device
        self.rank = rank

        torch.manual_seed(self.config.seed)
        torch.cuda.manual_seed(self.config.seed)

        self.get_datasets()

        self.model = self.config.model_class(self.config.model_config).to(self.device)
    
        # TODO: Reimplement this
        # print(f"model parameter count: ", self.model.get_num_params() / 1e6)

        ## Ensure all process models share the same params
        if self.config.num_nodes > 1:
            for _, param in self.model.named_parameters():
                broadcast(param.data, src=0)

        self.local_step = 0
        self.max_steps = len(self.train_dataloader) * self.config.num_epochs
        if self.config.gradient_config.max_local_steps:
            self.max_steps = min(self.max_steps, 
                                 self.config.gradient_config.max_local_steps)

            self.config.gradient_config.max_local_steps = self.max_steps

        if self.rank == 0:
            self.logger = WandbLogger(rank=self.rank, 
                                      device=self.device, 
                                      config=self.config, 
                                      model=self.model, 
                                      max_steps=self.max_steps)

        self.gradient_strategy = self.config.gradient_class(self.rank, 
                                                            self.model, 
                                                            self.config,
                                                            self.logger if self.rank == 0 else None)

        self.epoch = 0
        
        # Attempt to load checkpoint before starting training
        self._load_checkpoint()

    def get_datasets(self):
        """
        Imports datasets.
        Currently ignores dataset_proportion parameters to favour generality.
        """
        # dataset_id = self.config.dataset_name.split('_')[0]

        train_start = (1 - self.config.dataset_config.val_proportion) * self.rank / self.config.num_nodes * self.config.dataset_config.dataset_proportion
        train_end = (1 - self.config.dataset_config.val_proportion) * (self.rank + 1) / self.config.num_nodes * self.config.dataset_config.dataset_proportion
        val_start = (1 - self.config.dataset_config.val_proportion)
        val_end = 1.0


        self.train_dataset = self.config.dataset_config.dataset_load_fn(self.config.dataset_config, device=self.device, start_pc=train_start, end_pc=train_end)
        self.val_dataset = self.config.dataset_config.dataset_load_fn(self.config.dataset_config, device=self.device, start_pc=val_start, end_pc=val_end)

        ## Build Dataloaders
        self.train_dataloader = DataLoader(self.train_dataset, 
                          batch_size=self.config.minibatch_size,
                          shuffle=True)

        self.val_dataloader = DataLoader(self.val_dataset, 
                          batch_size=self.config.minibatch_size,
                          shuffle=True)

        self.train_data_iter = iter(self.train_dataloader)
        self.val_data_iter = iter(self.val_dataloader)

    def _save_checkpoint(self):
        print(self.config.save_dir, self.config.wandb_project, self.config.wandb_name, self.rank)
        save_path_dir = os.path.join(self.config.save_dir,
                                 self.config.wandb_project if self.config.wandb_project else 'unnamed',
                                 self.config.wandb_name if self.config.wandb_name else 'unnamed',
                                 str(self.rank))
        if not os.path.exists(save_path_dir):
            os.makedirs(save_path_dir, exist_ok=True)

        filename = f"{self.local_step}.pt"
        full_save_path = os.path.join(save_path_dir, filename)

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.gradient_strategy.optim.state_dict(),
            'local_step': self.local_step,
            'epoch': self.epoch,
            'rng_state': torch.get_rng_state(),
        }
        if self.gradient_strategy.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.gradient_strategy.scheduler.state_dict()
        
        if self.device.type == 'cuda':
            checkpoint['cuda_rng_state'] = torch.cuda.get_rng_state()

        try:
            torch.save(checkpoint, full_save_path)
            print(f"Rank {self.rank} saved checkpoint to {full_save_path} at step {self.local_step}")
            self._delete_other_checkpoints(save_path_dir, full_save_path)
        except OSError as e:
            print(f"Rank {self.rank}: Failed to save checkpoint {full_save_path} due to OSError: {e}. Attempting to delete oldest checkpoint and retry.")
            
            oldest_step = float('inf')
            oldest_checkpoint_file = None
            # Ensure save_path_dir exists before listing its contents, though it should have been created.
            if os.path.exists(save_path_dir):
                for f_name in os.listdir(save_path_dir):
                    if f_name.endswith('.pt'):
                        try:
                            # Checkpoints are named as {step_num}.pt
                            step_num = int(f_name.split('.')[0])
                            if step_num < oldest_step:
                                oldest_step = step_num
                                oldest_checkpoint_file = f_name
                        except ValueError:
                            # Skip files not matching the expected N.pt pattern
                            continue
            
            if oldest_checkpoint_file:
                oldest_checkpoint_path = os.path.join(save_path_dir, oldest_checkpoint_file)
                try:
                    os.remove(oldest_checkpoint_path)
                    print(f"Rank {self.rank}: Deleted oldest checkpoint {oldest_checkpoint_path} to free space.")
                    
                    # Retry saving the current checkpoint
                    try:
                        torch.save(checkpoint, full_save_path)
                        print(f"Rank {self.rank}: Successfully saved checkpoint {full_save_path} after deleting oldest.")
                        self._delete_other_checkpoints(save_path_dir, full_save_path)
                    except OSError as e2:
                        print(f"Rank {self.rank}: Still failed to save checkpoint {full_save_path} after deleting oldest: {e2}. Giving up.")
                        raise # Re-raise the second error, as we couldn't save even after cleanup
                except OSError as del_e:
                    print(f"Rank {self.rank}: Failed to delete oldest checkpoint {oldest_checkpoint_path}: {del_e}. Original save error will be raised.")
                    raise e # Re-raise the original save error, as cleanup failed
            else:
                print(f"Rank {self.rank}: No old checkpoints found to delete in {save_path_dir}. Original save error will be raised.")
                raise e # Re-raise the original save error, as no space could be freed

    def _delete_other_checkpoints(self, save_path_dir: str, current_checkpoint_full_path: str):
        if not os.path.exists(save_path_dir):
            return

        current_checkpoint_filename = os.path.basename(current_checkpoint_full_path)
        deleted_count = 0
        for f_name in os.listdir(save_path_dir):
            if f_name.endswith('.pt') and f_name != current_checkpoint_filename:
                try:
                    file_to_delete = os.path.join(save_path_dir, f_name)
                    os.remove(file_to_delete)
                    # print(f"Rank {self.rank}: Deleted old checkpoint {file_to_delete}")
                    deleted_count += 1
                except OSError as del_e:
                    print(f"Rank {self.rank}: Warning - Failed to delete old checkpoint {file_to_delete}: {del_e}")
        if deleted_count > 0:
            print(f"Rank {self.rank}: Deleted {deleted_count} other checkpoint(s) in {save_path_dir}.")

    def _get_batch(self, eval=False):
        if not eval or self.val_data_iter is None:
            try:
                batch = next(self.train_data_iter)
            except StopIteration:
                self.epoch += 1
                self.train_data_iter = iter(self.train_dataloader)
                batch = next(self.train_data_iter)
        else:
            try:
                batch = next(self.val_data_iter)
            except StopIteration:
                self.val_data_iter = iter(self.val_dataloader)
                batch = next(self.val_data_iter)

        # print(batch)
        if isinstance(batch, tuple) or isinstance(batch, list):
            batch = tuple(x.to(self.device) for x in batch)
        elif isinstance(batch, dict):
            batch = {k: v.to(self.device) for k, v in batch.items()}
        else:
            batch = batch.to(self.device)

        return batch

    def _train_step(self):
        self.gradient_strategy.zero_grad()
        
        for i in range(self.config.batch_size // self.config.minibatch_size):
            minibatch = self._get_batch()

            ## TODO: Do we want this?
            if self.config.autocast:
                with torch.autocast(device_type=self.config.device_type, dtype=torch.bfloat16):
                    loss = self.model(minibatch)
            else:
                loss = self.model(minibatch)

            loss.backward()

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.grad /= (self.config.batch_size / self.config.minibatch_size)
        
        self.gradient_strategy.step()

        if self.rank == 0:
            self.logger.log_train(loss=loss.item())

        if self.config.checkpoint_interval and self.local_step % self.config.checkpoint_interval == 0:
            self._save_checkpoint()

    def _evaluate(self):
        model_clone = self.config.model_class(self.config.model_config).to(self.device)
        model_clone.load_state_dict(copy.deepcopy(self.model.state_dict()))

        for name, param in model_clone.named_parameters():
            all_reduce(param.data, op=dist.ReduceOp.SUM)
            param.data = param.data / dist.get_world_size()

        if self.rank == 0:
            # For rank 0, we will calculate the local loss
            this_model = self.model

        if self.rank == 1:
            # For rank 1, we want to calculate the average model loss
            this_model = model_clone

        

        if self.rank == 0 or self.rank == 1:
            this_model.eval()
            
            loss_total = 0

            with torch.no_grad():
                for _ in range(int(self.config.val_size / self.config.batch_size)):

                    for i in range(self.config.batch_size // self.config.minibatch_size):
                        minibatch = self._get_batch(eval=True)

                        if self.config.autocast:
                            with torch.autocast(device_type=self.config.device_type, dtype=torch.bfloat16):
                                loss = this_model(minibatch)
                        else:
                            loss = this_model(minibatch)

                        loss_total += loss.item() / (self.config.batch_size // self.config.minibatch_size)

        # Rank 0 logs the local evaluation.
        if self.rank == 0:
            # print(f"LOCAL: Eval Loss: {loss_total / int(self.config.val_size / self.config.batch_size):.4f}, "
            #         f"Eval Perplexity: {math.exp(loss_total / int(self.config.val_size / self.config.batch_size)):.4f}")
            self.logger.log_pure(loss=loss_total / int(self.config.val_size / self.config.batch_size), 
                                    name='val_local')

        # Broadcast the global loss from rank 1 to all ranks.
        if self.config.num_nodes > 1:
            # All ranks create a dummy tensor to participate.
            global_loss_tensor = torch.empty(1, device=next(self.model.parameters()).device)
            if self.rank == 1:
                global_loss_tensor[0] = loss_total / int(self.config.val_size / self.config.batch_size)
            broadcast(global_loss_tensor, src=1)

            # Only rank 0 logs the global evaluation.
            if self.rank == 0:
                global_loss = global_loss_tensor.item()
                # print(f"GLOBAL: Eval Loss: {global_loss:.4f}, Eval Perplexity: {math.exp(global_loss):.4f}")
                self.logger.log_pure(loss=global_loss, name='global')

        del model_clone

    def _correlation_calculation(self):
        if self.config.num_nodes < 2:
            raise Exception('Correlation calculation cannot be used with < 2 nodes')
        
        # Ensure correlation is only calculated if interval is set
        if not self.config.correlation_interval:
             return None
        
        # Create a temporary directory for this timestep's checkpoints
        tmp_dir = os.path.join(self.config.save_dir, f"tmp_corr_{self.local_step}")
        # Only rank 0 creates the directory to avoid race conditions
        if self.rank == 0:
            os.makedirs(tmp_dir, exist_ok=True)
        torch.distributed.barrier() # Wait for rank 0 to create dir

        # Save model state dict for each rank
        checkpoint_path = os.path.join(tmp_dir, f"{self.rank}.pt")
        torch.save(self.model.state_dict(), checkpoint_path)

        # Wait for all processes to save their checkpoints
        torch.distributed.barrier()

        corr_value = None
        if self.rank == 0:
            # Load all models as vectors
            model_vectors = []
            for r in range(self.config.num_nodes):
                model_path = os.path.join(tmp_dir, f"{r}.pt")
                # Ensure the file exists before trying to load
                if os.path.exists(model_path):
                    checkpoint = torch.load(model_path, map_location='cpu')
                    vector_list = []
                    for key in sorted(checkpoint.keys()):
                        value = checkpoint[key]
                        if isinstance(value, torch.Tensor):
                            vector_list.append(value.cpu().numpy().ravel())
                    if vector_list: # Check if we actually got any tensors
                        model_vectors.append(np.concatenate(vector_list))
                else:
                    print(f"Warning: Checkpoint file {model_path} not found for rank {r}.")


            if len(model_vectors) >= 2: # Need at least two models to compare
                # Calculate correlations between all pairs
                correlations = []
                for i in range(len(model_vectors)):
                    for j in range(i+1, len(model_vectors)):
                        corr = np.corrcoef(model_vectors[i], model_vectors[j])[0, 1]
                        correlations.append(corr)

                if correlations: # Ensure correlations list is not empty
                    corr_value = np.mean(correlations)

                    # Log average correlation to wandb using the logger
                    if self.logger:
                         self.logger.log(data={'avg_model_correlation': corr_value})
                else:
                    print("Warning: Could not calculate correlation, not enough valid model pairs.")
            else:
                 print(f"Warning: Not enough models loaded ({len(model_vectors)}) to calculate correlation.")


            # Clean up temporary directory
            import shutil
            shutil.rmtree(tmp_dir)

        # Wait for rank 0 to finish cleanup
        torch.distributed.barrier()

        return corr_value # Only rank 0 returns a value, others return None

    def _load_checkpoint(self):
        save_path_dir = os.path.join(self.config.save_dir,
                                 self.config.wandb_project if self.config.wandb_project else 'unnamed',
                                 self.config.wandb_name if self.config.wandb_name else 'unnamed',
                                 str(self.rank))

        if not os.path.exists(save_path_dir):
            print(f"Rank {self.rank}: Checkpoint directory {save_path_dir} not found. Starting from scratch.")
            return False

        checkpoint_files = []
        for f_name in os.listdir(save_path_dir):
            if f_name.endswith('.pt'):
                try:
                    step_num = int(f_name.split('.')[0])
                    checkpoint_files.append((step_num, f_name))
                except ValueError:
                    # Not a valid checkpoint file name pattern
                    continue
        
        # Sort by step number in descending order (latest first)
        checkpoint_files.sort(key=lambda x: x[0], reverse=True)

        loaded_successfully = False
        for step_num, f_name in checkpoint_files:
            full_checkpoint_path = os.path.join(save_path_dir, f_name)
            try:
                print(f"Rank {self.rank}: Attempting to load checkpoint from {full_checkpoint_path}")
                checkpoint = torch.load(full_checkpoint_path, map_location=self.device)

                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.gradient_strategy.optim.load_state_dict(checkpoint['optimizer_state_dict'])
                
                if 'scheduler_state_dict' in checkpoint and self.gradient_strategy.scheduler is not None:
                    self.gradient_strategy.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                
                self.local_step = checkpoint['local_step']
                self.epoch = checkpoint['epoch']
                
                torch.set_rng_state(checkpoint['rng_state'].cpu()) # Ensure RNG state is on CPU before loading
                if self.device.type == 'cuda' and 'cuda_rng_state' in checkpoint:
                    if isinstance(checkpoint['cuda_rng_state'], torch.Tensor):
                        torch.cuda.set_rng_state(checkpoint['cuda_rng_state'].cpu(), device=self.device) 
                    else:
                        torch.cuda.set_rng_state(checkpoint['cuda_rng_state'], device=self.device)

                self.train_data_iter = iter(self.train_dataloader)
                self.val_data_iter = iter(self.val_dataloader)

                if len(self.train_dataloader) > 0:
                    batches_to_skip = self.local_step % len(self.train_dataloader)
                    print(f"Rank {self.rank}: Restored to epoch {self.epoch}, step {self.local_step}. Skipping {batches_to_skip} batches.")
                    for _ in range(batches_to_skip):
                        try:
                            next(self.train_data_iter)
                        except StopIteration:
                            print(f"Rank {self.rank}: Warning - StopIteration while fast-forwarding train_data_iter.")
                            break 
                else:
                    print(f"Rank {self.rank}: Restored to epoch {self.epoch}, step {self.local_step}. Train dataloader empty.")

                if self.rank == 0 and hasattr(self.logger, 'set_step'):
                    self.logger.set_step(self.local_step)
                elif self.rank == 0:
                    print(f"Rank 0: Logger step will resume from loaded local_step: {self.local_step}")

                print(f"Rank {self.rank}: Successfully loaded checkpoint {f_name}. Resuming at epoch {self.epoch}, step {self.local_step}.")
                loaded_successfully = True
                break # Exit loop once a checkpoint is successfully loaded
            except (RuntimeError, EOFError, zipfile.BadZipFile) as e: # Catch specific errors related to corrupted files
                print(f"Rank {self.rank}: Failed to load checkpoint {full_checkpoint_path}: {e}. Trying next available checkpoint.")
                # Optionally, delete the corrupted checkpoint file
                try:
                    os.remove(full_checkpoint_path)
                    print(f"Rank {self.rank}: Deleted corrupted checkpoint {full_checkpoint_path}.")
                except OSError as del_e:
                    print(f"Rank {self.rank}: Warning - Failed to delete corrupted checkpoint {full_checkpoint_path}: {del_e}")
            except Exception as e: # Catch any other unexpected error during loading
                print(f"Rank {self.rank}: An unexpected error occurred while loading checkpoint {full_checkpoint_path}: {e}. Trying next.")
                # Optionally, delete or move the problematic checkpoint

        if not loaded_successfully:
            print(f"Rank {self.rank}: No valid checkpoint found in {save_path_dir} after trying all options. Starting from scratch.")
            # Reset relevant states if starting from scratch, though __init__ defaults should cover this.
            self.local_step = 0 
            self.epoch = 0
            return False
        
        return True

    def train(self):
        while self.local_step < self.max_steps:
            # if self.local_step % self.config.eval_interval == 0:
            #     self._evaluate()
            

            self._train_step()

            self.local_step += 1
            if self.rank == 0:
                self.logger.increment_step()

            # if self.local_step == 5:
            #     break

            # Calculate correlation if interval is set and it's time
            if self.config.correlation_interval and self.local_step > 0 and self.local_step % self.config.correlation_interval == 0:
                self._correlation_calculation()

            dist.barrier()


        self._evaluate()

        if self.config.checkpoint_interval is not None:
            self._save_checkpoint()
