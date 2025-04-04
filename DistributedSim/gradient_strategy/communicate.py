import torch.distributed as dist
import inspect
import datetime

def mps_compatible(func):
    # Wrapper for all_gather which handles tensor_list and tensor
    def all_gather_wrapper(tensor_list, tensor, *args, **kwargs):
        # Check if either is on MPS
        is_tensor_mps = hasattr(tensor, 'device') and tensor.device.type == 'mps'
        is_list_mps = any(hasattr(t, 'device') and t.device.type == 'mps' for t in tensor_list)
        
        if is_tensor_mps or is_list_mps:
            # Convert tensor to CPU if needed
            if is_tensor_mps:
                cpu_tensor = tensor.data.to('cpu')
            else:
                cpu_tensor = tensor
            
            # Convert tensor_list to CPU if needed
            cpu_tensor_list = []
            for t in tensor_list:
                if hasattr(t, 'device') and t.device.type == 'mps':
                    cpu_tensor_list.append(t.data.to('cpu'))
                else:
                    cpu_tensor_list.append(t)
            
            # Call function with CPU tensors
            result = func(cpu_tensor_list, cpu_tensor, *args, **kwargs)
            
            # Copy data back to original devices
            if is_tensor_mps:
                tensor.data.copy_(cpu_tensor.to('mps'))
            
            for i, t in enumerate(tensor_list):
                if hasattr(t, 'device') and t.device.type == 'mps':
                    t.data.copy_(cpu_tensor_list[i].to('mps'))
            
            return result
        else:
            return func(tensor_list, tensor, *args, **kwargs)
    
    # Wrapper for all other functions that handle a single tensor
    def standard_wrapper(tensor, *args, **kwargs):
        if hasattr(tensor, 'device') and tensor.device.type == 'mps':
            # Move the tensor to CPU
            cpu_tensor = tensor.data.to('cpu')
            # Call the function on CPU
            result = func(cpu_tensor, *args, **kwargs)
            # Copy the result back to mps
            tensor.data.copy_(cpu_tensor.to('mps'))
            return result
        else:
            return func(tensor, *args, **kwargs)
    
    # Return the appropriate wrapper based on function name
    if func.__name__ == 'all_gather':
        return all_gather_wrapper
    else:
        return standard_wrapper

@mps_compatible
def broadcast(tensor, src=0):
    return dist.broadcast(tensor, src=src)

@mps_compatible
def all_reduce(tensor, op=dist.ReduceOp.SUM):
    return dist.all_reduce(tensor, op=op)

@mps_compatible
def all_gather(tensor_list, tensor, group=None, async_op=False):
    return dist.all_gather(tensor_list, tensor, group=group, async_op=async_op)

# @mps_compatible
# def reduce_scatter(tensor):
#     return dist.reduce_scatter(tensor)

# @mps_compatible
def reduce(tensor, dst=None, op=dist.ReduceOp.SUM):
    return dist.reduce(tensor, dst=dst, op=op)

# @mps_compatible
# def gather(tensor):
#     return dist.gather(tensor)
