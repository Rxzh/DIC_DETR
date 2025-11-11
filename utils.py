import torch

def move_batch_to_device(batch,device='mps'):
    
    assert device in ['cpu', 'gpu', 'cuda', 'mps']

    labels_on_device = []

    for label in batch['labels']:
        new_label = {'boxes':label['boxes'].to(torch.device(device)),
                    'class_labels':label['class_labels'].to(torch.device(device))}
        labels_on_device.append(new_label)


    batch['pixel_values'] = batch['pixel_values'].to(torch.device(device))
    batch['pixel_mask'] = batch['pixel_mask'].to(torch.device(device))
    batch['labels'] = labels_on_device
    return batch
