import torch
import torch.nn as nn
from tqdm import tqdm
from model import build_transformer
from dataset import OTTODataset
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
from ignite.metrics import *
from ignite.engine import *
import json
from validation import run_validation
import time

def custom_collate_fn(batch):
    max_len = max(item['encoder_input'].shape[0] for item in batch)
    max_cart_len = max(item['decoder_input_carts'].shape[1] for item in batch)
    max_click_len = max(item['decoder_input_clicks'].shape[1] for item in batch)
    max_order_len = max(item['decoder_input_orders'].shape[1] for item in batch)  # For decoder_input_orders

    def pad_tensor(tensor, max_len, max_second_dim, pad_value=0):
        pad_size_first = max_len - tensor.size(0)
        pad_size_second = max_second_dim - tensor.size(1)
        
        if pad_size_first > 0 or pad_size_second > 0:
            padding = (0, pad_size_second, 0, pad_size_first)  # Correct shape of padding for multi-dim tensors
            return F.pad(tensor, padding, value=pad_value)
        else:
            return tensor
    
    batch_out = {
        'encoder_input': torch.stack([pad_tensor(item['encoder_input'], max_len, item['encoder_input'].shape[1]) for item in batch]),
        'encoder_mask': torch.stack([item['encoder_mask'] for item in batch]),  
        'decoder_input_clicks': torch.stack([pad_tensor(item['decoder_input_clicks'], max_len, max_click_len) for item in batch]),
        'decoder_input_carts': torch.stack([pad_tensor(item['decoder_input_carts'], max_len, max_cart_len) for item in batch]),
        'decoder_input_orders': torch.stack([pad_tensor(item['decoder_input_orders'], max_len, max_order_len) for item in batch]),  
        'decoder_mask': torch.stack([item['decoder_mask'] for item in batch]),  
        'label_clicks': torch.stack([pad_tensor(item['label_clicks'], max_len, item['label_clicks'].shape[1]) for item in batch]),
        'label_carts': torch.stack([pad_tensor(item['label_carts'], max_len, max_cart_len) for item in batch]),
        'label_orders': torch.stack([pad_tensor(item['label_orders'], max_len, max_order_len) for item in batch]),  
        'actual_len': torch.tensor([item['actual_len'] for item in batch])  
    }

    return batch_out


def apply_mask_click(item, mask, device):
    item = torch.where(mask.bool(), item, torch.tensor(-1, device=device))
    return item

def apply_mask_cart_order(item, mask):
    return item[mask.squeeze(2)]

def eval_step(engine, batch):
    return batch

def thresholded_output_transform(output):
    y_pred, y = output
    y_pred = torch.round(y_pred)
    return y_pred, y

default_evaluator_1 = Engine(eval_step)
default_evaluator_2 = Engine(eval_step)
metric_fn_click = Accuracy()
metric_fn_click.attach(default_evaluator_1, "accuracy")

metric_fn_cart_order = Accuracy(output_transform=thresholded_output_transform)
metric_fn_cart_order.attach(default_evaluator_2, "accuracy")

def accuracy(predicted_clicks, predicted_carts, predicted_orders, label_clicks, label_carts, label_orders):
    state_1 = default_evaluator_1.run([[predicted_clicks, label_clicks]])                
    state_2 = default_evaluator_2.run([[predicted_carts, label_carts]])    
    state_3 = default_evaluator_2.run([[predicted_orders, label_orders]])
    return state_1.metrics["accuracy"], state_2.metrics["accuracy"], state_3.metrics["accuracy"]
         
def train_model(item_size, seq_len, d_model, batch_size, split_ratio, num_epochs, learning_rate, device):
    best_val_accuracy = 0.0
    
    history_logs = {
        'train_total_loss': [],
        'train_loss_click': [],
        'train_loss_cart': [],
        'train_loss_order': [],
        'train_total_accuracy': [],
        'train_accuracy_click': [],
        'train_accuracy_cart': [],
        'train_accuracy_order': [],
        'val_total_accuracy': [],
        'val_accuracy_click': [],
        'val_accuracy_cart': [],
        'val_accuracy_order': []
    }
    
    model = build_transformer(item_size, item_size, seq_len, seq_len, d_model=d_model).to(device)
    dataset = OTTODataset(jsonfile='train_100000.jsonl', seq_len=seq_len, item_size=item_size)

    train_size = int(split_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, eps=1e-9)
    softmax = nn.Softmax()

    loss_fn_click = nn.CrossEntropyLoss(ignore_index=dataset.spec_key['pad'], label_smoothing=0.1).to(device)
    loss_fn_cart_order = nn.BCEWithLogitsLoss().to(device)
    
    scaler = torch.amp.GradScaler("cuda")

    for epoch in range(num_epochs):
        train_total_loss = []
        total_loss_clicks = []
        total_loss_carts = [] 
        total_loss_orders = []
        total_accuracys = []
        total_accuracy_clicks = []
        total_accuracy_carts = [] 
        total_accuracy_orders = []
    
        torch.cuda.empty_cache()
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
        start = time.time()
        for batch in batch_iterator:
            optimizer.zero_grad(set_to_none=True)

            encoder_input = batch['encoder_input'].to(device) 
            
            encoder_mask = batch['encoder_mask'].to(device) 
            
            decoder_input_clicks = batch['decoder_input_clicks'].to(device) 
            decoder_input_carts = batch['decoder_input_carts'].to(device) 
            decoder_input_orders = batch['decoder_input_orders'].to(device) 
            
            label_clicks = batch['label_clicks'].to(device) 
            label_clicks = torch.argmax(label_clicks, dim=2).view(-1)
            label_carts = batch['label_carts'].to(device) 
            label_orders = batch['label_orders'].to(device)
    
            decoder_mask = batch['decoder_mask'].to(device)
            
            actual_len = batch["actual_len"]
                        
            with torch.amp.autocast("cuda", dtype=torch.float16):
                encoder_output = model.encode(encoder_input[:, :, 1], encoder_input[:, :, 0], encoder_mask) 
                decoder_output_clicks, decoder_output_carts, decoder_output_orders = model.decode(decoder_input_clicks, 
                                                                                                    decoder_input_carts, 
                                                                                                    decoder_input_orders,
                                                                                                    encoder_output, 
                                                                                                    encoder_mask,
                                                                                                    decoder_mask) 
                
                
                proj_output_clicks = model.project(decoder_output_clicks)
                proj_output_carts = model.project(decoder_output_carts)
                proj_output_orders = model.project(decoder_output_orders)
                
                ignore_value = F.one_hot(torch.tensor(dataset.spec_key['pad'], dtype=torch.int64), num_classes=item_size).to(device)
                        
                mask_clicks = label_clicks != dataset.spec_key['pad']
                mask_carts = (label_carts != ignore_value).any(dim=2).unsqueeze(-1)
                mask_orders = (label_orders != ignore_value).any(dim=2).unsqueeze(-1)
                
                label_carts = apply_mask_cart_order(label_carts, mask_carts)
                label_orders = apply_mask_cart_order(label_orders, mask_orders)
                
                proj_output_carts = apply_mask_cart_order(proj_output_carts, mask_carts)
                proj_output_orders = apply_mask_cart_order(proj_output_orders, mask_orders)
                
                loss_clicks = loss_fn_click(proj_output_clicks.view(-1, 100004), label_clicks)
                loss_carts = loss_fn_cart_order(proj_output_carts, label_carts) 
                loss_orders = loss_fn_cart_order(proj_output_orders, label_orders) 
                
                '''is_carts_all_pad = torch.all(torch.eq(batch['label_carts'].to(device)[:, :actual_len].squeeze(0), ignore_value)).item()
                is_orders_all_pad = torch.all(torch.eq(batch['label_orders'].to(device)[:, :actual_len].squeeze(0), ignore_value)).item()'''            
                
                proj_output_clicks = apply_mask_click(softmax(proj_output_clicks.view(-1, item_size)), mask_clicks.unsqueeze(1), device)
                label_clicks = apply_mask_click(label_clicks, mask_clicks, device)
                
                total_accuracy_click, total_accuracy_cart, total_accuracy_order = accuracy(proj_output_clicks, softmax(proj_output_carts), softmax(proj_output_orders), label_clicks, label_carts, label_orders)
        
            '''if is_carts_all_pad and is_orders_all_pad:
                total_accuracy = total_accuracy_click
                total_loss = loss_clicks
            elif is_carts_all_pad and not is_orders_all_pad:
                total_accuracy = total_accuracy_click * 0.2 + total_accuracy_order * 0.8
                total_loss = loss_clicks * 0.2 + loss_orders * 0.8
                total_loss_orders.append(loss_orders.item())
                total_accuracy_orders.append(total_accuracy_order)
            elif not is_carts_all_pad and is_orders_all_pad:
                total_accuracy = total_accuracy_click * 0.4 + total_accuracy_cart * 0.6
                total_loss = loss_clicks * 0.4 + loss_carts * 0.6 
                total_loss_carts.append(loss_carts.item())
                total_accuracy_carts.append(total_accuracy_cart)
            else:
                total_accuracy = total_accuracy_click * 0.1 + total_accuracy_cart * 0.3 + total_accuracy_order * 0.6
                total_loss = loss_clicks * 0.1 + loss_carts * 0.3 + loss_orders * 0.6
                total_loss_carts.append(loss_carts.item())
                total_loss_orders.append(loss_orders.item())
                total_accuracy_orders.append(total_accuracy_order)
                total_accuracy_carts.append(total_accuracy_cart)'''
                
            total_accuracy = total_accuracy_click * 0.1 + total_accuracy_cart * 0.3 + total_accuracy_order * 0.6
            total_accuracy_clicks.append(total_accuracy_click)
            total_accuracy_orders.append(total_accuracy_order)
            total_accuracy_carts.append(total_accuracy_cart)
            total_accuracys.append(total_accuracy)
            
            total_loss = loss_clicks * 0.1 + loss_carts * 0.3 + loss_orders * 0.6
            train_total_loss.append(total_loss.item())
            total_loss_clicks.append(loss_clicks.item())
            total_loss_carts.append(loss_carts.item())
            total_loss_orders.append(loss_orders.item())
            
            scaler.scale(total_loss).backward()

            scaler.step(optimizer)

            scaler.update()

        val_accuracy, val_accuracy_clicks, val_accuracy_carts, val_accuracy_orders = run_validation(model, dataset.spec_key['sos'], dataset.spec_key['eos'], dataset.spec_key['pad'], val_dataloader, device, item_size)
        end = time.time()
        print(end-start)
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {sum(train_total_loss) / len(train_total_loss):.4f}, Train Accuracy: {sum(total_accuracys) / len(total_accuracys):.2f}%, Val Accuracy: {val_accuracy:.2f}%")
        
        history_logs['train_total_loss'].append(sum(train_total_loss) / len(train_total_loss))
        history_logs['train_loss_click'].append(sum(total_loss_clicks) / len(total_loss_clicks))
        history_logs['train_loss_cart'].append(sum(total_loss_carts) / len(total_loss_carts))
        history_logs['train_loss_order'].append(sum(total_loss_orders) / len(total_loss_orders))
        history_logs['train_total_accuracy'].append(sum(total_accuracys) / len(total_accuracys))
        history_logs['train_accuracy_click'].append(sum(total_accuracy_clicks) / len(total_accuracy_clicks))
        history_logs['train_accuracy_cart'].append(sum(total_accuracy_carts) / len(total_accuracy_carts))
        history_logs['train_accuracy_order'].append(sum(total_accuracy_orders) / len(total_accuracy_orders))
        history_logs['val_total_accuracy'].append(val_accuracy)
        history_logs['val_accuracy_click'].append(val_accuracy_clicks)
        history_logs['val_accuracy_cart'].append(val_accuracy_carts)
        history_logs['val_accuracy_order'].append(val_accuracy_orders)

        if val_accuracy > best_val_accuracy:
            torch.save(model.state_dict(), '/home/transformers/checkpoints/best.pt')
            best_val_accuracy = val_accuracy
            
        with open('/home/transformers/logs.json', 'w') as f:
            json.dump(history_logs, f, indent=4)
            
    torch.save(model.state_dict(), '/home/transformers/checkpoints/last.pt')
        
if __name__ == '__main__':
    train_model(100004, 228, 64, 32, 0.9, 3, 0.001, 'cuda')