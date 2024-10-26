import torch
from dataset import causal_mask
import torch.nn.functional as F
from torchmetrics import Accuracy
from torcheval.metrics import MultilabelAccuracy
from tqdm import tqdm

def convert_onehot(item, item_size, device):
    item = item.long()
    one_hot_tensor = torch.zeros(item.size(0), item_size, device=device)
    if item.dim() == 1:
        item = item.unsqueeze(1)
    one_hot_tensor.scatter_(1, item, 1)
    return one_hot_tensor

def greedy_decode(model, sos_idx, eos_idx, encoder_input, encoder_mask, actual_len, device):
    encoder_output = model.encode(encoder_input[:, :, 1], encoder_input[:, :, 0], encoder_mask)
    decoder_input_clicks = torch.empty(1, 1).fill_(sos_idx).to(device).int()
    decoder_input_carts = decoder_input_orders = torch.empty(1, 5).fill_(sos_idx).unsqueeze(0).to(device).int()
    while True:
        if decoder_input_clicks.size(1) == actual_len + 1:
            break

        decoder_mask = causal_mask(decoder_input_clicks.squeeze(-1).size(0)).to(device)

        decoder_output_clicks, decoder_output_carts, decoder_output_orders = model.decode(decoder_input_clicks, 
                                                                                                decoder_input_carts, 
                                                                                                decoder_input_orders,
                                                                                                encoder_output, 
                                                                                                encoder_mask,
                                                                                                decoder_mask) 
        
        proj_output_clicks = model.project(decoder_output_clicks[:, -1])
        proj_output_carts = model.project(decoder_output_carts[:, -1])
        proj_output_orders = model.project(decoder_output_orders[:, -1])
            
        _, next_clicks = torch.max(proj_output_clicks, dim=1)
        _, next_carts = torch.topk(proj_output_carts, 5, dim=1)
        _, next_orders = torch.topk(proj_output_orders, 5, dim=1)
        
        decoder_input_clicks = torch.cat(
            [decoder_input_clicks, torch.empty(1, 1).type_as(encoder_output).fill_(next_clicks.item()).to(device)], dim=1
        )
                
        decoder_input_carts = torch.cat(
            [decoder_input_carts, torch.empty(1, 5).type_as(encoder_output).copy_(next_carts).unsqueeze(0).to(device)], dim=1
        )
        
        decoder_input_orders = torch.cat(
            [decoder_input_orders, torch.empty(1, 5).type_as(encoder_output).copy_(next_orders).unsqueeze(0).to(device)], dim=1
        )

        if next_clicks == eos_idx:
            break

    return decoder_input_clicks.squeeze(0), decoder_input_carts.squeeze(0), decoder_input_orders.squeeze(0)

def run_validation(model, sos_idx, eos_idx, pad_idx, validation_ds, device, item_size):
    model.eval()
    total_accuracys = []
    total_accuracys_clicks = []
    total_accuracys_carts = []
    total_accuracys_orders = []
    accuracy_int = Accuracy(task="multiclass", num_classes=100004).to(device)
    accuracy_one_hot = MultilabelAccuracy()

    with torch.no_grad():
        batch_iterator = tqdm(validation_ds, desc=f"Validating...")
        for batch in batch_iterator:
            try:
                encoder_input = batch["encoder_input"].to(device) 
                encoder_mask = batch["encoder_mask"].to(device) 
                actual_len = batch["actual_len"].item()

                assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"

                model_out_clicks, model_out_carts, model_out_orders = greedy_decode(model, 
                                                                                    sos_idx, 
                                                                                    eos_idx, 
                                                                                    encoder_input, 
                                                                                    encoder_mask, 
                                                                                    actual_len, 
                                                                                    device)

                model_out_carts = convert_onehot(model_out_carts, item_size, device)[1:, :]
                model_out_orders = convert_onehot(model_out_orders, item_size, device)[1:, :]

                label_clicks = batch['label_clicks'].to(device)[:, :actual_len].squeeze(0)
                label_carts = batch['label_carts'].to(device)[:, :actual_len].squeeze(0)
                label_orders = batch['label_orders'].to(device)[:, :actual_len].squeeze(0)

                accuracy_clicks = accuracy_int(model_out_clicks[1:].int().to(device), torch.argmax(label_clicks, dim=1).to(device))

                accuracy_one_hot.update(model_out_carts, label_carts)
                accuracy_carts = accuracy_one_hot.compute()
                accuracy_one_hot.reset()

                accuracy_one_hot.update(model_out_orders, label_orders)
                accuracy_orders = accuracy_one_hot.compute()
                accuracy_one_hot.reset()

                pad = F.one_hot(torch.tensor(pad_idx, dtype=torch.int64), num_classes=item_size).to(device)
                is_carts_all_pad = torch.all(torch.eq(label_carts, pad)).item()
                is_orders_all_pad = torch.all(torch.eq(label_orders, pad)).item()

                if is_carts_all_pad and is_orders_all_pad:
                    total_accuracy = accuracy_clicks
                elif is_carts_all_pad and not is_orders_all_pad:
                    total_accuracy = accuracy_clicks * 0.2 + accuracy_orders * 0.8
                    total_accuracys_orders.append(accuracy_orders.item())
                elif not is_carts_all_pad and is_orders_all_pad:
                    total_accuracy = accuracy_clicks * 0.4 + accuracy_carts * 0.6
                    total_accuracys_carts.append(accuracy_carts.item())
                else:
                    total_accuracy = accuracy_clicks * 0.1 + accuracy_carts * 0.3 + accuracy_orders * 0.6
                    total_accuracys_orders.append(accuracy_orders.item())
                    total_accuracys_carts.append(accuracy_carts.item())

                total_accuracys_clicks.append(accuracy_clicks.item())
                total_accuracys.append(total_accuracy.item())

            except Exception as e:
                print(f"Skipping batch due to error: {e}")
                continue  

    return sum(total_accuracys) / len(total_accuracys), sum(total_accuracys_clicks) / len(total_accuracys_clicks), sum(total_accuracys_carts) / len(total_accuracys_carts), sum(total_accuracys_orders) / len(total_accuracys_orders)

    
