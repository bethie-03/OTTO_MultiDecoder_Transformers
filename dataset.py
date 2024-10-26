import torch
import torch.nn as nn
import json
from labels import ground_truth, list_to_onehot, add_pad
from torch.utils.data import Dataset
import torch.nn.functional as F

def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.readlines() 
    return content

class OTTODataset(Dataset):

    def __init__(self, jsonfile, seq_len, item_size):
        super().__init__()
        
        self.jsonfile = jsonfile
        self.content = read_json(jsonfile)
        self.event_type_dict = {
                                "clicks": 0,
                                "carts": 1,
                                "orders": 2
                            }
        self.spec_key = {
                        "sos": item_size - 3,
                        "eos": item_size - 2,
                        "pad": item_size - 1
        }
        self.seq_len = seq_len
        self.item_size = item_size
        self.pad = torch.zeros(item_size)
        self.pad[item_size - 1] = 1
        self.pad = self.pad.unsqueeze(0)
        
    def __len__(self):
        return len(self.content)

    def __getitem__(self, idx):
        session = json.loads(self.content[idx])
        
        input_pair = []
        for event in session['events']:
            input_pair.append([self.event_type_dict.get(event['type'], "Unknow"), event['aid']])
        input_pair.pop()
        
        event_ground_truth = ground_truth(session['events'])
        clicks = []
        carts = []
        orders = []
        for event in event_ground_truth:
            label = event['labels']

            click = int(label['clicks']) if 'clicks' in label else -1
            cart = list(label['carts']) if 'carts' in label else [-1]
            order = list(label['orders']) if 'orders' in label else [-1]
            
            clicks.append([click])
            carts.append(cart)
            orders.append(order)
                        
        clicks_onehot = torch.stack([list_to_onehot(click, self.item_size) for click in clicks])
        carts_onehot = torch.stack([list_to_onehot(cart, self.item_size) for cart in carts])
        orders_onehot = torch.stack([list_to_onehot(order, self.item_size) for order in orders])
                        
        if len(carts[0]) > 1:
            carts = add_pad(carts)
            pad_carts = [self.spec_key["pad"]] * len(carts[0])
            sos_carts = [self.spec_key["sos"]] * len(carts[0])
        else:
            sos_carts = [self.spec_key["sos"]]
            pad_carts = [self.spec_key["pad"]] 
            
        if len(orders[0]) > 1:
            orders = add_pad(orders)
            pad_orders = [self.spec_key["pad"]] * len(orders[0]) 
            sos_orders = [self.spec_key["sos"]] * len(orders[0]) 
        else:
            sos_orders = [self.spec_key["sos"]]
            pad_orders = [self.spec_key["pad"]] 
                    
        enc_num_padding_tokens = self.seq_len - len(input_pair) - 2
        dec_num_padding_tokens = self.seq_len - len(clicks) - 1
        
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("Session is too long")
        
        encoder_input = torch.cat(
            [
                torch.tensor([3, self.spec_key["sos"]], dtype=torch.int64).unsqueeze(0),
                torch.tensor(input_pair, dtype=torch.int64),
                torch.tensor([4, self.spec_key["eos"]], dtype=torch.int64).unsqueeze(0),
                torch.tensor([[5, self.spec_key["pad"]]] * enc_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )
                        
        decoder_input_clicks = torch.cat(
            [
                torch.tensor([[self.spec_key["sos"]]], dtype=torch.int64),
                torch.tensor(clicks, dtype=torch.int64),
                torch.tensor([[self.spec_key["pad"]]] * dec_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )
                    
        decoder_input_carts = torch.cat(
            [
                torch.tensor([sos_carts], dtype=torch.int64),
                torch.tensor(carts, dtype=torch.int64),
                torch.tensor([pad_carts] * dec_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )
        
        decoder_input_orders = torch.cat(
            [
                torch.tensor([sos_orders], dtype=torch.int64),
                torch.tensor(orders, dtype=torch.int64),
                torch.tensor([pad_orders] * dec_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )
        
        label_clicks = torch.cat(
            [
                clicks_onehot,
                F.one_hot(torch.tensor([self.spec_key["eos"]], dtype=torch.int64), num_classes=self.item_size),
                F.one_hot(torch.tensor([self.spec_key["pad"]] * dec_num_padding_tokens, dtype=torch.int64), num_classes=self.item_size),
            ],
            dim=0,
        )

        label_carts = torch.cat(
            [
                carts_onehot,
                F.one_hot(torch.tensor([self.spec_key["eos"]], dtype=torch.int64), num_classes=self.item_size),
                F.one_hot(torch.tensor([self.spec_key["pad"]] * dec_num_padding_tokens, dtype=torch.int64), num_classes=self.item_size),
            ],
            dim=0,
        )
                
        label_orders = torch.cat(
            [
                orders_onehot,
                F.one_hot(torch.tensor([self.spec_key["eos"]], dtype=torch.int64), num_classes=self.item_size),
                F.one_hot(torch.tensor([self.spec_key["pad"]] * dec_num_padding_tokens, dtype=torch.int64), num_classes=self.item_size),
            ],
            dim=0,
        )
                
        return {
                "encoder_input": encoder_input,  
                "encoder_mask": (encoder_input != torch.tensor([[5, self.spec_key["pad"]]])).unsqueeze(0).unsqueeze(0)[..., 0].int(),
                "decoder_input_clicks" : decoder_input_clicks,
                "decoder_input_carts" : decoder_input_carts,
                "decoder_input_orders" : decoder_input_orders,
                "decoder_mask": (decoder_input_clicks.squeeze(-1) != torch.tensor([[self.spec_key["pad"]]])).unsqueeze(0).int() & causal_mask(decoder_input_clicks.squeeze(-1).size(0)),
                "label_clicks": label_clicks,  
                "label_carts": label_carts,
                "label_orders": label_orders, 
                "actual_len": len(input_pair),
            }
    
def causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0

