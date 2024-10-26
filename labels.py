import torch
from itertools import zip_longest

def add_pad(data):
    max_len = max(len(row) for row in data)
    padded_list = [row + [-1] * (max_len - len(row)) for row in data]
    
    return padded_list

def list_to_onehot(label, num_classes):
    one_hot = torch.zeros(num_classes)
    one_hot[label] = 1
    return one_hot

def ground_truth(events: list[dict]):
    prev_labels = {"clicks": None, "carts": set(), "orders": set()}

    for event in reversed(events):
        event["labels"] = {}

        for label in ['clicks', 'carts', 'orders']:
            if prev_labels[label]:
                if label != 'clicks':
                    event["labels"][label] = prev_labels[label].copy()
                else:
                    event["labels"][label] = prev_labels[label]

        if event["type"] == "clicks":
            prev_labels['clicks'] = event["aid"]
        if event["type"] == "carts":
            prev_labels['carts'].add(event["aid"])
        elif event["type"] == "orders":
            prev_labels['orders'].add(event["aid"])

    return events[:-1]
