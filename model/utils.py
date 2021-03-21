import torch

def get_pad_mask(seq, pad_token):
    return (seq == pad_token).all(-1)

def get_self_attention_mask(seq):
    ''' For masking out the future info. '''
    # batch, seq_len, hidden
    _, len_s, _ = seq.size()
    atten_mask = (torch.triu(
        torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
    return atten_mask