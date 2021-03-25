import torch
import torch.nn as nn
import torch.nn.functional as F

class ACT(nn.Module):
    def __init__(self, fn, hidden_size, max_hop, timestamp_emb, position_emb, act_epilson):
        super(ACT, self).__init__()
        self.p = nn.Linear(hidden_size,1)  
        nn.init.ones_(self.p.bias)
        
        self.register_buffer("threshold", 1 - act_epilson)
        self.register_buffer("max_hop", max_hop)

        self.fn = fn
        self.timestamp_emb = timestamp_emb
        self.position_emb = position_emb

    def forward(self, state, encoder_output=None, source_mask=None, target_mask=None):
        # init_hdd
        update_shape = state.shape[:-1]
        ## [batch, seq, 1]
        halting_probability = torch.zeros(update_shape).type_as(state)
        ## [batch, seq]
        remainders = torch.zeros(update_shape).type_as(state)
        ## [batch, seq, 1]
        n_updates = torch.zeros(update_shape).type_as(state)
        ## [batch, seq, hidden]
        previous_state = torch.zeros_like(state).type_as(state)
        step = 0
        # for l in range(self.num_layers):
        while(((halting_probability < self.threshold) & (n_updates < self.max_hop)).byte().any()):
            # Add timing signal
            state = self.timestamp_emb(state, step)
            state = self.position_emb(state, step)

            p = torch.sigmoid(self.p(state)).squeeze(-1)
            # Mask for inputs which have not halted yet
            still_running = (halting_probability < 1.0).float()

            # Mask of inputs which halted at this step
            new_halted = (halting_probability + p * still_running > self.threshold).float() * still_running

            # Mask of inputs which haven't halted, and didn't halt this step
            still_running = (halting_probability + p * still_running <= self.threshold).float() * still_running

            # Add the halting probability for this step to the halting
            # probabilities for those input which haven't halted yet
            halting_probability = halting_probability + p * still_running

            # Compute remainders for the inputs which halted at this step
            remainders = remainders + new_halted * (1 - halting_probability)

            # Add the remainders to those inputs which halted at this step
            halting_probability = halting_probability + new_halted * remainders

            # Increment n_updates for all inputs which are still running
            n_updates = n_updates + still_running + new_halted

            # Compute the weight to be applied to the new state and output
            # 0 when the input has already halted
            # p when the input hasn't halted yet
            # the remainders when it halted this step
            update_weights = p * still_running + new_halted * remainders

            # apply decoder on the state
            if(encoder_output != None):
                state = self.fn(encoder_output, state, source_mask, target_mask)
            else:
                # apply encoder on the state
                state = self.fn(state, source_mask)

            # update running part in the weighted state and keep the rest
            previous_state = ((state * update_weights.unsqueeze(-1)) + (previous_state * (1 - update_weights.unsqueeze(-1))))
            ## previous_state is actually the new_state at end of hte loop 
            ## to save a line I assigned to previous_state so in the next 
            ## iteration is correct. Notice that indeed we return previous_state
            step+=1
        return previous_state, (remainders, n_updates)