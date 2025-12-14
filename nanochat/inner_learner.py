
import torch
import torch.nn as nn
import torch.nn.functional as F

class InnerLearner:
    def __init__(self, config, device, dtype=torch.float32, 
                 lr=0.01, decay=0.999, clip=1.0, state_max_norm=5.0):
        self.n_embd = config.n_embd
        self.device = device
        self.dtype = dtype
        
        self.lr = lr
        self.decay = decay
        self.clip = clip
        self.state_max_norm = state_max_norm
        
        # Initialize state
        self.state = None
        self.reset()
        
    def reset(self):
        """Reset the inner learner state to zero."""
        self.state = torch.zeros(self.n_embd, device=self.device, dtype=self.dtype)
        
    def apply(self, x):
        """
        Apply the inner state to the input tensor x.
        x: (B, T, C) or (B, C)
        """
        # Broadcast state over batch and time dimensions
        # State is (C,) -> (1, 1, C) or (1, C) depending on x input
        if self.state is None:
            return x
            
        if x.dim() == 3:
            return x + self.state.view(1, 1, -1)
        elif x.dim() == 2:
            return x + self.state.view(1, -1)
        else:
            return x + self.state

    def update(self, h_t, logits_t, sampled_token):
        """
        Update the inner state based on surprisal and gradient signal.
        
        h_t: (B, C) - last hidden state
        logits_t: (B, V) - logits at the last step
        sampled_token: int or (B,) - variable containing the token index
        """
        # Ensure inputs are on correct device/dtype
        h_t = h_t.to(dtype=self.dtype)
        
        # 1. Calculate Surprisal
        # p = softmax(logits)
        probs = F.softmax(logits_t, dim=-1)
        
        # Get probability of the sampled token
        # If sampled_token is scalar (batch size 1)
        if isinstance(sampled_token, int):
             p_token = probs[0, sampled_token]
        else:
             # handle batch behavior if needed, currently assuming batch 1 for inner loop simplicity per spec
             p_token = probs[0, sampled_token[0]]
             
        # Surprisal = -log(p)
        surprisal = -torch.log(p_token + 1e-10) # Prevent log(0)
        
        # 2. Calculate Update Direction
        # grad_signal = normalize(h_t)
        # Using h_t as a proxy for the gradient direction
        # Normalize to unit length
        grad_signal = h_t.mean(dim=0) # Average over batch if B > 1
        grad_signal = F.normalize(grad_signal, p=2, dim=0)
        
        # 3. Calculate Update Delta
        # delta = lr * surprisal * grad_signal
        delta = self.lr * surprisal * grad_signal
        
        # 4. Clip Update
        # clamp norm of delta
        delta_norm = torch.norm(delta)
        if delta_norm > self.clip:
            delta = delta * (self.clip / delta_norm)
            
        # 5. Apply Update
        # state = decay * state + delta
        self.state = self.decay * self.state + delta
        
        # 6. Clip State Norm
        # ||state|| <= state_max_norm
        state_norm = torch.norm(self.state)
        if state_norm > self.state_max_norm:
            self.state = self.state * (self.state_max_norm / state_norm)
            
        # Return metrics for logging
        return {
            'surprisal': surprisal.item(),
            'update_norm': delta_norm.item(),
            'state_norm': state_norm.item()
        }
