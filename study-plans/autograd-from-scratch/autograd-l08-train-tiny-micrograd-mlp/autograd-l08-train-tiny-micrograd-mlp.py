import torch

def train_tiny_micrograd_mlp(
    inputs: torch.Tensor, 
    targets: torch.Tensor, 
    weights: list[torch.Tensor], 
    biases: list[torch.Tensor], 
    learning_rate: float, 
    steps: int
) -> tuple:
    """
    Huấn luyện mạng MLP bằng Full-Batch Gradient Descent từ đầu (From-Scratch),
    tự tính toán lan truyền ngược mà không dùng Autograd.
    """
    w_list = [w.clone().detach() for w in weights]
    b_list = [b.clone().detach() for b in biases]
    
    loss_history = []
    dtype = inputs.dtype
    device = inputs.device
    B = inputs.shape[0]

    for step in range(steps):

        H_list = [inputs]
        for w, b in zip(w_list, b_list):
            H_next = torch.tanh(H_list[-1] @ w.t() + b)
            H_list.append(H_next)
            
        predictions = H_list[-1].squeeze(-1) if B > 0 else H_list[-1].view(-1)
        
        if B == 0:
            loss = torch.tensor(0.0, dtype=dtype, device=device)
        else:
            loss = torch.sum((predictions - targets) ** 2)
        loss_history.append(loss)
        
        if B > 0:
            dL_dH = (2.0 * (predictions - targets)).unsqueeze(-1)
            
            w_grads = []
            b_grads = []
            
            for l in reversed(range(len(w_list))):
                H_prev = H_list[l]      
                
                H_curr = H_list[l + 1]   
                
                W_curr = w_list[l]       
                
                dL_dA = dL_dH * (1.0 - H_curr ** 2)
                

                dW = dL_dA.t() @ H_prev

                db = dL_dA.sum(dim=0)
                
                w_grads.insert(0, dW)
                b_grads.insert(0, db)
                
                dL_dH = dL_dA @ W_curr
            
            for l in range(len(w_list)):
                w_list[l] -= learning_rate * w_grads[l]
                b_list[l] -= learning_rate * b_grads[l]

    H = inputs
    for w, b in zip(w_list, b_list):
        H = torch.tanh(H @ w.t() + b)
    final_predictions = H.squeeze(-1) if B > 0 else H.view(-1)
    
    if B == 0:
        final_loss = torch.tensor(0.0, dtype=dtype, device=device)
    else:
        final_loss = torch.sum((final_predictions - targets) ** 2)
        
    return final_predictions, final_loss, w_list, b_list, loss_history
