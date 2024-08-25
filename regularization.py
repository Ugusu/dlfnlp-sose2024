import torch
from torch import nn
import torch.nn.functional as F

class SMART:
    """
    Smoothness-Inducing Adversarial Regularization implementation.
    """
    def __init__(
        self, 
        model: torch.nn.Module, 
        epsilon: float=1e-5, 
        alpha: float=0.02, 
        steps: int=1
    ):
        """
        Initializes the SMART Regularizer with specified parameters.
        
        Args:
            model (torch.nn.Module): The model being trained.
            epsilon (float): Norm constraint for the perturbation.
            alpha (float): Step size for adversarial perturbation.
            steps (int): Number of steps for generating perturbations.
        """
        print(f"Initialized SMART with epsilon={epsilon}, alpha={alpha}, steps={steps}")
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.steps = steps
    
    def perturb(self, input_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Applies adversarial perturbation to the input embeddings.
        
        Args:
            input_embeddings (torch.Tensor): The input embeddings.
            attention_mask (torch.Tensor): The attention mask for the embeddings.
        
        Returns:
            perturbed_embeddings (torch.Tensor): The embeddings with adversarial perturbation.
        """
        perturbation = torch.randn_like(input_embeddings, requires_grad=True)
        
        for _ in range(self.steps):
            perturbed_embeddings = input_embeddings + perturbation
            perturbed_embeddings.requires_grad_(True)
            outputs = self.model.encode(perturbed_embeddings, attention_mask)
            loss = outputs.norm()
            print("Before zero", perturbation.grad)
            if perturbation.grad is not None:
                perturbation.grad.zero_()
            loss.backward(retain_graph=True)
            print("After zero", perturbation.grad)
            perturbation = perturbation + self.alpha * perturbation.grad.sign()
            perturbation = torch.clamp(perturbation, -self.epsilon, self.epsilon)
            self.model.zero_grad()

        return input_embeddings + perturbation
    
    def forward(self, logits: torch.Tensor, input_ids: list[torch.Tensor], attention_masks: list[torch.Tensor], classifier: bool=True) -> torch.Tensor:
        if not isinstance(input_ids, list):
            input_ids = [input_ids]
        
        if not isinstance(attention_masks, list):
            attention_masks = [attention_masks]

        concatenated_embeddings = None
        concatenated_attention_masks = attention_masks[0]

        for i in range(len(input_ids)):
            embeddings = self.model.embed(input_ids[i])
            embeddings = self.perturb(embeddings, attention_masks[i])
            if concatenated_embeddings is None:
                concatenated_embeddings = embeddings
            else:
                concatenated_embeddings = torch.cat((concatenated_embeddings, embeddings[:, 1:, :]), dim=1)
                concatenated_attention_masks = torch.cat((concatenated_attention_masks, attention_masks[i][:, 1:]), dim=1)

        with torch.no_grad():
            perturbed_outputs = self.model.encode(concatenated_embeddings, concatenated_attention_masks)
            perturbed_logits = self.model.pooler_dense(perturbed_outputs[:, 0])
            perturbed_logits = self.model.pooler_af(perturbed_logits)
        
        if classifier:
            # Classification: KL-divergence
            kl_loss = nn.KLDivLoss(reduction='batchmean')
            smart_loss = kl_loss(
                F.log_softmax(perturbed_logits, dim=-1),
                F.softmax(logits, dim=-1)
            ) + kl_loss(
                F.log_softmax(logits, dim=-1),
                F.softmax(perturbed_logits, dim=-1)
            )
            return smart_loss
        else:
            # Regression: Mean Squared Error
            squared_loss = nn.MSELoss()
            smart_loss = squared_loss(perturbed_logits, logits)
            return smart_loss