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
        
        for step in range(self.steps):
            perturbed_embeddings = input_embeddings + perturbation
            outputs = self.model.bert.encode(perturbed_embeddings, attention_mask)
            loss = outputs.norm()
            
            if perturbation.grad is not None:
                perturbation.grad.zero_()

            loss.backward(retain_graph=True)

            perturbation = perturbation + self.alpha * perturbation.grad.sign()
            perturbation = torch.clamp(perturbation, -self.epsilon, self.epsilon).detach()
            perturbation.requires_grad_(True)
            self.model.bert.zero_grad()

        return input_embeddings + perturbation
    
    def forward(
        self, 
        logits: torch.Tensor, 
        input_ids: list[torch.Tensor], 
        attention_masks: list[torch.Tensor], 
        layers: list,
        classifier: bool=True,
    ) -> torch.Tensor:
        if not isinstance(input_ids, list):
            input_ids = [input_ids]
        
        if not isinstance(attention_masks, list):
            attention_masks = [attention_masks]

        all_input_ids = input_ids[0]
        all_attention_mask = attention_masks[0]

        for i in range(1, len(input_ids)):
            all_input_ids = torch.cat((all_input_ids, input_ids[i]), dim=1)
            all_attention_mask = torch.cat((all_attention_mask, attention_masks[i]), dim=1)

        embeddings = self.model.bert.embed(all_input_ids)
        perturbed_embeddings = self.perturb(embeddings, all_attention_mask)

        with torch.no_grad():
            perturbed_hidden_state = self.model.bert.encode(perturbed_embeddings, all_attention_mask)
            perturbed_cls = self.model.bert.pooler_dense(perturbed_hidden_state[:, 0])
            perturbed_cls = self.model.bert.pooler_af(perturbed_cls)
            perturbed_logits = perturbed_cls

            for layer in layers:
                perturbed_logits = layer(perturbed_logits)
        
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