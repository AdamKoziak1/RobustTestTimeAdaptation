import math
import random
from copy import deepcopy
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import torchvision.transforms as tvT
import torchvision.transforms.functional as tvF
from torchvision.models.feature_extraction import create_feature_extractor
from torch.distributions import Beta
from utils.svd import SVDDrop2D
from utils.fft import FFTDrop2D
from utils.image_ops import GaussianBlur2D
from utils.safer_view import SAFERViewModule
from utils.tesla_aug.optaug import OptAug
from utils.tesla_losses import EntropyClassMarginals

@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


VIEW_POOL_STRATEGIES = {"mean", "worst", "entropy", "top1", "cc", "cc_drop"}
VIEW_POOL_STRATEGIES_WITH_MATCHING = VIEW_POOL_STRATEGIES | {"matching"}


def _resolve_input_stats(
    mean: Optional[Sequence[float]],
    std: Optional[Sequence[float]],
    modules: Sequence[nn.Module],
) -> Tuple[torch.Tensor, torch.Tensor]:
    if mean is not None and std is not None:
        mean_t = torch.tensor(mean, dtype=torch.float32).view(1, -1, 1, 1)
        std_t = torch.tensor(std, dtype=torch.float32).view(1, -1, 1, 1)
        return mean_t, std_t
    for module in modules:
        if hasattr(module, "_in_mean") and hasattr(module, "_in_std"):
            mean_t = module._in_mean.detach().clone().to(dtype=torch.float32)
            std_t = module._in_std.detach().clone().to(dtype=torch.float32)
            return mean_t, std_t
    default_mean = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(1, -1, 1, 1)
    default_std = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(1, -1, 1, 1)
    return default_mean, default_std


def _normalize_input(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (x - mean) / std


def _denormalize_input(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return x * std + mean


def _detect_normalized_input(x: torch.Tensor, override: Optional[bool] = None) -> bool:
    if override is not None:
        return override
    x_min = x.amin().item()
    x_max = x.amax().item()
    return (x_min < -0.05) or (x_max > 1.05)


def _maybe_denormalize(
    x: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    normalized: bool,
) -> tuple[torch.Tensor, bool]:
    if normalized:
        return _denormalize_input(x, mean, std), True
    return x, False


def _maybe_normalize(
    x: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    was_normalized: bool,
) -> torch.Tensor:
    if was_normalized:
        return _normalize_input(x, mean, std)
    return x


def _normalize_views(views: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    b, v, c, h, w = views.shape
    flat = views.reshape(-1, c, h, w)
    flat = _normalize_input(flat, mean, std)
    return flat.reshape(b, v, c, h, w)


def collect_params(model):
    """Collect the affine scale + shift parameters from batch norms.
    Walk the model's modules and collect all batch normalization parameters.
    Return the parameters and their names.
    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np}")
    return params, names


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def configure_model(model):
    """Configure model for use with tent."""
    # train mode, because tent optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what tent updates
    model.requires_grad_(False)
    # configure norm for tent updates: enable grad + force batch statisics
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            # force use of batch stats in train and eval modes
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
    return model


def check_model(model):
    """Check model for compatability with tent."""
    is_training = model.training
    assert is_training, "tent needs train mode: call model.train()"
    param_grads = [p.requires_grad for p in model.parameters()]
    has_any_params = any(param_grads)
    has_all_params = all(param_grads)
    assert has_any_params, "tent needs params to update: " \
                           "check which require grad"
    assert not has_all_params, "tent should not update all params: " \
                               "check which require grad"
    has_bn = any([isinstance(m, nn.BatchNorm2d) for m in model.modules()])
    assert has_bn, "tent needs normalization for its optimization"


def adversarial_weight_perturb_predict(
        model,
        images, 
        logits,
        perturb_init_scale=0.01,
        perturb_grad_scale=0.01
    ):
    fc_layer = model.classifier.fc
    feature = model.featurizer(images).detach()
    #----- step 1: generate random perturbation -----#
    weight = fc_layer.weight.data
    # generate random perturbation
    delta = torch.randn(weight.shape).to(logits.device)
    # normalize to unit ball
    delta = delta.div(torch.norm(delta, p=2, dim=1, keepdim=True) + 1e-8)
    # require grad
    delta.requires_grad = True
    # not perturb bias
    bias = fc_layer.bias.data

    #----- step 2: forward with perturbation -----#
    logits_perturb = F.linear(feature, weight + perturb_init_scale * delta, bias)
    # calculate KL div loss
    loss_kl = F.kl_div(F.log_softmax(logits_perturb, dim=1), F.softmax(logits, dim=1), reduction='batchmean')
    loss_kl.backward()
    
    #----- step 3: forward with new perturbation -----#
    grad = delta.grad
    grad = grad.div(torch.norm(grad, p=2, dim=1, keepdim=True) + 1e-8)
    logits_perturb = F.linear(feature, weight + perturb_grad_scale * grad, bias)

    return logits_perturb.detach()


class ERM(nn.Module):
    def __init__(self,model):
        super().__init__()
        self.model = model.eval()
        self.featurizer = model.featurizer
        self.classifier = model.classifier
        
    @torch.no_grad()
    def forward(self, x):
        outputs = self.model.predict(x)
        return outputs
        

class BN(nn.Module):
    def __init__(self, model, steps=1, episodic=False):
        super().__init__()
        self.model = model
        self.steps = steps
        self.episodic = episodic
        assert self.steps>=0, 'steps must be non-negative'
        if self.steps==0:
            self.model.eval()
    
    @torch.no_grad()
    def forward(self, x):
        if self.steps>0:
            for _ in range(self.steps):
                outputs = self.model.predict(x)
        else:
            outputs = self.model.predict(x)
        return outputs


class Tent(nn.Module):
    """
    ICLR,2021
    Tent adapts a model by entropy minimization during testing.
    Once tented, a model adapts itself by updating on every forward.
    """
    def __init__(self, model, optimizer, steps=1, episodic=False):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "tent requires >= 1 step(s) to forward and update"
        self.episodic = episodic

        # note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)

    def forward(self, x):
        if self.episodic:
            self.reset()

        for _ in range(self.steps):
            outputs = self.forward_and_adapt(x, self.model, self.optimizer)

        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)
    
    @torch.enable_grad()
    def forward_and_adapt(self,x, model, optimizer):
        """Forward and adapt model on batch of data.
        Measure entropy of the model prediction, take gradients, and update params.
        """
        # forward
        outputs = model.predict(x)
        # adapt
        loss = softmax_entropy(outputs).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()        
        return outputs


class SAFERPooledPredictor(nn.Module):
    def __init__(self, model: nn.Module, view_module: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.view_module = view_module

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.predict(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        defense = getattr(self.model, "defense", None)
        if isinstance(defense, nn.Module):
            x = defense(x)
        view_out = self.view_module(x, self.model)
        pooled_prob = view_out.pooled_prob.clamp_min(1e-6)
        return pooled_prob.log()

    def __getattr__(self, name: str):
        if name in {"model", "view_module"}:
            return super().__getattr__(name)
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


class PseudoLabel(nn.Module):
    def __init__(self, model, optimizer, beta=0.9,steps=1, episodic=False):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "requires >= 1 step(s) to forward and update"
        self.episodic = episodic
        self.beta = beta  #threshold for selecting pseudo labels
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)

    def forward(self, x):
        if self.episodic:
            self.reset()

        for _ in range(self.steps):
            outputs = self.forward_and_adapt(x, self.model, self.optimizer)

        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)
    
    @torch.enable_grad()
    def forward_and_adapt(self,x, model, optimizer):
        # forward
        outputs = model.predict(x)
        # adapt        
        scores = F.softmax(outputs,1)
        py,y_prime = torch.max(scores,1)
        mask = py > self.beta
        loss = F.cross_entropy(outputs[mask],y_prime[mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return outputs


class SHOTIM(nn.Module):
    """
    SHOT-IM ,ICML 2020
    """
    def __init__(self, model, optimizer, steps=1, episodic=False):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "requires >= 1 step(s) to forward and update"
        self.episodic = episodic

        # note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)

    def forward(self, x):
        if self.episodic:
            self.reset()

        for _ in range(self.steps):
            outputs = self.forward_and_adapt(x, self.model, self.optimizer)

        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)
    
    @torch.enable_grad()
    def forward_and_adapt(self,x, model, optimizer):
        # forward
        outputs = model.predict(x)
        # adapt
        ent_loss = softmax_entropy(outputs).mean(0)
        softmax_out = F.softmax(outputs, dim=1)
        msoftmax = softmax_out.mean(dim=0)
        div_loss = torch.sum(msoftmax * torch.log(msoftmax + 1e-5))
        loss = ent_loss + div_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return outputs
        
        
class T3A(nn.Module):
    """
    T3A, NeurIPS 2021
    """
    def __init__(self,model,filter_K=100,steps=1,episodic=False, view_module: Optional[nn.Module] = None):
        super().__init__()
        self.model = model.eval()
        self.featurizer = model.featurizer
        self.classifier = model.classifier
        self.steps = steps
        assert steps > 0, "requires >= 1 step(s) to forward and update"
        self.episodic = episodic
        self.filter_K = filter_K
        self.view_module = view_module
        
        warmup_supports = self.classifier.fc.weight.data
        self.num_classes = warmup_supports.size()[0]
        self.warmup_supports = warmup_supports
        warmup_prob = self.classifier(self.warmup_supports)
        self.warmup_ent = softmax_entropy(warmup_prob)
        self.warmup_labels = F.one_hot(warmup_prob.argmax(1), num_classes=self.num_classes).float()
        self.supports = self.warmup_supports.data
        self.labels = self.warmup_labels.data
        self.ent = self.warmup_ent.data
             
    @torch.no_grad() 
    def forward(self,x):
        if self.view_module is None:
            z = self.featurizer(x)
            p = self.classifier(z)
        else:
            defense = getattr(self.model, "defense", None)
            if isinstance(defense, nn.Module):
                x = defense(x)
            view_out = self.view_module(x, self.model)
            z = _pool_features(view_out.features, view_out.pooled_weights)
            pooled_prob = view_out.pooled_prob
            p = pooled_prob.clamp_min(1e-6).log()
        yhat = torch.nn.functional.one_hot(p.argmax(1), num_classes=self.num_classes).float()
        ent = softmax_entropy(p)

        self.supports = self.supports.to(z.device)
        self.labels = self.labels.to(z.device)
        self.ent = self.ent.to(z.device)
        self.supports = torch.cat([self.supports,z])
        self.labels = torch.cat([self.labels,yhat])
        self.ent = torch.cat([self.ent,ent])
        
        supports, labels = self.select_supports()
        supports = torch.nn.functional.normalize(supports, dim=1)
        weights = (supports.T @ (labels))
        
        return z @ torch.nn.functional.normalize(weights, dim=0)

    def select_supports(self):
        ent_s = self.ent
        y_hat = self.labels.argmax(dim=1).long()
        filter_K = self.filter_K
        if filter_K == -1:
            indices = torch.LongTensor(list(range(len(ent_s))))

        indices = []
        indices1 = torch.LongTensor(list(range(len(ent_s)))).cuda()
        for i in range(self.num_classes):
            _, indices2 = torch.sort(ent_s[y_hat == i])
            indices.append(indices1[y_hat==i][indices2][:filter_K])
        indices = torch.cat(indices)

        self.supports = self.supports[indices]
        self.labels = self.labels[indices]
        self.ent = self.ent[indices]
        
        return self.supports, self.labels
        

class TSD(nn.Module):
    """
    Test-time Self-Distillation (TSD)
    CVPR 2023
    """
    def __init__(self,model,optimizer,lam=0,filter_K=100,steps=1,episodic=False, view_module: Optional[nn.Module] = None):
        super().__init__()
        self.model = model
        self.featurizer = model.featurizer
        self.classifier = model.classifier
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "requires >= 1 step(s) to forward and update"
        self.episodic = episodic
        self.filter_K = filter_K
        self.view_module = view_module
        
        warmup_supports = self.classifier.fc.weight.data.detach()
        self.num_classes = warmup_supports.size()[0]
        self.warmup_supports = warmup_supports
        warmup_prob = self.classifier(self.warmup_supports)
        
        self.warmup_ent = softmax_entropy(warmup_prob)
        self.warmup_labels = F.one_hot(warmup_prob.argmax(1), num_classes=self.num_classes).float()
        self.warmup_scores = F.softmax(warmup_prob,1)
                
        self.supports = self.warmup_supports.data
        self.labels = self.warmup_labels.data
        self.ent = self.warmup_ent.data
        self.scores = self.warmup_scores.data
        self.lam = lam
        
        
    def forward(self,x):
        if self.view_module is None:
            z = self.featurizer(x)
            p = self.classifier(z)
            scores = F.softmax(p,1)
        else:
            defense = getattr(self.model, "defense", None)
            if isinstance(defense, nn.Module):
                x = defense(x)
            view_out = self.view_module(x, self.model)
            z = _pool_features(view_out.features, view_out.pooled_weights)
            scores = view_out.pooled_prob
            p = scores.clamp_min(1e-6).log()
                       
        yhat = F.one_hot(p.argmax(1), num_classes=self.num_classes).float()
        ent = softmax_entropy(p)

        with torch.no_grad():
            self.supports = self.supports.to(z.device)
            self.labels = self.labels.to(z.device)
            self.ent = self.ent.to(z.device)
            self.scores = self.scores.to(z.device)
            self.supports = torch.cat([self.supports,z])
            self.labels = torch.cat([self.labels,yhat])
            self.ent = torch.cat([self.ent,ent])
            self.scores = torch.cat([self.scores,scores])
        
            supports, labels = self.select_supports()
            supports = F.normalize(supports, dim=1)
            weights = (supports.T @ (labels))
                
        dist,loss = self.prototype_loss(z,weights.T,scores,use_hard=False)

        loss_local = topk_cluster(z.detach().clone(),supports,self.scores,p,k=3)
        loss += self.lam*loss_local
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return p

    def select_supports(self):
        ent_s = self.ent
        y_hat = self.labels.argmax(dim=1).long()
        filter_K = self.filter_K
        if filter_K == -1:
            indices = torch.LongTensor(list(range(len(ent_s))))

        indices = []
        indices1 = torch.LongTensor(list(range(len(ent_s)))).cuda()
        for i in range(self.num_classes):
            _, indices2 = torch.sort(ent_s[y_hat == i])
            indices.append(indices1[y_hat==i][indices2][:filter_K])
        indices = torch.cat(indices)

        self.supports = self.supports[indices]
        self.labels = self.labels[indices]
        self.ent = self.ent[indices]
        self.scores = self.scores[indices]
        
        return self.supports, self.labels
    
    def prototype_loss(self,z,p,labels=None,use_hard=False,tau=1):
        #z [batch_size,feature_dim]
        #p [num_class,feature_dim]
        #labels [batch_size,]        
        z = F.normalize(z,1)
        p = F.normalize(p,1)
        dist = z @ p.T / tau
        if labels is None:
            _,labels = dist.max(1)
        if use_hard:
            """use hard label for supervision """
            #_,labels = dist.max(1)  #for prototype-based pseudo-label
            labels = labels.argmax(1)  #for logits-based pseudo-label
            loss =  F.cross_entropy(dist,labels)
        else:
            """use soft label for supervision """
            loss = softmax_kl_loss(labels.detach(),dist).sum(1).mean(0)  #detach is **necessary**
            #loss = softmax_kl_loss(dist,labels.detach()).sum(1).mean(0) achieves comparable results
        return dist,loss
        

def topk_labels(feature,supports,scores,k=3):
    feature = F.normalize(feature,1)
    supports = F.normalize(supports,1)
    sim_matrix = feature @ supports.T  #B,M
    _,idx_near = torch.topk(sim_matrix,k,dim=1)  #batch x K
    scores_near = scores[idx_near]  #batch x K x num_class
    soft_labels = torch.mean(scores_near,1)  #batch x num_class
    soft_labels = torch.argmax(soft_labels,1)
    return soft_labels
    

def topk_cluster(feature,supports,scores,p,k=3):
    #p: outputs of model batch x num_class
    feature = F.normalize(feature,1)
    supports = F.normalize(supports,1)
    sim_matrix = feature @ supports.T  #B,M
    topk_sim_matrix,idx_near = torch.topk(sim_matrix,k,dim=1)  #batch x K
    scores_near = scores[idx_near].detach().clone()  #batch x K x num_class
    diff_scores = torch.sum((p.unsqueeze(1) - scores_near)**2,-1)
    
    loss = -1.0* topk_sim_matrix * diff_scores
    return loss.mean()
    
    
def knn_affinity(X,knn):
    #x [N,D]
    N = X.size(0)
    X = F.normalize(X,1)
    dist = torch.norm(X.unsqueeze(0) - X.unsqueeze(1), dim=-1, p=2)  # [N, N]
    n_neighbors = min(knn + 1, N)
    knn_index = dist.topk(n_neighbors, -1, largest=False).indices[:, 1:]  # [N, knn]
    W = torch.zeros(N, N, device=X.device)
    W.scatter_(dim=-1, index=knn_index, value=1.0)
    return W
    
       
def softmax_mse_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)

    mse_loss = (input_softmax-target_softmax)**2
    return mse_loss

    
def softmax_kl_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns KL divergence

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_log_softmax = F.log_softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)

    kl_div = F.kl_div(input_log_softmax, target_softmax, reduction='none')
    return kl_div        
        

def get_distances(X, Y, dist_type="cosine"):
    """
    Args:
        X: (N, D) tensor
        Y: (M, D) tensor
    """
    if dist_type == "euclidean":
        distances = torch.cdist(X, Y)
    elif dist_type == "cosine":
        distances = 1 - torch.matmul(F.normalize(X, dim=1), F.normalize(Y, dim=1).T)
    else:
        raise NotImplementedError(f"{dist_type} distance not implemented.")

    return distances


@torch.no_grad()
def soft_k_nearest_neighbors(features, features_bank, probs_bank):
    pred_probs = []
    K = 4
    for feats in features.split(64):
        distances = get_distances(feats, features_bank,"cosine")
        _, idxs = distances.sort()
        idxs = idxs[:, : K]
        # (64, num_nbrs, num_classes), average over dim=1
        probs = probs_bank[idxs, :].mean(1)
        pred_probs.append(probs)
    pred_probs = torch.cat(pred_probs)
    _, pred_labels = pred_probs.max(dim=1)

    return pred_labels, pred_probs

def _normalize(d, norm=2):
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= torch.norm(d_reshaped, norm, dim=1, keepdim=True) + 1e-8
    return d


class TTA3(nn.Module):
    def __init__(self, model, optimizer, steps=1, episodic=False, 
                 lam_flat=1.0, lam_adv=1.0, lam_cr=1.0, lam_pl=1.0, lambda_nuc=0.0, r=4, 
                 cr_type='cosine', cr_start=0, use_mi=False, lam_em=0.0, lam_recon=0.0, 
                 lam_reg=1.0, reg_type='l2logits', ema=0.999, x_lr=1.0/255.0, x_steps=1):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        self.episodic = episodic
        self.lam_flat = lam_flat
        self.lam_adv = lam_adv
        self.lam_cr = lam_cr
        self.lam_pl = lam_pl
        self.r = r
        self.cr_type = cr_type.lower()
        assert self.cr_type in ['cosine', 'l2']
        assert 0 <= cr_start <= 3, \
            f"cr_start ∈ {{0,1,2,3}}, got {cr_start}"
        self.cr_start = cr_start
        self.use_mi = use_mi
        self.beta=0.9
        
        self.lambda_nuc = float(lambda_nuc)
        self.lam_em = float(lam_em)
        self.lam_recon = float(lam_recon)

        # Save initial states for episodic adaptation
        if self.episodic:
            self.model_state, self.optimizer_state = copy_model_and_optimizer(self.model, self.optimizer)

        return_nodes = {
            'layer1': 'feat1',
            'layer2': 'feat2',
            'layer3': 'feat3',
            'layer4': 'feat4',
        }
        self.feat_extractor = create_feature_extractor(
            self.model.featurizer,
            return_nodes,
            tracer_kwargs={"leaf_modules": [SVDDrop2D, FFTDrop2D, GaussianBlur2D]},
        )
        
        self.lam_reg=lam_reg
        self.reg_type=reg_type
        self.ema=ema
        self.x_lr=x_lr
        self.x_steps=x_steps
        self.use_teacher = (x_steps > 0) or (lam_reg > 1e-8)

        self.teacher = None
        if self.use_teacher:
            self.teacher = deepcopy(self.model).eval()
            self.teacher.requires_grad_(False)
            for m in self.teacher.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.track_running_stats = False
                    m.running_mean = None
                    m.running_var  = None
            assert reg_type in ('l2logits', 'klprob')
        

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("Cannot reset without saved model/optimizer state.")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)

    def forward(self, x):
        if self.episodic:
            self.reset()

        for _ in range(self.steps):
            outputs = self.forward_and_adapt(x, self.model, self.optimizer)
        return outputs

    def mi_loss(self, logits, prob):
        # Compute the average entropy over the batch.
        ent = softmax_entropy(logits).mean()
        # Compute the entropy of the mean prediction.
        mean_prob = prob.mean(dim=0)
        overall_ent = -torch.sum(mean_prob * torch.log(mean_prob + 1e-6))
        return ent - overall_ent

# --- Flatness Regularization Loss (L_Flat) ---
    def flat_loss(self, x, model, logits):
        if self.lam_flat <= 1e-8:
            return torch.tensor(0.0, device=x.device)

        logits_perturb = adversarial_weight_perturb_predict(model, x, logits.detach())
        log_prob = F.log_softmax(logits, dim=1)
        prob_perturb = F.softmax(logits_perturb, dim=1)
        return F.kl_div(log_prob, prob_perturb, reduction='batchmean')

# --- Adversarial (Instance-level Flatness) Loss (L_Adv) ---
    def adv_loss(self, x, model, prob):
        if self.lam_adv <= 1e-8:
            return torch.tensor(0.0, device=x.device)

        # Initialize a random perturbation within [-r, r]. 
        epsilon = torch.rand(x.shape).sub(0.5).to(x.device)
        epsilon = _normalize(epsilon)
        bound = self.r/255
        epsilon.requires_grad_()
        pred_hat = model.predict(x + bound * epsilon)
        logp_hat = F.log_softmax(pred_hat, dim=1)
        adv_distance = F.kl_div(logp_hat, prob.detach(), reduction='batchmean')
        adv_distance.backward()
        epsilon = _normalize(epsilon.grad)

        model.zero_grad()
        r_adv = epsilon * bound 
        pred_hat = model.predict(x + r_adv)
        logp_hat = F.log_softmax(pred_hat, dim=1)
        return F.kl_div(logp_hat, prob, reduction='batchmean')
    
# --- Consistency Regularization Loss (L_CR) ---    
    def similarity_matrix(self, x):
        if self.cr_type == "cosine":
            x = F.normalize(x, dim=1)
        return x @ x.t()
    
    def similarity_loss(self, S1, S2):
        if self.cr_type == 'l2': # (eq 7 + 11)
            # Frobenius inner product normalised by L2 norms
            return (S1 * S2).sum() / (
                torch.norm(S1) * torch.norm(S2) + 1e-8
            )
        else:  # 'cosine'  eq  8 + 12
            return torch.norm(S1 - S2, p=2) / S1.numel()
        
    def cr_loss(self, x, prob): 
        l_cr = torch.tensor(0.0, device=x.device)
        if self.lam_cr <= 1e-8:
            return l_cr

        feats = self.feat_extractor(x) 
        feats = [v.flatten(1) for v in feats.values()]
        feats.append(prob)
        feats = feats[self.cr_start:] # keep only from user-chosen layer onwards
        
        S_feat_last = self.similarity_matrix(feats[0])
        for i in range(len(feats)-1):
            S_feat_next = self.similarity_matrix(feats[i+1])
            l_cr += self.similarity_loss(S_feat_last.detach(), S_feat_next)
            S_feat_last = S_feat_next
        return l_cr
    
    def pl_loss(self, logits):   
        if self.lam_pl <= 1e-8:
            return torch.tensor(0.0, device=logits.device)
        scores = F.softmax(logits,1)
        py,y_prime = torch.max(scores,1)
        mask = py > self.beta
        return  F.cross_entropy(logits[mask],y_prime[mask])
    
    def nuclear_losses(self, model, logits):
        if self.lambda_nuc > 0.0 and hasattr(model.featurizer, "pop_nuc_penalty"):
            return model.featurizer.pop_nuc_penalty()
        return torch.tensor(0.0, device=logits.device), torch.tensor(0.0, device=logits.device)

    @torch.no_grad()
    def _ema_update_teacher(self):
        s_params, _ = collect_params(self.model)
        t_params, _ = collect_params(self.teacher)
        for p_t, p_s in zip(t_params, s_params):
            p_t.data.mul_(self.ema).add_(p_s.data, alpha=(1.0 - self.ema))
    
    def _reg_loss(self, logits_s: torch.Tensor, logits_t: torch.Tensor) -> torch.Tensor:
        if self.lam_reg <= 1e-8:
            return torch.tensor(0.0, device=logits_s.device)
        p_t = F.softmax(logits_t.detach(), dim=1)
        if self.reg_type == 'l2logits':
            p_s = F.softmax(logits_s, dim=1)
            return torch.mean((p_s - p_t)**2)
        else:
            # KL(student || teacher) over softmax
            logp_s = F.log_softmax(logits_s, dim=1)
            return F.kl_div(logp_s, p_t, reduction='batchmean')
    
    def base_loss(self, logits_t, probs_t):
        return self.mi_loss(logits_t, probs_t) if self.use_mi else softmax_entropy(logits_t).mean()
    
    def forward_and_adapt(self, x, model, optimizer):
        # 1) Teacher-guided input refinement: x -> x_tilde
        x_tilde = x.detach()
        for _ in range(self.x_steps):
            x_tilde.requires_grad_(True)
            logits_t = self.teacher.predict(x_tilde) if self.use_teacher else self.model.predict(x_tilde) 
            probs_t  = F.softmax(logits_t, dim=1)

            L_t = self.base_loss(logits_t, probs_t) 
            (g_x,)   = torch.autograd.grad(L_t, x_tilde, only_inputs=True)
            with torch.no_grad():
                x_tilde = (x_tilde - self.x_lr * g_x).clamp(0.0, 1.0).detach()

        # 2) Student update on refined input
        logits_s = self.model.predict(x_tilde)
        probs_s  = F.softmax(logits_s, dim=1)

        L_Base = self.base_loss(logits_s, probs_s)
        L_Flat = self.flat_loss(x, model, logits_s)
        L_Adv = self.adv_loss(x, model, probs_s)
        L_CR = self.cr_loss(x, probs_s)
        L_PL = self.pl_loss(logits_s)
        L_NUC, L_RECON = self.nuclear_losses(model, logits_s)

        # Teacher consistency regularization
        L_Reg = torch.tensor(0.0, device=logits_s.device)
        if self.use_teacher and self.lam_reg >= 1e-8:
            with torch.no_grad():
                logits_t_bar = self.teacher.predict(x_tilde)
            L_Reg = self._reg_loss(logits_s, logits_t_bar)

        loss =    (self.lam_em * L_Base) \
                + (self.lam_flat * L_Flat) \
                + (self.lam_adv * L_Adv) \
                + (self.lam_cr * L_CR) \
                + (self.lam_pl * L_PL) \
                + (self.lambda_nuc * L_NUC)  \
                + (self.lam_recon * L_RECON) \
                + (self.lam_reg * L_Reg)
        wandb.log({"Loss": loss.item(),
                   "L_Base": L_Base.item(),
                   "L_Flat": L_Flat.item(),
                   "L_Adv": L_Adv.item(),
                   "L_CR": L_CR.item(),
                   "L_PL": L_PL.item(),
                   "L_NUC": L_NUC.item(),
                   "L_RECON": L_RECON.item(),
                   "L_Reg": L_Reg.item()})
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 3) EMA update of teacher (BN affine only)
        if self.use_teacher:
            with torch.no_grad():
                self._ema_update_teacher()
        return logits_s


def _js_divergence(
    probs: torch.Tensor,
    ref_probs: Optional[torch.Tensor] = None,
    view_weights: Optional[torch.Tensor] = None,
    mode: str = "pooled",
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Jensen–Shannon divergence across augmentation views.

    Args:
        probs: tensor with shape (B, V, K) containing per-view class probabilities.
        ref_probs: optional pooled distribution used as the JS reference.
        view_weights: optional weights used for pooling entropies or weighting pairs.
        mode: 'pooled' (default) or 'pairwise' for full pairwise computation.
    """
    mode = mode.lower()
    if mode not in {"pooled", "pairwise"}:
        raise ValueError(f"Unknown JS computation mode '{mode}'.")

    if mode == "pairwise":
        bsz, num_views, _ = probs.shape
        if num_views < 2:
            return torch.tensor(0.0, device=probs.device, dtype=probs.dtype)
        total = probs.new_tensor(0.0)
        weight_total = probs.new_tensor(0.0)
        for i in range(num_views):
            for j in range(i + 1, num_views):
                p, q = probs[:, i], probs[:, j]
                m = 0.5 * (p + q)
                js_pair = 0.5 * _prob_kl_divergence(p, m, eps=eps) + 0.5 * _prob_kl_divergence(q, m, eps=eps)
                pair_weight = None
                if view_weights is not None:
                    pair_weight = view_weights[:, i] * view_weights[:, j]
                    total += (js_pair * pair_weight).sum()
                    weight_total += pair_weight.sum()
                else:
                    total += js_pair.sum()
                    weight_total += js_pair.numel()
        return total / weight_total.clamp_min(eps)

    if ref_probs is None:
        ref_probs = probs.mean(dim=1)
    entropy_each = -torch.xlogy(probs, probs.clamp_min(eps)).sum(dim=-1)
    entropy_mean = -torch.xlogy(ref_probs, ref_probs.clamp_min(eps)).sum(dim=-1)
    if view_weights is None:
        entropy_each = entropy_each.mean(dim=1)
    else:
        entropy_each = (entropy_each * view_weights).sum(dim=1)
    return (entropy_mean - entropy_each).mean()

def _barlow_twins_loss(
    features: torch.Tensor,
    offdiag_weight: float = 1.0,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Barlow Twins style feature cross-correlation penalty.

    Args:
        features: tensor with shape (B, V, D) where V is the number of views.
    """
    bsz, num_views, dim = features.shape
    if num_views < 2:
        return torch.tensor(0.0, device=features.device, dtype=features.dtype)

    mean = features.mean(dim=0, keepdim=True)
    std  = features.std(dim=0, unbiased=False, keepdim=True)
    z = (features - mean) / (std + eps)

    loss = features.new_tensor(0.0)
    pairs = 0
    eye = torch.eye(dim, device=features.device, dtype=features.dtype)

    for i in range(num_views):
        zi = z[:, i]
        for j in range(i + 1, num_views):
            zj = z[:, j]
            c = (zi.T @ zj) / float(bsz)

            c_diff = (c - eye).pow(2)
            diag_loss = torch.diagonal(c_diff).sum()
            offdiag_loss = c_diff.sum() - diag_loss

            loss = loss + diag_loss + offdiag_weight * offdiag_loss
            pairs += 1

    return loss / pairs


def _barlow_twins_loss_einsum(
    features: torch.Tensor,
    offdiag_weight: float = 1.0,
    eps: float = 1e-12,
) -> torch.Tensor:
    bsz, num_views, dim = features.shape
    if num_views < 2:
        return torch.tensor(0.0, device=features.device, dtype=features.dtype)

    z = features - features.mean(dim=0, keepdim=True)
    norm = torch.sqrt(torch.sum(z ** 2, dim=0) + eps)

    C = torch.einsum('bvi,bwj->vwij', z, z)
    denom = torch.einsum('vi,wj->vwij', norm, norm)
    C = C / (denom + eps)

    loss = features.new_tensor(0.0)
    pairs = 0
    eye = torch.eye(dim, device=features.device, dtype=features.dtype)

    for i in range(num_views):
        for j in range(i + 1, num_views):
            c_ij = C[i, j]
            c_diff = (c_ij - eye).pow(2)
            diag_loss = torch.diagonal(c_diff).sum()
            offdiag_loss = c_diff.sum() - diag_loss
            loss += diag_loss + offdiag_weight * offdiag_loss
            pairs += 1

    return loss / pairs if pairs > 0 else loss


def _prob_kl_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    KL(p || q) for probability tensors with a shared trailing class dimension.
    """
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    return (p * (p.log() - q.log())).sum(dim=-1)


def _weighted_mean(values: torch.Tensor, weights: Optional[torch.Tensor], eps: float = 1e-6) -> torch.Tensor:
    """
    Weighted mean across all elements of `values`. Falls back to the simple mean when
    `weights` is None.
    """
    if weights is None:
        return values.mean()
    weight_sum = weights.sum().clamp_min(eps)
    return (values * weights).sum() / weight_sum


def _confidence_weighted_pseudo_label(
    pooled_prob: torch.Tensor,
    confidence_scale: bool = True,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Confidence-weighted pseudo-label loss over pooled predictions.
    """
    conf, pseudo = torch.max(pooled_prob, dim=-1)
    log_prob = torch.log(pooled_prob.clamp_min(eps))
    ce = F.nll_loss(log_prob, pseudo, reduction="none")
    if confidence_scale:
        ce = conf * ce
    return ce.mean()


def _entropy_minimization_loss(
    mean_prob: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Entropy minimisation over averaged predictions.
    """
    entropy = -(mean_prob * torch.log(mean_prob.clamp_min(eps))).sum(dim=-1)
    return entropy.mean()


def _cross_view_reliability(features: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Estimate per-view reliability from feature cross-correlation diagonals.
    """
    bsz, num_views, _ = features.shape
    if num_views < 2:
        return torch.ones(num_views, device=features.device, dtype=features.dtype)

    z = features - features.mean(dim=0, keepdim=True)
    std = torch.sqrt(torch.mean(z ** 2, dim=0) + eps)
    z = z / (std + eps)

    reliabilities = []
    for i in range(num_views):
        zi = z[:, i]
        pair_scores = []
        for j in range(num_views):
            if i == j:
                continue
            zj = z[:, j]
            diag_corr = (zi * zj).mean(dim=0)
            pair_scores.append(diag_corr.mean())
        reliabilities.append(torch.stack(pair_scores).mean())
    return torch.stack(reliabilities)


def _view_pool_weights(
    probs: torch.Tensor,
    features: Optional[torch.Tensor],
    strategy: str = "mean",
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Compute per-view weights for pooling strategies. The weights are normalised to sum to 1
    across the view dimension.
    """
    bsz, num_views, num_classes = probs.shape
    strategy = strategy.lower()
    if strategy == "mean":
        weights = torch.ones(bsz, num_views, device=probs.device, dtype=probs.dtype)
    elif strategy == "entropy":
        entropy = -(probs * torch.log(probs.clamp_min(eps))).sum(dim=-1)
        weights = 1.0 - entropy / math.log(num_classes)
    elif strategy == "top1":
        weights = probs.max(dim=-1).values
    elif strategy in {"cc", "cc_drop"}:
        if features is None:
            raise ValueError("Feature tensor is required for cc-based pooling.")
        rel = _cross_view_reliability(features, eps=eps).clamp_min(0.0)
        weights = rel.unsqueeze(0).expand(bsz, -1)
    elif strategy == "worst":
        entropy = -(probs * torch.log(probs.clamp_min(eps))).sum(dim=-1)
        worst_idx = entropy.argmax(dim=1)
        weights = torch.zeros(bsz, num_views, device=probs.device, dtype=probs.dtype)
        weights.scatter_(1, worst_idx.unsqueeze(1), 1.0)
    else:
        raise ValueError(f"Unknown SAFER view pooling strategy '{strategy}'.")

    weights = weights.clamp_min(0.0)
    if strategy == "cc_drop":
        min_idx = weights.argmin(dim=1, keepdim=True)
        weights = weights.clone()
        weights.scatter_(1, min_idx, 0.0)

    zero_mask = weights.sum(dim=1, keepdim=True) <= eps
    weights = torch.where(zero_mask, torch.ones_like(weights), weights)
    weight_sum = weights.sum(dim=1, keepdim=True).clamp_min(eps)
    return weights / weight_sum


def _pool_features(
    features: torch.Tensor,
    view_weights: Optional[torch.Tensor],
    eps: float = 1e-6,
) -> torch.Tensor:
    if view_weights is None:
        return features.mean(dim=1)
    weight_sum = view_weights.sum(dim=1, keepdim=True).clamp_min(eps)
    return (features * view_weights.unsqueeze(-1)).sum(dim=1) / weight_sum


def _aggregate_view_probs(
    probs: torch.Tensor,
    features: torch.Tensor,
    strategy: str = "mean",
    use_weights: bool = True,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pool per-view probabilities using a chosen weighting strategy.
    Returns pooled probabilities and normalised view weights (B, V).
    """
    bsz, num_views, num_classes = probs.shape
    if num_views == 1:
        weights = torch.ones(bsz, 1, device=probs.device, dtype=probs.dtype)
        return probs[:, 0], weights

    if use_weights:
        norm_weights = _view_pool_weights(probs, features, strategy=strategy, eps=eps)
    else:
        norm_weights = torch.ones(bsz, num_views, device=probs.device, dtype=probs.dtype)
        norm_weights = norm_weights / norm_weights.sum(dim=1, keepdim=True)
    pooled = (probs * norm_weights.unsqueeze(-1)).sum(dim=1)
    return pooled, norm_weights


def _barlow_twins_two_view(
    z_a: torch.Tensor,
    z_b: torch.Tensor,
    offdiag_weight: float = 1.0,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Two-view Barlow Twins loss used for pooled reference comparisons.
    """
    assert z_a.ndim == 2 and z_b.ndim == 2
    assert z_a.shape == z_b.shape
    bsz, dim = z_a.shape

    z_a = z_a - z_a.mean(dim=0, keepdim=True)
    z_b = z_b - z_b.mean(dim=0, keepdim=True)

    norm_a = torch.sqrt(torch.sum(z_a ** 2, dim=0) + eps)
    norm_b = torch.sqrt(torch.sum(z_b ** 2, dim=0) + eps)

    c = (z_a.T @ z_b) / (norm_a.unsqueeze(1) * norm_b.unsqueeze(0) + eps)
    eye = torch.eye(dim, device=z_a.device, dtype=z_a.dtype)
    c_diff = (c - eye).pow(2)
    diag_loss = torch.diagonal(c_diff).sum()
    offdiag_loss = c_diff.sum() - diag_loss
    return diag_loss + offdiag_weight * offdiag_loss


def _barlow_twins_against_pooled(
    features: torch.Tensor,
    pooled: torch.Tensor,
    offdiag_weight: float = 1.0,
    eps: float = 1e-12,
    view_weights: Optional[torch.Tensor] = None,
    impl: str = "fast",
) -> torch.Tensor:
    """
    Cross-correlation between each view and pooled features instead of pairwise views.
    """
    bsz, num_views, _ = features.shape
    losses = []
    impl_fn = _barlow_twins_loss if impl == "fast" else _barlow_twins_loss_einsum
    for i in range(num_views):
        pair_feats = torch.stack([features[:, i], pooled], dim=1)
        losses.append(impl_fn(pair_feats, offdiag_weight=offdiag_weight, eps=eps))
    view_weights_reduced: Optional[torch.Tensor] = None
    if view_weights is not None:
        view_weights_reduced = view_weights.mean(dim=0)
    return _weighted_mean(torch.stack(losses), view_weights_reduced)


def _weighted_pseudo_label_loss(
    logits: torch.Tensor,
    agg_prob: torch.Tensor,
    view_weights: torch.Tensor,
    confidence_scale: bool = True,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Cross-entropy to a pooled pseudo-label, weighted across views.
    """
    bsz, num_views, num_classes = logits.shape
    pseudo = agg_prob.argmax(dim=-1)
    ce = F.cross_entropy(
        logits.view(-1, num_classes),
        pseudo.repeat_interleave(num_views),
        reduction="none",
    ).view(bsz, num_views)
    if view_weights is None:
        weighted = ce.mean(dim=1)
    else:
        weighted = (ce * view_weights).sum(dim=1)
    if confidence_scale:
        conf = agg_prob.max(dim=-1).values
        weighted = weighted * conf
    return weighted.mean()


def _soft_cross_entropy(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return -torch.sum(p * torch.log(q + eps), dim=-1)


class TeSLA(nn.Module):
    """
    TeSLA: Test-Time Self-Learning with automatic adversarial augmentation.
    """

    def __init__(
        self,
        model,
        optimizer,
        steps: int = 1,
        episodic: bool = False,
        sub_policy_dim: int = 2,
        aug_mult: int = 1,
        aug_mult_easy: int = 4,
        hard_augment: str = "optimal",
        lmb_kl: float = 1.0,
        lmb_norm: float = 1.0,
        ema_momentum: float = 0.9,
        no_kl_hard: bool = False,
        nn_queue_size: int = 0,
        n_neigh: int = 4,
        pl_ce: bool = False,
        pl_fce: bool = False,
        mean: Optional[Sequence[float]] = None,
        std: Optional[Sequence[float]] = None,
        input_is_normalized: Optional[bool] = None,
        view_pool: str = "mean",
        js_weight: float = 0.0,
    ):
        super().__init__()
        assert steps > 0, "TeSLA requires >= 1 step(s) to forward and update"
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        self.episodic = episodic
        self.sub_policy_dim = sub_policy_dim
        self.aug_mult = int(aug_mult)
        self.aug_mult_easy = int(aug_mult_easy)
        self.hard_augment = hard_augment.lower()
        self.lmb_kl = lmb_kl
        self.lmb_norm = lmb_norm
        self.ema_momentum = ema_momentum
        self.no_kl_hard = bool(no_kl_hard)
        self.pl_ce = bool(pl_ce)
        self.pl_fce = bool(pl_fce)
        self.input_is_normalized = input_is_normalized
        view_pool = view_pool.lower()
        if view_pool not in VIEW_POOL_STRATEGIES:
            raise ValueError(
                f"Unknown view pooling strategy '{view_pool}'. Expected one of {sorted(VIEW_POOL_STRATEGIES)}."
            )
        self.view_pool = view_pool
        self.js_weight = js_weight
        if self.hard_augment not in {"optimal", "aa", "randaugment"}:
            raise ValueError("hard_augment must be one of ['optimal', 'aa', 'randaugment']")
        if self.aug_mult < 0 or self.aug_mult_easy < 0:
            raise ValueError("aug_mult and aug_mult_easy must be non-negative")

        self.featurizer = model.featurizer
        self.classifier = model.classifier
        self.softmax = nn.Softmax(dim=-1)
        self.criterian_cm = EntropyClassMarginals()

        self.num_classes = self._infer_num_classes()
        self.feat_dim = self._infer_feat_dim()
        self.nn_queue_size = int(nn_queue_size)
        self.n_neigh = int(n_neigh)
        self.feats_nn_queue = None
        self.prob_nn_queue = None

        self.teacher = deepcopy(model).eval()
        self.teacher.requires_grad_(False)

        mean_t, std_t = _resolve_input_stats(mean, std, (self.featurizer, self.model))
        self.register_buffer("norm_mean", mean_t)
        self.register_buffer("norm_std", std_t)

        self.normalize_fn = lambda x: _normalize_input(x, self.norm_mean, self.norm_std)
        self.denormalize_fn = lambda x: _denormalize_input(x, self.norm_mean, self.norm_std)

        self.hard_opt_aug = None
        if (not self.no_kl_hard) and self.aug_mult > 0 and self.hard_augment == "optimal":
            self.hard_opt_aug = OptAug(
                self.teacher,
                self.sub_policy_dim,
                self.aug_mult,
                "Hard",
                self.normalize_fn,
                self.denormalize_fn,
                self.lmb_norm,
            )
        self.hard_aug_transform = None
        if self.hard_augment in {"aa", "randaugment"}:
            policy = tvT.AutoAugment() if self.hard_augment == "aa" else tvT.RandAugment()
            self.hard_aug_transform = tvT.Compose(
                [
                    tvT.Lambda(lambda img: (img * 255.0).to(torch.uint8)),
                    tvT.RandomHorizontalFlip(),
                    policy,
                    tvT.Lambda(lambda img: (img.to(torch.float32) / 255.0)),
                ]
            )

        if self.episodic:
            self.model_state, self.optimizer_state = copy_model_and_optimizer(self.model, self.optimizer)
            self.teacher_state = deepcopy(self.teacher.state_dict())
            if self.hard_opt_aug is not None:
                self.policy_state = deepcopy(self.hard_opt_aug.policy_predictor.state_dict())
                self.policy_optim_state = deepcopy(self.hard_opt_aug.optimizer_policy.state_dict())
            else:
                self.policy_state = None
                self.policy_optim_state = None

    def _infer_num_classes(self) -> int:
        if hasattr(self.classifier, "fc"):
            return self.classifier.fc.weight.size(0)
        if hasattr(self.classifier, "weight"):
            return self.classifier.weight.size(0)
        raise ValueError("Unable to infer num_classes from classifier.")

    def _infer_feat_dim(self) -> int:
        if hasattr(self.classifier, "fc"):
            return self.classifier.fc.weight.size(1)
        if hasattr(self.classifier, "weight"):
            return self.classifier.weight.size(1)
        raise ValueError("Unable to infer feature dim from classifier.")

    def _apply_easy_augment(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        out = []
        for idx in range(b):
            img = x[idx]
            i, j, h_crop, w_crop = tvT.RandomResizedCrop.get_params(
                img, scale=(0.8, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)
            )
            img = tvF.resized_crop(
                img,
                i,
                j,
                h_crop,
                w_crop,
                (h, w),
                interpolation=tvF.InterpolationMode.BILINEAR,
            )
            if torch.rand((), device=img.device) < 0.5:
                img = tvF.hflip(img)
            out.append(img)
        return torch.stack(out, dim=0)

    def _sample_easy_views(self, x: torch.Tensor, normalized: bool) -> Optional[torch.Tensor]:
        if self.aug_mult_easy <= 0:
            return None
        x_raw, was_norm = _maybe_denormalize(x, self.norm_mean, self.norm_std, normalized)
        views = []
        for _ in range(self.aug_mult_easy):
            view = self._apply_easy_augment(x_raw)
            views.append(view.clamp(0.0, 1.0))
        views = torch.stack(views, dim=1)
        return _normalize_views(views, self.norm_mean, self.norm_std) if was_norm else views

    def _apply_hard_augment(self, x: torch.Tensor, normalized: bool) -> torch.Tensor:
        if self.hard_aug_transform is None:
            return x
        x_raw, was_norm = _maybe_denormalize(x, self.norm_mean, self.norm_std, normalized)
        x_cpu = x_raw.detach().cpu()
        out = [self.hard_aug_transform(x_cpu[i]) for i in range(x_cpu.size(0))]
        out = torch.stack(out, dim=0).to(device=x.device, dtype=x.dtype)
        out = out.clamp(0.0, 1.0)
        return _maybe_normalize(out, self.norm_mean, self.norm_std, was_norm)

    def _ensure_nn_queue(self, device):
        if self.nn_queue_size <= 0:
            return
        if self.feats_nn_queue is None or next(iter(self.feats_nn_queue.values())).device != device:
            self.feats_nn_queue = {
                k: torch.empty((0, self.feat_dim), device=device) for k in range(self.num_classes)
            }
            self.prob_nn_queue = {
                k: torch.empty((0, self.num_classes), device=device) for k in range(self.num_classes)
            }

    @torch.no_grad()
    def update_nearest_neighbours(self, feats, labels):
        hard_labels = torch.argmax(labels, dim=-1)
        for l in range(self.num_classes):
            mask = hard_labels == l
            if mask.sum() == 0:
                continue
            curr_feats = feats[mask][: self.nn_queue_size]
            curr_labels = labels[mask][: self.nn_queue_size]
            feats_cat = torch.cat([self.feats_nn_queue[l], curr_feats], dim=0)
            labels_cat = torch.cat([self.prob_nn_queue[l], curr_labels], dim=0)
            if feats_cat.size(0) > self.nn_queue_size:
                feats_cat = feats_cat[-self.nn_queue_size :]
                labels_cat = labels_cat[-self.nn_queue_size :]
            self.feats_nn_queue[l] = feats_cat
            self.prob_nn_queue[l] = labels_cat

    @torch.no_grad()
    def get_pseudo_labels_nearest_neighbours(self, feats):
        all_feats = torch.cat(list(self.feats_nn_queue.values()), dim=0)
        all_probs = torch.cat(list(self.prob_nn_queue.values()), dim=0)
        if all_feats.numel() == 0:
            return None, None
        k = min(self.n_neigh, all_feats.size(0))
        norm_feats = F.normalize(feats, dim=-1)
        norm_all = F.normalize(all_feats, dim=-1)
        cosine_sim = torch.einsum("bd,cd->bc", norm_feats, norm_all)
        _, idx_neighbours = torch.topk(cosine_sim, k=k, dim=-1)
        pred_top_k = all_probs[idx_neighbours]
        soft_voting = torch.mean(pred_top_k, dim=1)
        pseudo_label = torch.argmax(soft_voting, dim=-1)
        return pseudo_label, soft_voting

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("Cannot reset without stored model/optimizer state.")
        load_model_and_optimizer(self.model, self.optimizer, self.model_state, self.optimizer_state)
        self.teacher.load_state_dict(self.teacher_state)
        if self.hard_opt_aug is not None and self.policy_state is not None:
            self.hard_opt_aug.policy_predictor.load_state_dict(self.policy_state)
            self.hard_opt_aug.optimizer_policy.load_state_dict(self.policy_optim_state)
        self.feats_nn_queue = None
        self.prob_nn_queue = None

    def forward(self, x):
        if self.episodic:
            self.reset()
        outputs = None
        for _ in range(self.steps):
            outputs = self.forward_and_adapt(x, self.model, self.optimizer)
        return outputs

    @torch.no_grad()
    def _ema_update_teacher(self):
        for p_t, p_s in zip(self.teacher.parameters(), self.model.parameters()):
            p_t.data.mul_(self.ema_momentum).add_(p_s.data, alpha=1.0 - self.ema_momentum)

    @torch.enable_grad()
    def forward_and_adapt(self, x, model, optimizer):
        normalized = _detect_normalized_input(x, self.input_is_normalized)
        easy_views = self._sample_easy_views(x, normalized)
        if easy_views is None:
            all_views = x.unsqueeze(1)
        else:
            all_views = torch.cat([x.unsqueeze(1), easy_views], dim=1)

        num_views = all_views.size(1)
        if num_views > 1:
            view_idx = random.randrange(num_views)
            batch_x = all_views[:, view_idx]
        else:
            batch_x = x

        x_aug_hard = None
        if not self.no_kl_hard:
            if self.hard_augment == "optimal" and self.hard_opt_aug is not None:
                self.hard_opt_aug.optimize(batch_x)
                x_aug_hard = self.hard_opt_aug.sample(batch_x)
            elif self.hard_augment in {"aa", "randaugment"}:
                x_aug_hard = self._apply_hard_augment(batch_x, normalized).unsqueeze(1)

        view_weights = None
        with torch.no_grad():
            scores_ema_easy = []
            feats_ema_easy = []
            for i in range(num_views):
                feats_view = self.teacher.featurizer(all_views[:, i])
                logits_view = self.teacher.classifier(feats_view)
                scores_ema_easy.append(self.softmax(logits_view))
                feat_vec = feats_view
                if feat_vec.dim() > 2:
                    feat_vec = torch.flatten(feat_vec, start_dim=1)
                feats_ema_easy.append(feat_vec)

            scores_ema_easy = torch.stack(scores_ema_easy, dim=1)
            feats_ema_easy = torch.stack(feats_ema_easy, dim=1)
            if num_views > 1 and self.view_pool != "mean":
                view_weights = _view_pool_weights(
                    probs=scores_ema_easy,
                    features=feats_ema_easy,
                    strategy=self.view_pool,
                )
            if view_weights is None:
                feats_pooled = feats_ema_easy.mean(dim=1)
                soft_pseudo_labels = scores_ema_easy.mean(dim=1)
            else:
                feats_pooled = _pool_features(feats_ema_easy, view_weights)
                soft_pseudo_labels = (scores_ema_easy * view_weights.unsqueeze(-1)).sum(dim=1)
            if self.nn_queue_size > 0 and self.n_neigh > 0:
                self._ensure_nn_queue(feats_pooled.device)
                self.update_nearest_neighbours(feats_pooled, soft_pseudo_labels)
                _, refined = self.get_pseudo_labels_nearest_neighbours(feats_pooled)
                if refined is not None and refined.numel() > 0:
                    soft_pseudo_labels = refined

        logits_student = model.predict(batch_x)
        scores_student = self.softmax(logits_student)

        js_loss = torch.tensor(0.0, device=x.device)
        if self.js_weight > 1e-8 and num_views > 1:
            bsz, _, c, h, w = all_views.shape
            flat_views = all_views.view(bsz * num_views, c, h, w)
            logits_views = model.predict(flat_views)
            probs_views = F.softmax(logits_views, dim=-1).view(bsz, num_views, -1)
            if view_weights is None:
                js_ref = probs_views.mean(dim=1)
                js_weights = None
            else:
                js_ref = (probs_views * view_weights.unsqueeze(-1)).sum(dim=1)
                js_weights = view_weights
            js_loss = _js_divergence(
                probs_views,
                ref_probs=js_ref,
                view_weights=js_weights,
                mode="pooled",
            )

        if self.pl_ce:
            loss_teach = _soft_cross_entropy(soft_pseudo_labels, scores_student).mean()
            loss_cm = torch.tensor(0.0, device=x.device)
        elif self.pl_fce:
            loss_teach = _soft_cross_entropy(scores_student, soft_pseudo_labels).mean()
            loss_cm = torch.tensor(0.0, device=x.device)
        else:
            loss_cm = self.criterian_cm(scores_student)
            loss_teach = _soft_cross_entropy(scores_student, soft_pseudo_labels).mean()

        loss = loss_cm + loss_teach + (self.js_weight * js_loss)

        if x_aug_hard is not None and self.lmb_kl > 0:
            loss_hard = torch.tensor(0.0, device=x.device)
            for j in range(x_aug_hard.size(1)):
                logits_hard = model.predict(x_aug_hard[:, j])
                scores_hard = self.softmax(logits_hard)
                loss_hard += F.kl_div(
                    torch.log(scores_hard + 1e-8),
                    soft_pseudo_labels,
                    reduction="batchmean",
                )
            loss = loss + self.lmb_kl * loss_hard

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        self._ema_update_teacher()

        return logits_student


class SAFER(nn.Module):
    """
    SAFER:Stochastically Augmented Feature Ensemble for Robust Test-Time Adaptation.

    Generates multiple stochastic augmentation views per input and minimises
    cross-view inconsistency via JS divergence and Barlow Twins penalties.
    """

    def __init__(
        self,
        model,
        optimizer,
        steps: int = 1,
        episodic: bool = False,
        num_aug_views: int = 4,
        include_original: bool = True,
        aug_prob: float = 0.7,
        aug_max_ops: Optional[int] = 4,
        augmentations: Optional[Sequence[str]] = None,
        force_noise_first: bool = False,
        require_freq_or_blur: bool = False,
        fixed_op: Optional[str] = None,
        fixed_blur_kernel: Optional[int] = None,
        fixed_blur_sigma: Optional[float] = None,
        fixed_fft_keep_ratio: Optional[float] = None,
        allow_noise: bool = True,
        noise_std: Optional[float] = None,
        js_weight: float = 1.0,
        cc_weight: float = 1.0,
        offdiag_weight: float = 1.0,
        feature_normalize: bool = False,
        aug_seed: Optional[int] = None,
        sample_params_per_image: bool = False,
        sup_mode: str = "none",
        sup_weight: float = 0.0,
        class_marginal_weight: float = 0.0,
        cc_impl: str = "fast",
        sup_view_pool: str = "mean",
        sup_pl_weighted: bool = False,
        sup_confidence_scale: bool = True,
        js_view_pool: str = "matching",
        js_mode: str = "pooled",
        view_weighting: bool = True,
        tta_loss: str = "none",
        tta_weight: float = 0.0,
        tta_target: str = "views",
        tta_view_pool: str = "matching",
        cc_mode: str = "pairwise",
        cc_view_pool: str = "matching",
        mean: Optional[Sequence[float]] = None,
        std: Optional[Sequence[float]] = None,
        input_is_normalized: Optional[bool] = None,
    ):
        super().__init__()
        assert steps > 0, "SAFER requires at least one update step"
        assert num_aug_views >= 0, "num_aug_views must be ≥ 0"
        if not include_original and num_aug_views == 0:
            raise ValueError("Need at least one view (original or augmented).")

        self.model = model
        self.classifier = model.classifier
        self.featurizer = model.featurizer
        self.optimizer = optimizer
        self.steps = steps
        self.episodic = episodic
        self.include_original = include_original
        self.num_aug_views = num_aug_views
        self.js_weight = js_weight
        self.cc_weight = cc_weight
        self.offdiag_weight = offdiag_weight
        self.feature_normalize = feature_normalize
        self.cm_weight = class_marginal_weight
        self.use_view_weights = bool(view_weighting)
        sup_mode = sup_mode.lower()
        if sup_mode not in {"none", "pl", "em"}:
            raise ValueError(f"Unknown supervision mode: {sup_mode}")
        self.sup_mode = sup_mode
        self.sup_weight = sup_weight
        self.class_marginal = EntropyClassMarginals()
        self.input_is_normalized = input_is_normalized
        sup_view_pool = sup_view_pool.lower()
        valid_pools = VIEW_POOL_STRATEGIES
        valid_pools_with_match = VIEW_POOL_STRATEGIES_WITH_MATCHING
        if sup_view_pool not in valid_pools:
            raise ValueError(
                f"Unknown view pooling strategy '{sup_view_pool}'. Expected one of {sorted(valid_pools)}."
            )
        self.primary_view_pool = sup_view_pool
        self.sup_pl_weighted = sup_pl_weighted
        self.sup_confidence_scale = sup_confidence_scale
        cc_impl = cc_impl.lower()
        self.cc_impl = cc_impl
        js_mode = js_mode.lower()
        if js_mode not in {"pooled", "pairwise"}:
            raise ValueError("js_mode must be 'pooled' or 'pairwise'.")
        self.js_mode = js_mode

        tta_loss = tta_loss.lower()
        if tta_loss not in {"none", "tent", "pl", "tsd"}:
            raise ValueError("tta_loss must be one of ['none', 'tent', 'pl', 'tsd'].")
        self.tta_loss = tta_loss
        self.tta_weight = tta_weight

        tta_target = tta_target.lower()
        if tta_target not in {"views", "pooled"}:
            raise ValueError("tta_target must be 'views' or 'pooled'.")
        self.tta_target = tta_target

        cc_mode = cc_mode.lower()
        if cc_mode not in {"pairwise", "pooled"}:
            raise ValueError("cc_mode must be 'pairwise' or 'pooled'.")
        self.cc_mode = cc_mode

        def _validate_pool(name: str, value: str) -> str:
            value = value.lower()
            if value not in valid_pools_with_match:
                raise ValueError(
                    f"Unknown view pooling strategy '{value}' for {name}. Expected one of {sorted(valid_pools_with_match)}."
                )
            return value

        self.sup_view_pool = sup_view_pool
        self.js_view_pool = _validate_pool("js_view_pool", js_view_pool)
        self.tta_view_pool = _validate_pool("tta_view_pool", tta_view_pool)
        self.cc_view_pool = _validate_pool("cc_view_pool", cc_view_pool)

        self.view_module = SAFERViewModule(
            num_aug_views=num_aug_views,
            include_original=include_original,
            aug_prob=aug_prob,
            aug_max_ops=aug_max_ops,
            augmentations=augmentations,
            force_noise_first=force_noise_first,
            require_freq_or_blur=require_freq_or_blur,
            sample_params_per_image=sample_params_per_image,
            aug_seed=aug_seed,
            fixed_op=fixed_op,
            fixed_blur_kernel=fixed_blur_kernel,
            fixed_blur_sigma=fixed_blur_sigma,
            fixed_fft_keep_ratio=fixed_fft_keep_ratio,
            allow_noise=allow_noise,
            noise_std=noise_std,
            feature_normalize=feature_normalize,
            view_weighting=view_weighting,
            primary_view_pool=sup_view_pool,
            js_weight=js_weight,
            js_mode=js_mode,
            js_view_pool=self.js_view_pool,
            cc_weight=cc_weight,
            cc_mode=self.cc_mode,
            cc_view_pool=self.cc_view_pool,
            cc_impl=self.cc_impl,
            offdiag_weight=offdiag_weight,
            mean=mean,
            std=std,
            input_is_normalized=input_is_normalized,
            stat_modules=(self.featurizer, self.model),
        )

        if self.episodic:
            self.model_state, self.optimizer_state = copy_model_and_optimizer(
                self.model, self.optimizer
            )
        else:
            self.model_state, self.optimizer_state = None, None

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("Cannot reset without stored state.")
        load_model_and_optimizer(self.model, self.optimizer, self.model_state, self.optimizer_state)

    def forward(self, x):
        if self.episodic:
            self.reset()

        outputs = None
        for _ in range(self.steps):
            outputs = self.forward_and_adapt(x, self.model, self.optimizer)
        return outputs

    def _compute_tta_loss(
        self,
        logits: torch.Tensor,
        probs: torch.Tensor,
        pooled_prob: torch.Tensor,
        view_weights: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.tta_loss == "tent":
            if self.tta_target == "views":
                ent = -(probs * probs.clamp_min(1e-6).log()).sum(dim=-1)
                return _weighted_mean(ent, view_weights)
            return _entropy_minimization_loss(pooled_prob)
        if self.tta_loss == "pl":
            if self.tta_target == "views":
                weights = view_weights if self.use_view_weights else None
                return _weighted_pseudo_label_loss(
                    logits=logits,
                    agg_prob=pooled_prob,
                    view_weights=weights,
                    confidence_scale=self.sup_confidence_scale,
                )
            return _confidence_weighted_pseudo_label(pooled_prob, confidence_scale=self.sup_confidence_scale)
        if self.tta_loss == "tsd":
            teacher = pooled_prob.detach()
            if self.tta_target == "views":
                kl = _prob_kl_divergence(probs, teacher.unsqueeze(1))
                return _weighted_mean(kl, view_weights)
            kl = _prob_kl_divergence(pooled_prob.unsqueeze(1), probs.detach())
            return _weighted_mean(kl, view_weights)
        return torch.tensor(0.0, device=logits.device)

    def forward_and_adapt(self, x, model, optimizer):
        view_out = self.view_module(x, model)
        logits = view_out.logits
        probs = view_out.probs
        feats_bt = view_out.features
        base_prob = view_out.pooled_prob
        base_weights = view_out.pooled_weights
        pooler = view_out.pooler
        js_loss = view_out.js_loss
        cc_loss = view_out.cc_loss

        cm_loss = torch.tensor(0.0, device=logits.device)
        if self.cm_weight > 1e-8:
            cm_loss = self.class_marginal(base_prob)

        sup_loss = torch.tensor(0.0, device=logits.device)
        sup_active = self.sup_mode != "none" and self.sup_weight > 1e-8
        if sup_active:
            sup_weights = base_weights if (self.sup_pl_weighted and self.use_view_weights) else None
            if self.sup_mode == "pl":
                if self.sup_pl_weighted:
                    sup_loss = _weighted_pseudo_label_loss(
                        logits=logits,
                        agg_prob=base_prob,
                        view_weights=sup_weights,
                        confidence_scale=self.sup_confidence_scale,
                    )
                else:
                    sup_loss = _confidence_weighted_pseudo_label(
                        base_prob, confidence_scale=self.sup_confidence_scale
                    )
            elif self.sup_mode == "em":
                sup_loss = _entropy_minimization_loss(base_prob)

        tta_loss = torch.tensor(0.0, device=logits.device)
        if self.tta_loss != "none" and self.tta_weight > 1e-8:
            tta_prob, tta_weights = pooler.pool(self.tta_view_pool)
            tta_loss = self._compute_tta_loss(
                logits=logits,
                probs=probs,
                pooled_prob=tta_prob,
                view_weights=tta_weights,
            )

        loss = (
            (self.js_weight * js_loss)
            + (self.cc_weight * cc_loss)
            + (self.cm_weight * cm_loss)
            + (self.sup_weight * sup_loss)
            + (self.tta_weight * tta_loss)
        )
        wandb.log(
            {
                "SAFER/Loss": loss.item(),
                "SAFER/L_js": js_loss.item(),
                "SAFER/L_cc": cc_loss.item(),
                "SAFER/L_cm": cm_loss.item(),
                "SAFER/L_sup": sup_loss.item(),
                "SAFER/L_tta": tta_loss.item(),
            }
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return base_prob


class MeanTeacherCorrection(nn.Module):
    """
    Adaptive mean-teacher with data correction and optional mixup regularisation.
    """

    def __init__(
        self,
        model,
        optimizer,
        steps: int = 1,
        episodic: bool = False,
        correction_alpha: float = 0.02,
        teacher_momentum: float = 0.99,
        pseudo_momentum: float = 0.5,
        kl_weight: float = 0.1,
        ce_weight: float = 1.0,
        ent_weight: float = 0.0,
        mixup_weight: float = 0.0,
        mixup_beta: float = 0.5,
        use_teacher_prediction: bool = True,
    ):
        super().__init__()
        assert steps > 0, "MeanTeacherCorrection requires ≥ 1 update step."
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        self.episodic = episodic
        self.correction_alpha = correction_alpha
        self.teacher_momentum = teacher_momentum
        self.pseudo_momentum = pseudo_momentum
        self.kl_weight = kl_weight
        self.ce_weight = ce_weight
        self.ent_weight = ent_weight
        self.mixup_weight = mixup_weight
        self.mixup_beta = mixup_beta
        self.use_teacher_prediction = use_teacher_prediction

        self.teacher = deepcopy(self.model).eval()
        self.teacher.requires_grad_(False)
        for m in self.teacher.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None

        if self.episodic:
            self.model_state, self.optimizer_state = copy_model_and_optimizer(self.model, self.optimizer)
            self.teacher_state = deepcopy(self.teacher.state_dict())
        else:
            self.model_state = None
            self.optimizer_state = None
            self.teacher_state = None

    def reset(self):
        if any(state is None for state in (self.model_state, self.optimizer_state, self.teacher_state)):
            raise Exception("Cannot reset without stored states.")
        load_model_and_optimizer(self.model, self.optimizer, self.model_state, self.optimizer_state)
        self.teacher.load_state_dict(self.teacher_state, strict=True)

    def forward(self, x):
        if self.episodic:
            self.reset()

        for _ in range(self.steps):
            self.forward_and_adapt(x, self.model, self.optimizer)
        return self._predict_outputs(x)

    def _predict_outputs(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_teacher_prediction:
            return self.teacher.predict(x)
        return self.model.predict(x)

    def _ema_update_teacher(self):
        if self.teacher is None:
            return
        with torch.no_grad():
            for t_param, s_param in zip(self.teacher.parameters(), self.model.parameters()):
                t_param.data.mul_(self.teacher_momentum).add_(
                    (1.0 - self.teacher_momentum) * s_param.data
                )
            for t_buffer, s_buffer in zip(self.teacher.buffers(), self.model.buffers()):
                if not t_buffer.is_floating_point():
                    t_buffer.data.copy_(s_buffer.data)
                    continue
                t_buffer.data.mul_(self.teacher_momentum).add_(
                    (1.0 - self.teacher_momentum) * s_buffer.data
                )

    def _data_correction(self, x: torch.Tensor, pseudo: torch.Tensor) -> torch.Tensor:
        if self.correction_alpha <= 0:
            return x.detach()
        x_in = x.detach().clone().requires_grad_(True)
        with torch.enable_grad():
            logits = self.teacher.predict(x_in)
            log_probs = F.log_softmax(logits, dim=1)
            target = pseudo.detach()
            correction_loss = -(target * log_probs).sum(dim=1).mean()
        grad = torch.autograd.grad(correction_loss, x_in, retain_graph=False)[0]
        corrected = (x_in - self.correction_alpha * grad).detach()
        return corrected

    def _mixup_loss(self, corrected: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        if self.mixup_weight <= 1e-8 or corrected.size(0) < 2 or self.mixup_beta <= 0:
            return torch.tensor(0.0, device=corrected.device)
        perm = torch.randperm(corrected.size(0), device=corrected.device)
        shuffled = corrected[perm]
        beta_dist = Beta(self.mixup_beta, self.mixup_beta)
        lam = beta_dist.sample((corrected.size(0),)).to(corrected.device)
        lam_img = lam.view(-1, 1, 1, 1)
        lam_prob = lam.view(-1, 1)
        mixed = lam_img * corrected + (1.0 - lam_img) * shuffled

        with torch.no_grad():
            probs_main = F.softmax(logits, dim=1)
            probs_shuffled = probs_main[perm]
            target = lam_prob * probs_main + (1.0 - lam_prob) * probs_shuffled

        mixed_logits = self.model.predict(mixed)
        mixed_probs = F.softmax(mixed_logits, dim=1)
        return F.mse_loss(mixed_probs, target)

    def forward_and_adapt(self, x, model, optimizer):
        student_logits = model.predict(x)
        with torch.no_grad():
            teacher_logits = self.teacher.predict(x)
            teacher_prob = F.softmax(teacher_logits, dim=1)

        student_prob = F.softmax(student_logits.detach(), dim=1)
        pseudo = self.pseudo_momentum * student_prob + (1.0 - self.pseudo_momentum) * teacher_prob
        corrected = self._data_correction(x, pseudo)

        with torch.no_grad():
            teacher_logits_corr = self.teacher.predict(corrected)
            teacher_prob_corr = F.softmax(teacher_logits_corr, dim=1)

        pseudo = self.pseudo_momentum * student_prob + (1.0 - self.pseudo_momentum) * teacher_prob_corr
        pseudo = pseudo.clamp_min(1e-6)

        student_logits_corr = model.predict(corrected)
        log_probs = F.log_softmax(student_logits_corr, dim=1)

        ce_loss = torch.tensor(0.0, device=x.device)
        if self.ce_weight > 1e-8:
            ce_loss = -(pseudo.detach() * log_probs).sum(dim=1).mean()

        ent_loss = torch.tensor(0.0, device=x.device)
        if self.ent_weight > 1e-8:
            ent_loss = softmax_entropy(student_logits_corr).mean()

        kl_loss = torch.tensor(0.0, device=x.device)
        if self.kl_weight > 1e-8:
            kl_loss = F.kl_div(
                log_probs,
                teacher_prob_corr.detach(),
                reduction="batchmean",
            )

        mix_loss = torch.tensor(0.0, device=x.device)
        if self.mixup_weight > 1e-8:
            mix_loss = self._mixup_loss(corrected, student_logits_corr)

        loss = (
            self.ce_weight * ce_loss
            + self.ent_weight * ent_loss
            + self.kl_weight * kl_loss
            + self.mixup_weight * mix_loss
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        self._ema_update_teacher()
        wandb.log(
            {
                "MTDC/Loss": loss.item(),
                "MTDC/L_ce": ce_loss.item(),
                "MTDC/L_ent": ent_loss.item(),
                "MTDC/L_kl": kl_loss.item(),
                "MTDC/L_mix": mix_loss.item(),
            }
        )
