from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torchvision.models.feature_extraction import create_feature_extractor
from utils.svd import SVDDrop2D
from utils.fft import FFTDrop2D

@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


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
    def __init__(self,model,filter_K=100,steps=1,episodic=False):
        super().__init__()
        self.model = model.eval()
        self.featurizer = model.featurizer
        self.classifier = model.classifier
        self.steps = steps
        assert steps > 0, "requires >= 1 step(s) to forward and update"
        self.episodic = episodic
        self.filter_K = filter_K
        
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
        z = self.featurizer(x)
        p = self.classifier(z)
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
    def __init__(self,model,optimizer,lam=0,filter_K=100,steps=1,episodic=False):
        super().__init__()
        self.model = model
        self.featurizer = model.featurizer
        self.classifier = model.classifier
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "requires >= 1 step(s) to forward and update"
        self.episodic = episodic
        self.filter_K = filter_K
        
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
        z = self.featurizer(x)
        p = self.classifier(z)
                       
        yhat = F.one_hot(p.argmax(1), num_classes=self.num_classes).float()
        ent = softmax_entropy(p)
        scores = F.softmax(p,1)

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
            tracer_kwargs={"leaf_modules": [SVDDrop2D, FFTDrop2D]},
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
