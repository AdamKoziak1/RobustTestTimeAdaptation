import torch
import torch.nn as nn

from .aug_predictor import PolicyPredictor
from .classification_policies import get_sub_policies, apply_augment
from utils.tesla_losses import EntropyLoss, EntropyClassMarginals


class OptAug(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        sub_policy_dim: int,
        aug_mult: int,
        name: str,
        normalize_fn,
        denormalize_fn,
        lmb_norm: float = 1.0,
    ) -> None:
        super().__init__()

        self.encoder = encoder
        self.softmax = nn.Softmax(dim=-1)
        self.lmb_norm = lmb_norm

        self.aug_mult = aug_mult
        self.sub_policy_dim = sub_policy_dim
        self.name = name
        self.sub_policies = torch.tensor(get_sub_policies(self.sub_policy_dim))
        self.policy_predictor = PolicyPredictor(len(self.sub_policies), self.sub_policy_dim, self.name)

        self.optimizer_policy = torch.optim.Adam(
            [{"params": self.policy_predictor.parameters(), "lr": 0.1}]
        )

        self.normalize_fn = normalize_fn
        self.denormalize_fn = denormalize_fn

        self.criterion_ent = EntropyLoss()
        self.criterian_cm = EntropyClassMarginals()
        self.criterian_l2 = torch.nn.MSELoss()

        self.current_norm_inputs = {}
        self.hook_handlers = {}
        self.register_norm_hooks()

    def _encode(self, x):
        if hasattr(self.encoder, "predict"):
            return self.encoder.predict(x)
        return self.encoder(x)

    def norm_hook(self, idx):
        def hook(module, input, output):
            input = input[0]
            self.current_norm_inputs[idx] = [
                input.mean(dim=(-2, -1)),
                input.var(dim=(-2, -1)),
            ]

        return hook

    def register_norm_hooks(self):
        idx = 0
        for m in self.encoder.modules():
            if isinstance(m, (torch.nn.BatchNorm2d, torch.nn.SyncBatchNorm, torch.nn.LayerNorm)):
                self.hook_handlers[idx] = m.register_forward_hook(self.norm_hook(idx))
                idx += 1

    def compute_norm_stat_loss(self, pre_norm_feats_curr, pre_norm_feats_target):
        device = pre_norm_feats_curr[0][0].device
        loss = torch.zeros(pre_norm_feats_curr[0][0].size(0), device=device)
        for i in range(len(pre_norm_feats_curr)):
            target_mean = pre_norm_feats_target[i][0].detach()
            curr_mean = pre_norm_feats_curr[i][0]
            loss += (curr_mean - target_mean).pow(2).mean(dim=-1)
        return loss / len(pre_norm_feats_curr)

    def batch_apply_augmentation(self, apply_fn, sub_policies, x, sub_policy_index, sub_policy_mags):
        fn_idxs = sub_policies[sub_policy_index]
        out = x.clone()
        for i in range(len(fn_idxs)):
            out = apply_fn(out, fn_idxs[i], sub_policy_mags[i])
        return out

    def sample_apply_augmentation(self, apply_fn, sub_policies, x, sub_policy_indices, sub_policy_mags):
        fns_idxs = sub_policies[sub_policy_indices.cpu()]
        x = x.unsqueeze(0)
        out = []
        for fns, mags in zip(fns_idxs, sub_policy_mags):
            x_aug = x.clone()
            for fn, m in zip(fns, mags):
                x_aug = apply_fn(x_aug, fn, m.unsqueeze(0))
            out.append(x_aug)
        return torch.cat(out, dim=0)

    def forward(self, x):
        return self.policy_predictor()

    @torch.no_grad()
    def sample(self, x_cuda):
        x_aug = []
        prob_sub_policy, _ = self.policy_predictor()
        prob_sub_policy = prob_sub_policy.unsqueeze(0).repeat(x_cuda.size(0), 1)
        selected_sub_policy_idx = torch.multinomial(prob_sub_policy, self.aug_mult, replacement=False)

        for k in range(self.aug_mult):
            prob_sub_policy, mag_sub_policy = self.policy_predictor()
            prob_sub_policy = prob_sub_policy.unsqueeze(0).repeat(x_cuda.size(0), 1)

            curr_selected_sub_policy_index = selected_sub_policy_idx[:, k]
            curr_selected_sub_policy_mag = mag_sub_policy[curr_selected_sub_policy_index,]

            x_cuda_denorm = self.denormalize_fn(x_cuda)
            x_aug_curr = []
            for i, x_item in enumerate(x_cuda_denorm):
                x_aug_curr.append(
                    self.sample_apply_augmentation(
                        apply_augment,
                        self.sub_policies,
                        x_item,
                        curr_selected_sub_policy_index[i : (i + 1)],
                        curr_selected_sub_policy_mag[i : (i + 1), :],
                    )
                )

            x_aug_curr = torch.cat(x_aug_curr, dim=0)
            x_aug_curr = self.normalize_fn(x_aug_curr)
            x_aug.append(x_aug_curr.detach())

        x_aug = torch.stack(x_aug, dim=1)
        return x_aug.detach()

    def optimize(self, x_cuda):
        with torch.no_grad():
            self._encode(x_cuda)
            orig_norm_stats = {}
            for k, v in self.current_norm_inputs.items():
                orig_norm_stats[k] = [v[0].detach(), v[1].detach()]

        self.optimizer_policy.zero_grad(set_to_none=True)

        with torch.no_grad():
            prob_sub_policy = torch.ones(
                len(self.sub_policies), dtype=torch.float32, device=x_cuda.device
            ) / len(self.sub_policies)
            prob_sub_policy = prob_sub_policy.unsqueeze(0).repeat(x_cuda.size(0), 1)
            selected_sub_policy_idx = torch.multinomial(
                prob_sub_policy, self.aug_mult, replacement=False
            )

        all_losses = {"H_ent_aug": [], "loss_policy": [], "H_norm": []}

        for k in range(self.aug_mult):
            prob_sub_policy, mag_sub_policy = self.policy_predictor()
            prob_sub_policy = prob_sub_policy.unsqueeze(0).repeat(x_cuda.size(0), 1)

            curr_selected_sub_policy_index = selected_sub_policy_idx[:, k]
            curr_selected_sub_policy_mag = mag_sub_policy[curr_selected_sub_policy_index,]
            curr_selected_prob_sub_policy = prob_sub_policy[
                torch.arange(x_cuda.size(0)), curr_selected_sub_policy_index
            ]

            x_cuda_denorm = self.denormalize_fn(x_cuda)
            x_aug_curr = []
            for i, x_item in enumerate(x_cuda_denorm):
                x_aug_curr.append(
                    self.sample_apply_augmentation(
                        apply_augment,
                        self.sub_policies,
                        x_item,
                        curr_selected_sub_policy_index[i : (i + 1)],
                        curr_selected_sub_policy_mag[i : (i + 1), :],
                    )
                )

            x_aug_curr = torch.cat(x_aug_curr, dim=0)
            x_aug_curr = self.normalize_fn(x_aug_curr)

            pred = self.softmax(self._encode(x_aug_curr))
            loss_ent_aug = self.criterion_ent(pred)

            currn_norm_stats = self.current_norm_inputs
            loss_norm = self.compute_norm_stat_loss(currn_norm_stats, orig_norm_stats)

            (-loss_ent_aug.mean() + loss_norm.mean()).backward()
            loss_curr = (-loss_ent_aug + loss_norm * self.lmb_norm).detach()

            loss_policy = torch.mean(
                loss_curr * torch.log(curr_selected_prob_sub_policy + 1e-8)
            )
            loss_policy.backward()

            all_losses["H_ent_aug"].append(loss_ent_aug.mean().detach().cpu().item())
            all_losses["H_norm"].append(loss_norm.mean().detach().cpu().item())
            all_losses["loss_policy"].append(loss_policy.mean().detach().cpu().item())

        self.optimizer_policy.step()
        for k, v in all_losses.items():
            all_losses[k] = torch.tensor(v).mean()
        return all_losses
