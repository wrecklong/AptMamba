# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Implements the knowledge distillation loss
"""
import torch
from torch.nn import functional as F


class DistillationLoss(torch.nn.Module):
    """
    This module wraps a standard criterion and adds an extra knowledge distillation loss by
    taking a teacher model prediction and using it as additional supervision.
    """
    def __init__(self, base_criterion: torch.nn.Module, teacher_model: torch.nn.Module,
                 distillation_type: str, alpha: float, tau: float):
        super().__init__()
        self.base_criterion = base_criterion
        self.teacher_model = teacher_model
        assert distillation_type in ['none', 'soft', 'hard']
        self.distillation_type = distillation_type
        self.alpha = alpha
        self.tau = tau

    def forward(self, inputs, outputs, labels):
        """
        Args:
            inputs: The original inputs that are feed to the teacher model
            outputs: the outputs of the model to be trained. It is expected to be
                either a Tensor, or a Tuple[Tensor, Tensor], with the original output
                in the first position and the distillation predictions as the second output
            labels: the labels for the base criterion
        """
        outputs_kd = None
        if not isinstance(outputs, torch.Tensor):
            # assume that the model outputs a tuple of [outputs, outputs_kd]
            outputs, outputs_kd = outputs
        base_loss = self.base_criterion(outputs, labels)
        if self.distillation_type == 'none':
            return base_loss

        if outputs_kd is None:
            raise ValueError("When knowledge distillation is enabled, the model is "
                             "expected to return a Tuple[Tensor, Tensor] with the output of the "
                             "class_token and the dist_token")
        # don't backprop throught the teacher
        with torch.no_grad():
            teacher_outputs = self.teacher_model(inputs)

        if self.distillation_type == 'soft':
            T = self.tau
            # taken from https://github.com/peterliht/knowledge-distillation-pytorch/blob/master/model/net.py#L100
            # with slight modifications
            distillation_loss = F.kl_div(
                F.log_softmax(outputs_kd / T, dim=1),
                #We provide the teacher's targets in log probability because we use log_target=True 
                #(as recommended in pytorch https://github.com/pytorch/pytorch/blob/9324181d0ac7b4f7949a574dbc3e8be30abe7041/torch/nn/functional.py#L2719)
                #but it is possible to give just the probabilities and set log_target=False. In our experiments we tried both.
                F.log_softmax(teacher_outputs / T, dim=1),
                reduction='sum',
                log_target=True
            ) * (T * T) / outputs_kd.numel()
            #We divide by outputs_kd.numel() to have the legacy PyTorch behavior. 
            #But we also experiments output_kd.size(0) 
            #see issue 61(https://github.com/facebookresearch/deit/issues/61) for more details
        elif self.distillation_type == 'hard':
            distillation_loss = F.cross_entropy(outputs_kd, teacher_outputs.argmax(dim=1))

        loss = base_loss * (1 - self.alpha) + distillation_loss * self.alpha
        return loss

class PruningLoss_dynamic(torch.nn.Module):
    """
    This module wraps a standard criterion and adds an extra knowledge distillation loss by
    taking a teacher model prediction and using it as additional supervision.
    """
    def __init__(self, base_criterion: torch.nn.Module, ratio_weight=2.0, reconstruction_weight = 0.5, pruning_weight = 0.1, dynamic=False, pruning_loc=[6,12,18], keep_ratio=[0.75, 0.5, 0.25], clf_weight=0, mse_token=False, print_mode=True):
        super().__init__()
        self.base_criterion = base_criterion
        self.clf_weight = clf_weight
        self.pruning_loc = pruning_loc
        self.keep_ratio = keep_ratio
        self.count = 0
        self.print_mode = print_mode
        self.cls_loss = 0
        self.ratio_loss = 0
    
        self.mse_loss = 0
        self.pruning_loss = 0

        self.mse_token = mse_token
        self.dynamic = dynamic

        self.ratio_weight = ratio_weight
        self.reconstruction_weight = reconstruction_weight
        self.pruning_weight = pruning_weight

        #print('ratio_weight=', ratio_weight, 'distill_weight', distill_weight, 'reconstruction_weight', reconstruction_weight, 'pruning_weight', pruning_weight)
        print('ratio_weight=', ratio_weight,  'reconstruction_weight', reconstruction_weight,  'pruning_weight', pruning_weight)


        if dynamic:
            print('using dynamic loss')

    def forward(self, inputs, outputs, labels):
        """
        Args:
            inputs: The original inputs that are feed to the teacher model
            outputs: the outputs of the model to be trained. It is expected to be
                either a Tensor, or a Tuple[Tensor, Tensor], with the original output
                in the first position and the distillation predictions as the second output
            labels: the labels for the base criterion
        """

        pred, out_pred_score, pruning_loss, mse_loss = outputs

        pred_loss = torch.tensor(0, device=pred.device)

        ratio = self.keep_ratio
        for i, score in enumerate(out_pred_score):
            if self.dynamic:
                pos_ratio = score.mean()
            else:
                pos_ratio = score.mean(1)
            pred_loss = pred_loss + ((pos_ratio - ratio[i]) ** 2).mean()

        cls_loss = self.base_criterion(pred, labels)

        loss_part = []

        # print(cls_loss, pred_loss)
        # loss = self.clf_weight * cls_loss + self.ratio_weight * pred_loss / len(self.pruning_loc) +  self.reconstruction_weight * mse_loss + self.distill_weight * cls_kl_loss + self.distill_weight * token_kl_loss 
        loss = self.clf_weight * cls_loss + self.ratio_weight * pred_loss / len(self.pruning_loc) + self.reconstruction_weight * mse_loss + self.pruning_weight * pruning_loss

        if self.print_mode:
            self.cls_loss += cls_loss.item()
            self.ratio_loss += pred_loss.item()
            # self.cls_distill_loss += cls_kl_loss.item()
            # self.token_distill_loss += token_kl_loss.item()
            self.pruning_loss += pruning_loss.item()
            self.mse_loss += mse_loss.item()
            loss_part.append(cls_loss)
            loss_part.append(pred_loss)
            loss_part.append(pruning_loss)
            # loss_part.append(cls_kl_loss)
            # loss_part.append(token_kl_loss)
            loss_part.append(mse_loss)
            self.count += 1
            if self.count == 100:
                #print('loss info: cls_loss=%.4f, ratio_loss=%.4f, cls_kl=%.4f, token_kl=%.4f, mse_loss=%.4f' % (self.cls_loss / 100, self.ratio_loss / 100, self.cls_distill_loss/ 100, self.token_distill_loss/ 100, self.mse_loss/100))
                print('loss info: cls_loss=%.4f, ratio_loss=%.4f, pruning_loss=%.4f, mse_loss=%.4f' % (self.cls_loss / 100, self.ratio_loss / 100, self.pruning_loss/100, self.mse_loss/100))
                self.count = 0
                self.cls_loss = 0
                self.ratio_loss = 0
                # self.cls_distill_loss = 0
                # self.token_distill_loss = 0
                self.mse_loss = 0
                self.pruning_loss = 0
        return loss, loss_part