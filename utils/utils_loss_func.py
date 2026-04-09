import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable




class FocalLoss(nn.Module):
    """
    Focal Loss for imbalanced classification
    
    Args:
        alpha (float or list): weighting factor for each class. 
                              If float, used for positive class (binary).
                              If list, weights for each class (multi-class).
        gamma (float): focusing parameter. Higher gamma focuses more on hard examples.
        reduction (str): 'mean', 'sum', or 'none'
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        """
        Args:
            inputs: raw logits of shape (N, C) where C = number of classes
            targets: ground truth labels of shape (N,) for class indices
        """
        # Convert targets to one-hot encoding for multi-class
        if inputs.dim() > 1 and inputs.size(1) > 1:  # Multi-class
            num_classes = inputs.size(1)
            
            # Convert targets to one-hot
            targets_one_hot = F.one_hot(targets, num_classes).float()
            
            # Apply softmax to get probabilities
            probs = F.softmax(inputs, dim=1)
            
            # Get probabilities of target classes
            pt = (probs * targets_one_hot).sum(dim=1)
            
            # Compute focal loss
            ce_loss = -torch.log(pt + 1e-8)  # Cross entropy
            focal_weight = (1 - pt) ** self.gamma
            
            # Apply class weighting
            if isinstance(self.alpha, (list, torch.Tensor)):
                if len(self.alpha) == num_classes:
                    alpha_t = torch.tensor(self.alpha, device=inputs.device)[targets]
                else:
                    alpha_t = 1.0
            else:
                alpha_t = self.alpha if self.alpha is not None else 1.0
                
            loss = alpha_t * focal_weight * ce_loss
        else:  # Binary classification
            # Sigmoid for binary classification
            probs = torch.sigmoid(inputs)
            
            # Flatten for binary case
            if targets.dim() > 1:
                targets = targets.view(-1, 1)
            probs = probs.view(-1, 1)
            
            # Get pt
            pt = torch.where(targets == 1, probs, 1 - probs)
            
            # Compute loss
            ce_loss = -torch.log(pt + 1e-8)
            focal_weight = (1 - pt) ** self.gamma
            
            # Apply alpha weighting
            if isinstance(self.alpha, (float, int)):
                alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
            else:
                alpha_t = 1.0
                
            loss = alpha_t * focal_weight * ce_loss
            loss = loss.view(-1)
        
        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss
         
class PolyLoss(nn.Module):
    """
    PolyLoss: A Polynomial Expansion Perspective of Classification Loss Functions
    
    Args:
        epsilon (float or list): polynomial coefficients. 
                                If float, applies Poly-1 loss with epsilon.
                                If list, coefficients for polynomial terms.
        reduction (str): 'mean', 'sum', or 'none'
    """
    def __init__(self, epsilon=1.0, reduction='mean'):
        super(PolyLoss, self).__init__()
        self.epsilon = epsilon
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        """
        PolyLoss forward pass
        
        Args:
            inputs: logits of shape (N, C) where C = number of classes
            targets: ground truth labels of shape (N,) for class indices
        """
        # Convert to one-hot for multi-class
        num_classes = inputs.size(1)
        targets_one_hot = F.one_hot(targets, num_classes).float()
        
        # Get probabilities
        probs = F.softmax(inputs, dim=1)
        
        # Get probability of target class
        pt = (probs * targets_one_hot).sum(dim=1)
        
        # Compute cross entropy
        ce_loss = -torch.log(pt + 1e-8)
        
        # Apply PolyLoss modification
        if isinstance(self.epsilon, (float, int)):
            # Poly-1 loss: CE + epsilon * (1 - pt)
            poly_loss = ce_loss + self.epsilon * (1 - pt)
        elif isinstance(self.epsilon, (list, torch.Tensor)):
            # General PolyLoss: sum over polynomial terms
            poly_loss = ce_loss
            for j, eps_j in enumerate(self.epsilon, start=1):
                poly_loss = poly_loss + eps_j * ((1 - pt) ** j)
        else:
            raise ValueError("epsilon must be float or list/tensor of coefficients")
        
        # Apply reduction
        if self.reduction == 'mean':
            return poly_loss.mean()
        elif self.reduction == 'sum':
            return poly_loss.sum()
        else:
            return poly_loss

class AdaptiveStructuralPolyLoss(nn.Module):
    """
    自适应结构化PolyLoss (ASPL)
    
    设计理念：
    针对无人机-卫星匹配中频域增强带来的高频伪影问题，通过动态调整PolyLoss的
    多项式系数epsilon，在“聚焦难例”和“抑制噪声”之间自动切换。
    
    特点：
    1. 不需要在__init__中传入num_class。
    2. 基于批次预测熵（Batch Entropy）动态计算epsilon。
    3. 集成动态Label Smoothing。
    """
    def __init__(self, 
                 base_epsilon: float = 2.5, #默认2，1.5不行,2.5shaogao
                 reduction: str = 'mean', 
                 label_smoothing: float = 0.1,
                 robustness_threshold: float = 0.5):
        """
        Args:
            base_epsilon (float): epsilon的基准缩放范围。默认为2.0。
            reduction (str): 'mean', 'sum', 或 'none'。
            label_smoothing (float): 标签平滑系数。
            robustness_threshold (float): 熵的阈值（0-1之间）。
                                          超过此阈值被视为噪声，epsilon转为负值,原0.6，0.7不行。
        """
        super().__init__()
        self.base_epsilon = base_epsilon
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        self.robustness_threshold = robustness_threshold

    def forward(self, logits, targets):
        """
        Args:
            logits: - 未经Softmax的输出
            targets: (LongTensor) 或 (FloatTensor)
        """
        # 1. 动态推断类别数 C
        num_classes = logits.size(-1)
        device = logits.device
        
        # 2. 准备软标签 (Soft Targets)
        # 如果是索引标签，转换为One-hot并进行平滑
        if targets.dim() == 1:
            with torch.no_grad():
                hard_targets = torch.zeros_like(logits)
                hard_targets.scatter_(1, targets.unsqueeze(1), 1)
                smooth_targets = hard_targets * (1 - self.label_smoothing) + \
                                 self.label_smoothing / num_classes
        else:
            # 假设输入已经是软标签（或者是混合过的标签）
            smooth_targets = targets

        # 3. 计算基础交叉熵 (Base Cross Entropy)
        # 使用 log_softmax + sum 结构保证数值稳定性
        log_probs = F.log_softmax(logits, dim=-1)
        ce_loss = -(smooth_targets * log_probs).sum(dim=-1)

        # 4. 计算自适应动态 Epsilon (Adaptive Dynamic Epsilon)
        # 核心创新：利用预测熵判断样本是“难样本”还是“噪声”
        with torch.no_grad():
            probs = F.softmax(logits, dim=-1)
            # 计算香农熵: H(p) = -sum(p * log(p))
            current_entropy = -(probs * log_probs).sum(dim=-1)
            
            # 计算最大可能的熵 log(C) 用于归一化
            max_entropy = torch.log(torch.tensor(float(num_classes), device=device))
            
            # 归一化熵值 
            # normalized_entropy 越接近1，表示模型越不确定（可能是伪影噪声）
            normalized_entropy = current_entropy / (max_entropy + 1e-6)
            
            # 调度逻辑：
            # epsilon = base * (阈值 - 归一化熵)
            # 例：设阈值0.6。
            # Case A (清晰样本): norm_entropy = 0.2 -> epsilon = 2 * (0.6 - 0.2) = +0.8 (Focus)
            # Case B (伪影/噪声): norm_entropy = 0.9 -> epsilon = 2 * (0.6 - 0.9) = -0.6 (Robust/Suppress)
            dynamic_epsilon = self.base_epsilon * (self.robustness_threshold - normalized_entropy)

        # 5. 计算 Poly-1 项
        # Pt: 模型对目标类别的预测概率。对于软标签，取加权和。
        pt = (smooth_targets * probs).sum(dim=-1)
        
        # PolyLoss公式: L = L_ce + epsilon * (1 - Pt)
        # 这里 epsilon 是动态的，针对每个样本独立计算
        poly_term = dynamic_epsilon * (1 - pt)

        # 6. 组合最终损失
        total_loss = ce_loss + poly_term

        # 7. Reduction
        if self.reduction == 'mean':
            return total_loss.mean()
        elif self.reduction == 'sum':
            return total_loss.sum()
        else:
            return total_loss

class DynamicSoftFocalLoss(nn.Module):
    def __init__(self, 
                 lb_smooth=0.14, 
                 reduction='mean', 
                 ignore_index=-1,
                 fl_gamma=2.0,    # Focal Loss的gamma
                 fl_alpha=0.25):  # 平衡正负样本权重
        super().__init__()
        self.lb_smooth = lb_smooth
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.fl_gamma = fl_gamma
        self.fl_alpha = fl_alpha

    def forward(self, input, target):   
        logits = input.float() # 确保fp32
        
        # --------------------------
        # 1. 准备数据与掩码
        # --------------------------
        # 创建忽略掩码
        valid_mask = target != self.ignore_index
        target = target * valid_mask # 将ignore的标签暂时置0，避免越界，后面会mask掉loss
        n_valid = valid_mask.sum()

        if n_valid == 0:
            return torch.tensor(0., device=logits.device)

        # --------------------------
        # 2. 构建平滑标签 (Soft Labels)
        # --------------------------
        num_classes = logits.size(1)
        # 生成One-hot
        target_one_hot = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1)
        
        # 应用Label Smoothing公式: y_ls = (1 - ε) * y + ε / K
        # 这里的target_one_hot就是y
        alpha = 1.0 - self.lb_smooth
        smoothing_value = self.lb_smooth / num_classes
        soft_targets = alpha * target_one_hot + smoothing_value

        # --------------------------
        # 3. 计算 Soft Focal Loss 核心部分
        # --------------------------
        # 计算概率 P
        probs = F.softmax(logits, dim=1)
        # 计算 Log P
        log_probs = F.log_softmax(logits, dim=1)

        # Focal Term: (1 - p)^gamma
        # 对于Soft Label场景，我们希望模型预测接近Target，
        # 所以这里的权重通常计算为：|target - prob|^gamma 或简单地使用 (1-p)^gamma
        # 为了保持标准Focal Loss特性，我们针对每个类别独立计算权重
        focal_weight = torch.pow(1.0 - probs, self.fl_gamma)
        
        # 原始Focal Loss结构: - alpha * (1-p)^gamma * log(p)
        # 在Soft Label下，我们要计算所有类别的加权和
        # 我们在这里加入 alpha 参数的处理 (通常alpha只针对正类，这里简化为统一缩放或根据需要定制)
        # 为了简单且适配多分类，这里暂不使用类别特定的alpha，依靠soft targets本身来平衡
        
        # 基础损失: - Soft_Target * Log_Prob * Focal_Weight
        pixel_loss = -soft_targets * log_probs * focal_weight
        
        # 按类别求和得到每个样本的Loss
        loss_per_sample = pixel_loss.sum(dim=1)

        # --------------------------
        # 4. 注入你的“动态权重” (Dynamic Weight)
        # --------------------------
        # 这里是将你原有的基于全局统计的权重机制，作为系数乘到Focal Loss上
        dynamic_weight = self.get_dynamic_weight(logits)
        
        loss_per_sample = loss_per_sample * dynamic_weight

        # --------------------------
        # 5. Reduction与清理
        # --------------------------
        # 应用忽略掩码
        loss_per_sample = loss_per_sample * valid_mask.float()

        if self.reduction == 'mean':
            return loss_per_sample.sum() / n_valid
        elif self.reduction == 'sum':
            return loss_per_sample.sum()
        else:
            return loss_per_sample

    @staticmethod
    def get_dynamic_weight(input):
        """
        保留了你原有的基于batch内统计特性的动态权重计算逻辑。
        这对于无人机/卫星匹配非常有用，因为它可以感知整个batch的图像难易度分布。
        """
        # 避免除零
        eps = 1e-6
        # 不需要梯度
        with torch.no_grad():
            probs = F.softmax(input, dim=1)
            # max_pred_b: [B], 每个样本的最大置信度
            max_pred_b, _ = torch.max(probs, dim=1) 
            # max_pred_c: [C], 整个Batch中每个类别的最大置信度
            max_pred_c, _ = torch.max(probs, dim=0) 
            
            # 全局难样本阈值 (所有类别最大概率的均值)
            u_t = torch.mean(max_pred_c) 
            
            # 计算方差 (用于高斯衰减)
            variance_t = torch.var(max_pred_c) # 使用torch.var更直接

            weight = torch.ones_like(max_pred_b)
            lambda_max = 1.0 # 可以作为超参暴露出去
            
            # 筛选难样本：置信度 < 全局阈值
            mask_hard = max_pred_b < u_t
            
            if mask_hard.any():
                # 高斯函数形式提升权重
                # 越接近 u_t (虽然小但接近)，提升幅度受方差控制
                # 这里的逻辑是：如果置信度很低，权重会增加
                diff = max_pred_b[mask_hard] - u_t
                scale = torch.exp(-(diff ** 2) / (2 * variance_t + eps))
                weight[mask_hard] = 1.0 + lambda_max * scale # 基础权重1.0 + 增量
                
        return weight

class DynamicLabelSmoothSoftmaxCEV1(nn.Module):
    def __init__(self, lb_smooth=0.10, reduction='mean', ignore_index=-1):
        super(DynamicLabelSmoothSoftmaxCEV1, self).__init__()
        self.lb_smooth = lb_smooth
        self.reduction = reduction
        self.lb_ignore = ignore_index
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, target):   
        logits = input.float()  # use fp32 to avoid nan    
        with torch.no_grad():
            num_classes = logits.size(1)
            label = target.clone().detach()  
            ignore = label.eq(self.lb_ignore)      
            n_valid = ignore.eq(0).sum()         
            label[ignore] = 0
            lb_pos, lb_neg = 1. - self.lb_smooth, self.lb_smooth / num_classes
            lb_one_hot = torch.empty_like(logits).fill_(lb_neg)\
                .scatter_(1, label.unsqueeze(1), lb_pos).detach()
   
        logs = self.log_softmax(logits)
        dynamic_weight = self.get_weight(input)
        loss = -torch.sum(logs * lb_one_hot, dim=1)
        loss = loss * dynamic_weight
        loss[ignore] = 0
        if self.reduction == 'mean':
            loss = loss.sum() / n_valid
        if self.reduction == 'sum':
            loss = loss.sum()

        return loss

    @staticmethod
    def get_weight(input):
        num_classes = input.size(1)
        x = 1e-6
        logits = input.float()
        probs_logits = F.softmax(logits, dim=1)
        max_pred_b, max_idx_b = torch.max(probs_logits, dim=1)
        max_pred_c, max_idx_c = torch.max(probs_logits, dim=0)
        u_t = torch.mean(max_pred_c, dim=0)
        u_t_tensor = torch.full_like(max_pred_b, u_t.item())
        diff_squared = (max_pred_c - u_t) ** 2
        variance_t = torch.mean(diff_squared, dim=0)

        weight = torch.ones_like(max_pred_b)
        lambda_max = 1.0
        update = max_pred_b < u_t_tensor
        if torch.any(update):
            weight[update] = lambda_max * torch.exp(
                -((max_pred_b[update] - u_t_tensor[update]) ** 2) / (2 * variance_t + x))

        return weight









### calculate Cross Entropy loss
def cal_loss(outputs, labels, loss_func):
    loss = 0
    if isinstance(outputs, list):
        for i in outputs:
            loss += loss_func(i, labels)
        loss = loss/len(outputs)
    else:
        loss = loss_func(outputs,labels)
    return loss

### calculate KL loss
def cal_kl_loss(outputs, outputs2, loss_func):
    loss = 0
    if isinstance(outputs, list):
        for i in range(len(outputs)):
            loss += loss_func(F.log_softmax(outputs[i], dim=1),
                               F.softmax(Variable(outputs2[i]), dim=1))
        loss = loss/len(outputs)
    else:
        loss = loss_func(F.log_softmax(outputs, dim=1),
                          F.softmax(Variable(outputs2), dim=1))
    return loss

### calculate Triplet loss
def cal_triplet_loss(outputs,outputs2,labels,loss_func,split_num=8):
    if isinstance(outputs,list):
        loss = 0
        for i in range(len(outputs)):
            out_concat = torch.cat((outputs[i], outputs2[i]), dim=0)
            labels_concat = torch.cat((labels,labels),dim=0)
            loss += loss_func(out_concat,labels_concat)
        loss = loss/len(outputs)
    else:
        out_concat = torch.cat((outputs, outputs2), dim=0)
        labels_concat = torch.cat((labels,labels),dim=0)
        loss = loss_func(out_concat,labels_concat)
    return loss

### noormalization
def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-6)
    return x

### calculate Euclidean distance
def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist = dist - 2 * torch.matmul(x, y.t())
    # dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-6).sqrt()  # for numerical stability
    return dist

### calculate cosine distance
def cosine_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    x_norm = torch.pow(x, 2).sum(1, keepdim=True).sqrt().expand(m, n)
    y_norm = torch.pow(y, 2).sum(1, keepdim=True).sqrt().expand(n, m).t()
    xy_intersection = torch.mm(x, y.t())
    dist = xy_intersection/(x_norm * y_norm)
    dist = (1. - dist) / 2
    return dist

### mining hard example
def hard_example_mining(dist_mat, labels, return_inds=False):
    """For each anchor, find the hardest positive and negative sample.
    Args:
      dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
      labels: pytorch LongTensor, with shape [N]
      return_inds: whether to return the indices. Save time if `False`(?)
    Returns:
      dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
      dist_an: pytorch Variable, distance(anchor, negative); shape [N]
      p_inds: pytorch LongTensor, with shape [N];
        indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
      n_inds: pytorch LongTensor, with shape [N];
        indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
    NOTE: Only consider the case in which all labels have same num of samples,
      thus we can cope with all anchors in parallel.
    """
    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)

    # shape [N, N]
    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())

    # `dist_ap` means distance(anchor, positive)
    # both `dist_ap` and `relative_p_inds` with shape [N, 1]
    dist_ap, relative_p_inds = torch.max(
        dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True)
    # print(dist_mat[is_pos].shape)
    # `dist_an` means distance(anchor, negative)
    # both `dist_an` and `relative_n_inds` with shape [N, 1]
    dist_an, relative_n_inds = torch.min(
        dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)
    # shape [N]
    dist_ap = dist_ap.squeeze(1)
    dist_an = dist_an.squeeze(1)

    if return_inds:
        # shape [N, N]
        ind = (labels.new().resize_as_(labels)
               .copy_(torch.arange(0, N).long())
               .unsqueeze(0).expand(N, N))
        # shape [N, 1]
        p_inds = torch.gather(
            ind[is_pos].contiguous().view(N, -1), 1, relative_p_inds.data)
        n_inds = torch.gather(
            ind[is_neg].contiguous().view(N, -1), 1, relative_n_inds.data)
        # shape [N]
        p_inds = p_inds.squeeze(1)
        n_inds = n_inds.squeeze(1)
        return dist_ap, dist_an, p_inds, n_inds

    return dist_ap, dist_an


### Triplet Loss ###

class TripletLoss(object):
    """
    Triplet loss using HARDER example mining,
    modified based on original triplet loss using hard example mining
    """

    def __init__(self, margin=None, hard_factor=0.0):
        self.margin = margin
        self.hard_factor = hard_factor
        if margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()

    def __call__(self, global_feat, labels, normalize_feature=False):
        if normalize_feature:
            global_feat = normalize(global_feat, axis=-1)
        dist_mat = euclidean_dist(global_feat, global_feat)
        dist_ap, dist_an = hard_example_mining(dist_mat, labels)

        dist_ap *= (1.0 + self.hard_factor)
        dist_an *= (1.0 - self.hard_factor)

        y = dist_an.new().resize_as_(dist_an).fill_(1)
        if self.margin is not None:
            loss = self.ranking_loss(dist_an, dist_ap, y)
        else:
            loss = self.ranking_loss(dist_an - dist_ap, y)
        return loss


class Tripletloss(nn.Module):
    """Triplet loss with hard positive/negative mining.

    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.

    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.

    Args:
        margin (float): margin for triplet.
    """
    def __init__(self, margin=0.3, hard_factor=0.0):
        super(Tripletloss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)  
                                                            
        self.hard_factor = hard_factor

    def forward(self, inputs, targets):
        """
        Args:
            inputs: feature matrix with shape (batch_size, feat_dim)
            targets: ground truth labels with shape (num_classes)
        """

        n = inputs.size(0)

        inputs = normalize(inputs, axis=-1)

        dist = euclidean_dist(inputs, inputs)
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []

        for i in range(n):
            if i < n/2:
                dist_ap.append(dist[i][int(n/2):n][mask[i][int(n/2):n]].max().unsqueeze(0))
                dist_an.append(dist[i][int(n/2):n][(mask[i] == 0)[int(n/2):n]].min().unsqueeze(0))
            else:
                dist_ap.append(dist[i][0:int(n/2)][mask[i][0:int(n/2)]].max().unsqueeze(0))
                dist_an.append(dist[i][0:int(n/2)][(mask[i] == 0)[0:int(n/2)]].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        dist_ap *= (1.0 + self.hard_factor)
        dist_an *= (1.0 - self.hard_factor)
        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)

        loss = self.ranking_loss(dist_an, dist_ap, y)
        return loss