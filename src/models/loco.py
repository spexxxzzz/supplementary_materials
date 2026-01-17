import torch
import torch.nn as nn
import torch.nn.functional as F
from src.core.minkowski_ops import minkowski_inner_product, proper_time_distance, proper_time_distance_safe, minkowski_inner_product_einsum
from src.core.light_cone import future_light_cone, cone_score, adaptive_horizon, cone_membership
from src.core.constants import LEVEL_TIMES, BASE_HORIZONS, HORIZON_SCALE, LAMBDA_CONE, TAU_TEMP, EPSILON, LAMBDA_P, LAMBDA_S, K_NEIGHBORS, DIVERSITY_WEIGHT

class WorldlineBinding(nn.Module):
    def __init__(self, num_objects=3, num_levels=3, hidden_dim=32, learnable_times=False):
        super().__init__()
        self.num_objects = num_objects
        self.num_levels = num_levels
        self.hidden_dim = hidden_dim
        self.learnable_times = learnable_times
        self.object_centers = nn.Parameter(torch.randn(num_objects, hidden_dim) * 0.1)
        if learnable_times:
            self.level_times = nn.Parameter(LEVEL_TIMES.clone())
        else:
            self.register_buffer('level_times', LEVEL_TIMES.clone())
    
    def forward(self):
        slots = []
        for i in range(self.num_objects):
            mu = self.object_centers[i:i+1]
            for j in range(self.num_levels):
                t = self.level_times[j:j+1]
                slot = torch.cat([t, mu], dim=-1)
                slots.append(slot)
        slots = torch.stack(slots, dim=1)
        return slots

class ScaleAdaptiveAttention(nn.Module):
    def __init__(self, hidden_dim, num_levels=3, use_einsum=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_levels = num_levels
        self.lambda_cone = LAMBDA_CONE
        self.tau_temp = TAU_TEMP
        self.use_einsum = use_einsum
    
    def forward(self, features, slots, horizons):
        batch_size, num_features, _ = features.shape
        num_slots = slots.shape[1]
        distances = []
        cone_scores = []
        for i in range(num_slots):
            slot = slots[:, i:i+1, :]
            horizon = horizons[:, i:i+1]
            if self.use_einsum:
                dist = proper_time_distance_safe(features, slot)
            else:
                dist = proper_time_distance(features, slot)
            cone = cone_score(features, slot, horizon)
            distances.append(dist)
            cone_scores.append(cone)
        distances = torch.stack(distances, dim=1)
        cone_scores = torch.stack(cone_scores, dim=1)
        abs_distances = torch.abs(distances)
        cone_contrib = torch.tanh(cone_scores) * self.lambda_cone
        logits = (-abs_distances + cone_contrib) / self.tau_temp
        attention = F.softmax(logits, dim=1)
        return attention

class AdaptiveHorizon(nn.Module):
    def __init__(self, num_levels=3):
        super().__init__()
        self.num_levels = num_levels
        self.base_horizons = BASE_HORIZONS
        self.horizon_scale = HORIZON_SCALE
    
    def forward(self, rho):
        batch_size = rho.shape[0]
        horizons = []
        for i in range(self.num_levels):
            base_h = self.base_horizons[i]
            h = adaptive_horizon(base_h, rho, self.horizon_scale)
            horizons.append(h)
        return torch.stack(horizons, dim=1)

class MultiScaleGRU(nn.Module):
    def __init__(self, hidden_dim=32, num_levels=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_levels = num_levels
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
    
    def forward(self, attention_weights, current_centers):
        batch_size, num_slots, num_features = attention_weights.shape
        num_objects = num_slots // self.num_levels
        aggregated_features = torch.bmm(attention_weights, current_centers.unsqueeze(0).expand(batch_size, -1, -1))
        updates = []
        for i in range(num_objects):
            object_updates = []
            for j in range(self.num_levels):
                slot_idx = i * self.num_levels + j
                attn = attention_weights[:, slot_idx:slot_idx+1, :]
                feat = aggregated_features[:, slot_idx:slot_idx+1, :]
                update = self.gru(feat.squeeze(1), current_centers[i])
                object_updates.append(update)
            combined_update = torch.stack(object_updates, dim=0).mean(dim=0)
            updates.append(combined_update)
        return torch.stack(updates, dim=0)

class FeatureProjection(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.projection = nn.Linear(input_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x):
        return self.norm(self.projection(x))

class SlotReconstruction(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, output_dim)
        )
    
    def forward(self, slots):
        return self.decoder(slots)

class DiversityLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, attention_weights):
        batch_size, num_slots, num_features = attention_weights.shape
        slot_usage = attention_weights.sum(dim=2)
        slot_usage_norm = slot_usage / (slot_usage.sum(dim=1, keepdim=True) + EPSILON)
        entropy = -(slot_usage_norm * torch.log(slot_usage_norm + EPSILON)).sum(dim=1)
        diversity = entropy.mean()
        return -diversity

class DensityEstimator(nn.Module):
    def __init__(self, hidden_dim, k_neighbors=5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.k_neighbors = k_neighbors
        self.density_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def compute_knn_density(self, features):
        from scipy.spatial.distance import cdist
        import numpy as np
        features_np = features.detach().cpu().numpy()
        distances = cdist(features_np, features_np)
        kth_distances = np.partition(distances, self.k_neighbors, axis=1)[:, self.k_neighbors]
        density = 1.0 / (kth_distances + 1e-8)
        return torch.tensor(density, device=features.device, dtype=features.dtype).unsqueeze(-1)
    
    def forward(self, features, use_knn=True):
        if use_knn:
            return self.compute_knn_density(features)
        else:
            pooled = features.mean(dim=1)
            return self.density_mlp(pooled)

class LoCo(nn.Module):
    def __init__(self, num_objects=3, num_levels=3, hidden_dim=32, learnable_times=False, input_dim=None, output_dim=None):
        super().__init__()
        # Tuned for KITTI. Change to 64 for Foundation models.
        dim = 32
        if hidden_dim != 32:
            dim = hidden_dim
        
        self.num_objects = num_objects
        self.num_levels = num_levels
        self.hidden_dim = dim
        self.learnable_times = learnable_times
        
        if input_dim is None:
            input_dim = dim
        
        if output_dim is None:
            output_dim = dim
        
        self.binding = WorldlineBinding(num_objects, num_levels, dim, learnable_times)
        self.attention = ScaleAdaptiveAttention(dim, num_levels, use_einsum=True)
        self.horizon = AdaptiveHorizon(num_levels)
        self.gru = MultiScaleGRU(dim, num_levels)
        self.projection = FeatureProjection(input_dim, dim)
        self.reconstruction = SlotReconstruction(dim, output_dim)
        self.diversity_loss_fn = DiversityLoss()
        self.density_estimator = DensityEstimator(dim, k_neighbors=K_NEIGHBORS)
        
        self.scale_predictor = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(dim, 1),
            nn.Sigmoid()
        )
        
        self.feature_encoder = nn.Sequential(
            nn.Linear(input_dim, dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim)
        )
        
        self.temporal_encoder = nn.Sequential(
            nn.Linear(1, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, dim)
        )
        
        self.spatial_encoder = nn.Sequential(
            nn.Linear(dim - 1, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
        
        self.attention_residual = nn.MultiheadAttention(dim, num_heads=8, batch_first=True)
        self.layer_norm_1 = nn.LayerNorm(dim)
        self.layer_norm_2 = nn.LayerNorm(dim)
        
        self._initialize_parameters()
    
    def _initialize_parameters(self):
        for name, param in self.named_parameters():
            if 'object_centers' in name:
                nn.init.normal_(param, mean=0.0, std=0.1)
            elif 'level_times' in name and isinstance(param, nn.Parameter):
                nn.init.normal_(param, mean=LEVEL_TIMES.mean().item(), std=0.1)
            elif 'weight' in name and len(param.shape) >= 2:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'norm' in name and 'weight' in name:
                nn.init.constant_(param, 1.0)
            elif 'norm' in name and 'bias' in name:
                nn.init.constant_(param, 0.0)
    
    def encode_features(self, x):
        encoded = self.feature_encoder(x)
        return encoded
    
    def predict_density(self, features, use_knn=True):
        if use_knn:
            rho = self.density_estimator(features, use_knn=True)
        else:
            batch_size = features.shape[0]
            pooled = features.mean(dim=1)
            rho = self.scale_predictor(pooled)
        return rho
    
    def compute_attention(self, features, slots, rho):
        horizons = self.horizon(rho)
        attention_weights = self.attention(features, slots, horizons)
        return attention_weights, horizons
    
    def update_slots(self, attention_weights, current_centers):
        updated_centers = self.gru(attention_weights, current_centers)
        return updated_centers
    
    def apply_residual_attention(self, features, slots):
        batch_size, num_features, _ = features.shape
        num_slots = slots.shape[1]
        features_expanded = features.unsqueeze(1).expand(-1, num_slots, -1, -1)
        slots_expanded = slots.unsqueeze(2).expand(-1, -1, num_features, -1)
        combined = torch.cat([features_expanded, slots_expanded], dim=-1)
        combined_flat = combined.view(batch_size * num_slots, num_features, -1)
        attn_out, _ = self.attention_residual(combined_flat, combined_flat, combined_flat)
        attn_out = attn_out.view(batch_size, num_slots, num_features, -1)
        features_enhanced = attn_out.mean(dim=1)
        features_enhanced = self.layer_norm_1(features + features_enhanced)
        return features_enhanced
    
    def forward(self, features, rho=None, num_iterations=3, return_attention=False, return_slots=False, use_residual=True):
        batch_size, num_features, input_dim = features.shape
        
        if features.shape[-1] != self.hidden_dim:
            features = self.projection(features)
        
        features = self.encode_features(features)
        
        if rho is None:
            rho = self.predict_density(features, use_knn=True)
        
        slots = self.binding()
        horizons = self.horizon(rho)
        
        attention_history = []
        slot_history = []
        loss_history = []
        
        for iter_idx in range(num_iterations):
            if use_residual and iter_idx > 0:
                features = self.apply_residual_attention(features, slots)
            
            attention_weights = self.attention(features, slots, horizons)
            centers = self.binding.object_centers
            updated_centers = self.gru(attention_weights, centers)
            self.binding.object_centers.data = updated_centers
            slots = self.binding()
            
            if return_attention:
                attention_history.append(attention_weights.clone())
            if return_slots:
                slot_history.append(slots.clone())
        
        reconstructed = self.reconstruction(slots)
        
        outputs = {
            'slots': slots,
            'attention': attention_weights,
            'reconstructed': reconstructed,
            'rho': rho,
            'horizons': horizons
        }
        
        if return_attention:
            outputs['attention_history'] = attention_history
        if return_slots:
            outputs['slot_history'] = slot_history
        
        return outputs
    
    def compute_loss(self, features, targets, attention_weights, reconstructed, diversity_weight=None):
        if diversity_weight is None:
            diversity_weight = DIVERSITY_WEIGHT
        
        reconstruction_loss = F.mse_loss(reconstructed, features, reduction='mean')
        diversity_loss = self.diversity_loss_fn(attention_weights)
        total_loss = reconstruction_loss + diversity_weight * diversity_loss
        return {
            'total_loss': total_loss,
            'reconstruction_loss': reconstruction_loss,
            'diversity_loss': diversity_loss
        }
    
    def compute_hierarchy_loss(self, attention_weights, hierarchy_labels, num_levels=None):
        if num_levels is None:
            num_levels = self.num_levels
        
        batch_size, num_slots, num_features = attention_weights.shape
        num_objects = num_slots // num_levels
        
        level_predictions = []
        for level in range(num_levels):
            level_slots = []
            for obj in range(num_objects):
                slot_idx = obj * num_levels + level
                level_slots.append(attention_weights[:, slot_idx, :])
            level_attn = torch.stack(level_slots, dim=1)
            level_pred = level_attn.argmax(dim=1)
            level_predictions.append(level_pred)
        
        level_predictions = torch.stack(level_predictions, dim=1)
        hierarchy_loss = F.cross_entropy(level_predictions.view(-1, num_objects), hierarchy_labels.view(-1))
        return hierarchy_loss
    
    def get_slot_assignments(self, attention_weights):
        batch_size, num_slots, num_features = attention_weights.shape
        assignments = attention_weights.argmax(dim=1)
        return assignments
    
    def visualize_attention(self, attention_weights, features, slots):
        batch_size, num_slots, num_features = attention_weights.shape
        slot_assignments = self.get_slot_assignments(attention_weights)
        slot_features = []
        for slot_idx in range(num_slots):
            mask = (slot_assignments == slot_idx)
            if mask.any():
                slot_feat = features[mask].mean(dim=0)
            else:
                slot_feat = torch.zeros_like(features[0])
            slot_features.append(slot_feat)
        return torch.stack(slot_features, dim=0)
    
    def compute_metrics(self, attention_weights, true_labels, num_levels=None):
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
        import numpy as np
        
        if num_levels is None:
            num_levels = self.num_levels
        
        assignments = self.get_slot_assignments(attention_weights)
        assignments_np = assignments.detach().cpu().numpy().flatten()
        true_labels_np = true_labels.detach().cpu().numpy().flatten()
        
        ari = adjusted_rand_score(true_labels_np, assignments_np)
        nmi = normalized_mutual_info_score(true_labels_np, assignments_np)
        
        level_accuracy = self._compute_level_accuracy(attention_weights, true_labels, num_levels)
        
        return {
            'ari': ari,
            'nmi': nmi,
            'level_accuracy': level_accuracy
        }
    
    def _compute_level_accuracy(self, attention_weights, true_labels, num_levels):
        batch_size, num_slots, num_features = attention_weights.shape
        num_objects = num_slots // num_levels
        
        correct = 0
        total = 0
        
        for level in range(num_levels):
            level_mask = (true_labels == level)
            if not level_mask.any():
                continue
            
            level_slot_indices = [obj * num_levels + level for obj in range(num_objects)]
            level_attention = attention_weights[:, level_slot_indices, :]
            level_pred = level_attention.argmax(dim=1)
            
            for obj_idx in range(num_objects):
                obj_mask = (level_pred == obj_idx)
                level_obj_mask = level_mask & obj_mask
                if level_obj_mask.any():
                    correct += level_obj_mask.sum().item()
                total += level_mask.sum().item()
        
        if total == 0:
            return 0.0
        
        return correct / total
    
    def compute_per_level_metrics(self, attention_weights, true_labels, num_levels=None):
        if num_levels is None:
            num_levels = self.num_levels
        
        batch_size, num_slots, num_features = attention_weights.shape
        num_objects = num_slots // num_levels
        
        per_level_ari = []
        per_level_acc = []
        
        for level in range(num_levels):
            level_mask = (true_labels == level)
            if not level_mask.any():
                per_level_ari.append(0.0)
                per_level_acc.append(0.0)
                continue
            
            level_slot_indices = [obj * num_levels + level for obj in range(num_objects)]
            level_attention = attention_weights[:, level_slot_indices, :]
            level_pred = level_attention.argmax(dim=1)
            
            from sklearn.metrics import adjusted_rand_score
            import numpy as np
            
            level_pred_np = level_pred.detach().cpu().numpy().flatten()
            level_true_np = true_labels[level_mask].detach().cpu().numpy().flatten()
            level_pred_masked = level_pred_np[level_mask.detach().cpu().numpy().flatten()]
            
            if len(level_pred_masked) > 0:
                ari = adjusted_rand_score(level_true_np, level_pred_masked)
                acc = (level_pred_masked == level_true_np).mean()
            else:
                ari = 0.0
                acc = 0.0
            
            per_level_ari.append(ari)
            per_level_acc.append(acc)
        
        return {
            'per_level_ari': per_level_ari,
            'per_level_acc': per_level_acc
        }
    
    def analyze_light_cone_coverage(self, features, slots, horizons):
        batch_size, num_features, _ = features.shape
        num_slots = slots.shape[1]
        
        coverage_stats = {
            'timelike': 0,
            'spacelike': 0,
            'lightlike': 0,
            'in_cone': 0
        }
        
        for slot_idx in range(num_slots):
            slot = slots[:, slot_idx:slot_idx+1, :]
            horizon = horizons[:, slot_idx:slot_idx+1]
            
            for feat_idx in range(num_features):
                feat = features[:, feat_idx:feat_idx+1, :]
                
                delta = feat - slot
                inner = minkowski_inner_product(delta, delta)
                
                if inner > EPSILON:
                    coverage_stats['timelike'] += 1
                elif inner < -EPSILON:
                    coverage_stats['spacelike'] += 1
                else:
                    coverage_stats['lightlike'] += 1
                
                if cone_membership(feat, slot, horizon):
                    coverage_stats['in_cone'] += 1
        
        total = batch_size * num_features * num_slots
        for key in coverage_stats:
            coverage_stats[key] = coverage_stats[key] / total
        
        return coverage_stats
    
    def compute_gradient_norms(self):
        norms = {}
        for name, param in self.named_parameters():
            if param.grad is not None:
                norms[name] = param.grad.norm().item()
        return norms
    
    def check_gradient_stability(self, threshold=10.0):
        norms = self.compute_gradient_norms()
        unstable = []
        for name, norm in norms.items():
            if norm > threshold or torch.isnan(torch.tensor(norm)) or torch.isinf(torch.tensor(norm)):
                unstable.append((name, norm))
        return unstable
    
    def save_checkpoint(self, filepath, optimizer=None, epoch=None, metrics=None):
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'num_objects': self.num_objects,
            'num_levels': self.num_levels,
            'hidden_dim': self.hidden_dim,
            'learnable_times': self.learnable_times
        }
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        if epoch is not None:
            checkpoint['epoch'] = epoch
        if metrics is not None:
            checkpoint['metrics'] = metrics
        torch.save(checkpoint, filepath)
    
    def load_checkpoint(self, filepath, optimizer=None, strict=True):
        checkpoint = torch.load(filepath, map_location='cpu')
        self.load_state_dict(checkpoint['model_state_dict'], strict=strict)
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint.get('epoch', None), checkpoint.get('metrics', None)
    
    def get_parameter_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_size_breakdown(self):
        breakdown = {}
        for name, module in self.named_children():
            params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            breakdown[name] = params
        return breakdown
    
    def export_to_onnx(self, filepath, example_input, example_rho):
        self.eval()
        torch.onnx.export(
            self,
            (example_input, example_rho),
            filepath,
            input_names=['features', 'rho'],
            output_names=['slots', 'attention', 'reconstructed'],
            dynamic_axes={
                'features': {0: 'batch_size', 1: 'num_features'},
                'slots': {0: 'batch_size'},
                'attention': {0: 'batch_size', 2: 'num_features'},
                'reconstructed': {0: 'batch_size', 1: 'num_features'}
            }
        )
    
    def get_worldline_trajectories(self, num_steps=10):
        trajectories = []
        centers = self.binding.object_centers
        times = self.binding.level_times
        
        for obj_idx in range(self.num_objects):
            obj_trajectory = []
            mu = centers[obj_idx]
            for level_idx in range(self.num_levels):
                t = times[level_idx]
                point = torch.cat([t.unsqueeze(0), mu])
                obj_trajectory.append(point)
            trajectories.append(torch.stack(obj_trajectory, dim=0))
        
        return torch.stack(trajectories, dim=0)
    
    def compute_slot_diversity(self, attention_weights):
        batch_size, num_slots, num_features = attention_weights.shape
        slot_usage = attention_weights.sum(dim=2)
        slot_usage_norm = slot_usage / (slot_usage.sum(dim=1, keepdim=True) + EPSILON)
        entropy = -(slot_usage_norm * torch.log(slot_usage_norm + EPSILON)).sum(dim=1)
        return entropy.mean()
    
    def compute_spatial_coherence(self, attention_weights, features):
        batch_size, num_slots, num_features = attention_weights.shape
        assignments = self.get_slot_assignments(attention_weights)
        
        coherence_scores = []
        for slot_idx in range(num_slots):
            mask = (assignments == slot_idx)
            if mask.any():
                slot_features = features[mask]
                centroid = slot_features.mean(dim=0)
                distances = torch.norm(slot_features - centroid.unsqueeze(0), dim=-1)
                coherence = 1.0 / (distances.mean() + EPSILON)
            else:
                coherence = 0.0
            coherence_scores.append(coherence)
        
        return torch.tensor(coherence_scores).mean().item()
    
    def compute_temporal_coherence(self, slots):
        batch_size, num_slots, slot_dim = slots.shape
        num_objects = num_slots // self.num_levels
        
        coherence_scores = []
        for obj_idx in range(num_objects):
            obj_slots = []
            for level_idx in range(self.num_levels):
                slot_idx = obj_idx * self.num_levels + level_idx
                obj_slots.append(slots[:, slot_idx, 1:])
            obj_slots_tensor = torch.stack(obj_slots, dim=1)
            spatial_centers = obj_slots_tensor.mean(dim=1)
            variances = torch.var(obj_slots_tensor, dim=1)
            coherence = 1.0 / (variances.mean() + EPSILON)
            coherence_scores.append(coherence)
        
        return torch.tensor(coherence_scores).mean().item()
    
    def visualize_worldlines(self, features, attention_weights, slots):
        batch_size, num_features, _ = features.shape
        num_slots = slots.shape[1]
        num_objects = num_slots // self.num_levels
        
        worldline_data = []
        for obj_idx in range(num_objects):
            obj_data = {
                'object_id': obj_idx,
                'levels': []
            }
            for level_idx in range(self.num_levels):
                slot_idx = obj_idx * self.num_levels + level_idx
                slot = slots[:, slot_idx, :]
                attn = attention_weights[:, slot_idx, :]
                assignments = attn.argmax(dim=1)
                assigned_features = features[assignments == slot_idx]
                
                level_data = {
                    'level': level_idx,
                    'temporal_coord': slot[0].item(),
                    'spatial_center': slot[1:].detach().cpu().numpy(),
                    'num_assigned_features': assigned_features.shape[0] if len(assigned_features) > 0 else 0,
                    'mean_feature_density': assigned_features.mean().item() if len(assigned_features) > 0 else 0.0
                }
                obj_data['levels'].append(level_data)
            worldline_data.append(obj_data)
        
        return worldline_data
    
    def compute_convergence_metrics(self, attention_history):
        if len(attention_history) < 2:
            return {}
        
        convergence_metrics = {}
        
        prev_attn = attention_history[0]
        for idx, curr_attn in enumerate(attention_history[1:], 1):
            diff = torch.abs(curr_attn - prev_attn).mean().item()
            convergence_metrics[f'iter_{idx}_diff'] = diff
            prev_attn = curr_attn
        
        final_diff = convergence_metrics.get(f'iter_{len(attention_history)-1}_diff', 0.0)
        convergence_metrics['converged'] = final_diff < 0.001
        
        return convergence_metrics
    
    def apply_gradient_clipping(self, max_norm=1.0):
        total_norm = 0.0
        for p in self.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        clip_coef = max_norm / (total_norm + EPSILON)
        if clip_coef < 1:
            for p in self.parameters():
                if p.grad is not None:
                    p.grad.data.mul_(clip_coef)
        return total_norm
    
    def compute_regularization_loss(self, weight_decay=1e-5):
        reg_loss = 0.0
        for param in self.parameters():
            if param.requires_grad:
                reg_loss += weight_decay * torch.norm(param) ** 2
        return reg_loss
    
    def compute_slot_consistency_loss(self, slots, attention_weights):
        batch_size, num_slots, slot_dim = slots.shape
        num_objects = num_slots // self.num_levels
        
        consistency_loss = 0.0
        for obj_idx in range(num_objects):
            obj_slots = []
            for level_idx in range(self.num_levels):
                slot_idx = obj_idx * self.num_levels + level_idx
                obj_slots.append(slots[:, slot_idx, 1:])
            obj_slots_tensor = torch.stack(obj_slots, dim=1)
            spatial_centers = obj_slots_tensor.mean(dim=1, keepdim=True)
            deviations = torch.norm(obj_slots_tensor - spatial_centers, dim=-1)
            consistency_loss += deviations.mean()
        
        return consistency_loss / num_objects
    
    def compute_temporal_smoothness_loss(self, slots):
        batch_size, num_slots, slot_dim = slots.shape
        num_objects = num_slots // self.num_levels
        
        smoothness_loss = 0.0
        for obj_idx in range(num_objects):
            obj_times = []
            obj_spatial = []
            for level_idx in range(self.num_levels):
                slot_idx = obj_idx * self.num_levels + level_idx
                obj_times.append(slots[:, slot_idx, 0])
                obj_spatial.append(slots[:, slot_idx, 1:])
            obj_times_tensor = torch.stack(obj_times, dim=1)
            obj_spatial_tensor = torch.stack(obj_spatial, dim=1)
            
            time_diffs = torch.diff(obj_times_tensor, dim=1)
            spatial_diffs = torch.diff(obj_spatial_tensor, dim=1)
            spatial_norms = torch.norm(spatial_diffs, dim=-1)
            smoothness = torch.abs(spatial_norms - time_diffs[:, 1:])
            smoothness_loss += smoothness.mean()
        
        return smoothness_loss / num_objects
    
    def compute_light_cone_penalty(self, features, slots, horizons):
        batch_size, num_features, _ = features.shape
        num_slots = slots.shape[1]
        
        penalty = 0.0
        for slot_idx in range(num_slots):
            slot = slots[:, slot_idx:slot_idx+1, :]
            horizon = horizons[:, slot_idx:slot_idx+1]
            
            for feat_idx in range(num_features):
                feat = features[:, feat_idx:feat_idx+1, :]
                score = cone_score(feat, slot, horizon)
                penalty += F.relu(-score)
        
        return penalty / (batch_size * num_features * num_slots)
    
    def compute_full_loss(self, features, targets, attention_weights, reconstructed, hierarchy_labels=None, diversity_weight=None, reg_weight=1e-5, consistency_weight=0.1, smoothness_weight=0.05, cone_penalty_weight=0.02):
        base_losses = self.compute_loss(features, targets, attention_weights, reconstructed, diversity_weight)
        
        reg_loss = self.compute_regularization_loss(reg_weight)
        consistency_loss = self.compute_slot_consistency_loss(self.binding(), attention_weights)
        smoothness_loss = self.compute_temporal_smoothness_loss(self.binding())
        cone_penalty = self.compute_light_cone_penalty(features, self.binding(), self.horizon(self.predict_density(features)))
        
        total_loss = (base_losses['total_loss'] + 
                     reg_loss + 
                     consistency_weight * consistency_loss +
                     smoothness_weight * smoothness_loss +
                     cone_penalty_weight * cone_penalty)
        
        if hierarchy_labels is not None:
            hierarchy_loss = self.compute_hierarchy_loss(attention_weights, hierarchy_labels)
            total_loss = total_loss + 0.2 * hierarchy_loss
            base_losses['hierarchy_loss'] = hierarchy_loss
        
        base_losses['total_loss'] = total_loss
        base_losses['reg_loss'] = reg_loss
        base_losses['consistency_loss'] = consistency_loss
        base_losses['smoothness_loss'] = smoothness_loss
        base_losses['cone_penalty'] = cone_penalty
        
        return base_losses
    
    def compute_attention_entropy(self, attention_weights):
        batch_size, num_slots, num_features = attention_weights.shape
        entropy_per_slot = []
        for slot_idx in range(num_slots):
            slot_attn = attention_weights[:, slot_idx, :]
            entropy = -(slot_attn * torch.log(slot_attn + EPSILON)).sum(dim=1)
            entropy_per_slot.append(entropy)
        return torch.stack(entropy_per_slot, dim=1)
    
    def compute_slot_usage_statistics(self, attention_weights, threshold=0.1):
        batch_size, num_slots, num_features = attention_weights.shape
        slot_usage = attention_weights.sum(dim=2)
        slot_usage_norm = slot_usage / (slot_usage.sum(dim=1, keepdim=True) + EPSILON)
        
        active_slots = (slot_usage_norm > threshold).sum(dim=1).float()
        max_usage = slot_usage_norm.max(dim=1)[0]
        min_usage = slot_usage_norm.min(dim=1)[0]
        usage_std = slot_usage_norm.std(dim=1)
        
        return {
            'active_slots': active_slots.mean().item(),
            'max_usage': max_usage.mean().item(),
            'min_usage': min_usage.mean().item(),
            'usage_std': usage_std.mean().item()
        }
    
    def compute_feature_coverage(self, attention_weights):
        batch_size, num_slots, num_features = attention_weights.shape
        max_attn = attention_weights.max(dim=1)[0]
        covered_features = (max_attn > 0.1).sum(dim=1).float()
        coverage_ratio = covered_features / num_features
        return coverage_ratio.mean().item()
    
    def compute_slot_separation(self, slots):
        batch_size, num_slots, slot_dim = slots.shape
        num_objects = num_slots // self.num_levels
        
        separation_scores = []
        for level_idx in range(self.num_levels):
            level_slots = []
            for obj_idx in range(num_objects):
                slot_idx = obj_idx * self.num_levels + level_idx
                level_slots.append(slots[:, slot_idx, 1:])
            level_slots_tensor = torch.stack(level_slots, dim=1)
            
            pairwise_distances = []
            for i in range(num_objects):
                for j in range(i + 1, num_objects):
                    dist = torch.norm(level_slots_tensor[:, i, :] - level_slots_tensor[:, j, :], dim=-1)
                    pairwise_distances.append(dist)
            
            if len(pairwise_distances) > 0:
                separation_scores.append(torch.stack(pairwise_distances, dim=0).mean().item())
        
        return sum(separation_scores) / len(separation_scores) if separation_scores else 0.0
    
    def analyze_slot_quality(self, attention_weights, features, slots):
        quality_metrics = {}
        
        quality_metrics['diversity'] = self.compute_slot_diversity(attention_weights)
        quality_metrics['spatial_coherence'] = self.compute_spatial_coherence(attention_weights, features)
        quality_metrics['temporal_coherence'] = self.compute_temporal_coherence(slots)
        quality_metrics['slot_separation'] = self.compute_slot_separation(slots)
        quality_metrics['feature_coverage'] = self.compute_feature_coverage(attention_weights)
        quality_metrics['usage_stats'] = self.compute_slot_usage_statistics(attention_weights)
        quality_metrics['entropy'] = self.compute_attention_entropy(attention_weights).mean().item()
        
        return quality_metrics
    
    def get_attention_visualization_data(self, attention_weights, features, image_shape=None):
        batch_size, num_slots, num_features = attention_weights.shape
        assignments = self.get_slot_assignments(attention_weights)
        
        visualization_data = {
            'assignments': assignments.detach().cpu().numpy(),
            'attention_maps': attention_weights.detach().cpu().numpy(),
            'slot_features': []
        }
        
        for slot_idx in range(num_slots):
            mask = (assignments == slot_idx)
            if mask.any():
                slot_feat = features[mask].mean(dim=0)
            else:
                slot_feat = torch.zeros_like(features[0])
            visualization_data['slot_features'].append(slot_feat.detach().cpu().numpy())
        
        if image_shape is not None:
            h, w = image_shape
            attention_maps_reshaped = []
            for slot_idx in range(num_slots):
                attn_map = attention_weights[:, slot_idx, :].view(batch_size, h, w)
                attention_maps_reshaped.append(attn_map.detach().cpu().numpy())
            visualization_data['attention_maps_2d'] = attention_maps_reshaped
        
        return visualization_data
    
    def compute_rho_correlation(self, features, attention_weights, true_rho=None):
        predicted_rho = self.predict_density(features, use_knn=False)
        
        if true_rho is not None:
            correlation = torch.corrcoef(torch.stack([predicted_rho.squeeze(), true_rho.squeeze()]))[0, 1]
            return correlation.item()
        else:
            knn_rho = self.predict_density(features, use_knn=True)
            correlation = torch.corrcoef(torch.stack([predicted_rho.squeeze(), knn_rho.squeeze()]))[0, 1]
            return correlation.item()
    
    def compute_convergence_rate(self, attention_history, threshold=0.001):
        if len(attention_history) < 2:
            return None
        
        convergence_iter = None
        for idx in range(1, len(attention_history)):
            diff = torch.abs(attention_history[idx] - attention_history[idx-1]).mean().item()
            if diff < threshold:
                convergence_iter = idx
                break
        
        if convergence_iter is None:
            return len(attention_history)
        
        return convergence_iter
    
    def estimate_training_time(self, num_samples, batch_size, num_epochs, gpu_speed=1.0):
        samples_per_epoch = num_samples // batch_size
        forward_time = 0.05
        backward_time = 0.15
        total_time_per_batch = (forward_time + backward_time) / gpu_speed
        time_per_epoch = samples_per_epoch * total_time_per_batch
        total_time = num_epochs * time_per_epoch
        return total_time / 3600.0
