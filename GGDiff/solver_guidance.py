import torch
import numpy as np
import abc
from tqdm import trange

from losses import get_score_fn
from utils.graph_utils import mask_adjs, mask_x, gen_noise
from sde import VPSDE, subVPSDE
import losses_guidance
from prodigy.project_bisection import drifted_project

class Predictor(abc.ABC):
    """The abstract class for a predictor algorithm."""
    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__()
        self.sde = sde
        # Compute the reverse SDE/ODE
        self.rsde = sde.reverse(score_fn, probability_flow)
        self.score_fn = score_fn

    @abc.abstractmethod
    def update_fn(self, x, t, flags):
        pass


class Corrector(abc.ABC):
    """The abstract class for a corrector algorithm."""
    def __init__(self, sde, score_fn, snr, scale_eps, n_steps):
        super().__init__()
        self.sde = sde
        self.score_fn = score_fn
        self.snr = snr
        self.scale_eps = scale_eps
        self.n_steps = n_steps

    @abc.abstractmethod
    def update_fn(self, x, t, flags):
        pass


class EulerMaruyamaPredictor(Predictor):
    def __init__(self, obj, sde, score_fn, probability_flow=False, guidance_args=None):
        super().__init__(sde, score_fn, probability_flow)
        self.obj = obj

        self.guidance_args = guidance_args

    def guidance(self, x, adj, flags, t, is_adj):
        obj = adj if is_adj else x

        dt = -1. / self.rsde.N
        timestep = (t * (self.rsde.N - 1) / self.rsde.T).long()

        loss_fn = getattr(losses_guidance, self.guidance_args.loss_fn)
        loss_kwargs = self.guidance_args.get('loss_kwargs', {})
        if "flags" in loss_kwargs:
            loss_kwargs = getattr(losses_guidance, loss_kwargs['flags'])(flags)
        
        if self.guidance_args.method == 'greedy':
            n_traj = self.guidance_args.n_traj

            drift, diffusion = self.rsde.sde(x, adj, flags, t, is_adj=is_adj)
            
            obj_mean = obj + drift * dt

            losses = torch.zeros(n_traj, obj.shape[0])
            obj_hats = []
            for i in range(n_traj):
                z = gen_noise(obj, flags, sym=is_adj)
                obj_hat = obj_mean.clone() + diffusion[:, None, None] * np.sqrt(-dt) * z
                obj_hats.append(obj_hat)
                
                score_i = self.score_fn(x, obj_hat, flags, t+dt) if is_adj else self.score_fn(obj_hat, adj, flags, t+dt)

                obj0hat = self.sde.obj0estimation(obj_hat, score_i, timestep)
                obj0hat_masked = mask_adjs(obj0hat, flags) if is_adj else mask_x(obj0hat, flags)
                
                losses[i,:] = loss_fn(x, obj0hat_masked, **loss_kwargs) if is_adj else loss_fn(obj0hat_masked, adj, **loss_kwargs)

            losses_expanded = torch.argmin(losses, dim=0).view(1, obj.shape[0], 1, 1).expand(1, obj.shape[0], obj.shape[1], obj.shape[2]).to(obj.device)

            return torch.gather(torch.stack(obj_hats, dim=0), 0, losses_expanded).squeeze(0), obj_mean

        elif self.guidance_args.method == 'zero':
            n_traj = self.guidance_args.n_traj

            drift, diffusion = self.rsde.sde(x, adj, flags, t, is_adj=is_adj)
            
            obj_mean = obj + drift * dt

            z = gen_noise(obj, flags, sym=is_adj)
            obj = obj_mean.clone() + diffusion[:, None, None] * np.sqrt(-dt) * z

            score_no_noise = self.score_fn(x, obj.clone(), flags, t+dt) if is_adj else self.score_fn(obj.clone(), adj, flags, t)
            obj0hat_no_noise = self.sde.obj0estimation(obj.clone(), score_no_noise, timestep)
            obj0hat_masked_no_noise = mask_adjs(obj0hat_no_noise, flags) if is_adj else mask_x(obj0hat_no_noise, flags)
            no_noise_loss = loss_fn(x, obj0hat_masked_no_noise, **loss_kwargs) if is_adj else loss_fn(obj0hat_masked_no_noise, adj, **loss_kwargs)

            losses = torch.zeros(n_traj, obj.shape[0], device=obj.device)
            noise_directions = []
            for i in range(n_traj):
                z = gen_noise(obj, flags, sym=is_adj)
                obj_hat = obj_mean.clone() + self.guidance_args.delta * z
                noise_directions.append(z)
                
                score_i = self.score_fn(x, obj_hat, flags, t+dt) if is_adj else self.score_fn(obj_hat, adj, flags, t+dt)

                obj0hat = self.sde.obj0estimation(obj_hat, score_i, timestep)
                obj0hat_masked = mask_adjs(obj0hat, flags) if is_adj else mask_x(obj0hat, flags)
                
                losses[i,:] = loss_fn(x, obj0hat_masked, **loss_kwargs) if is_adj else loss_fn(obj0hat_masked, adj, **loss_kwargs)

            weights = (losses - no_noise_loss[None,:]) / self.guidance_args.delta # (diffusion[None, :] * np.sqrt(-dt))
            directions = torch.stack(noise_directions, dim=0)
            weighted_directions = (weights[:,:,None,None] * directions).mean(dim=0)

            if self.guidance_args.clip_method == "clip":
                # Gradient clipping
                weighted_directions = torch.clamp(weighted_directions, -self.guidance_args.clip, self.guidance_args.clip)

                obj = obj_mean - self.guidance_args.lr_zero * weighted_directions
            elif self.guidance_args.clip_method == "norm":
                norm_grad_step = torch.linalg.norm(weighted_directions, dim=(1,2)) / (weighted_directions.shape[1] * weighted_directions.shape[2])
                norm_grad_step = torch.where(norm_grad_step < 1e-7, torch.ones_like(norm_grad_step), norm_grad_step)

                obj = obj_mean - self.guidance_args.lr_zero * weighted_directions / norm_grad_step[:,None,None]
            else:
                obj = obj_mean - self.guidance_args.lr_zero * weighted_directions
            return obj, obj_mean
        
        elif self.guidance_args.method == 'loss':

            with torch.enable_grad():

                obj.requires_grad = True

                score = self.score_fn(x, obj, flags, t) if is_adj else self.score_fn(obj, adj, flags, t)

                obj0hat = self.sde.obj0estimation(obj, score, timestep)
                obj0hat_masked = mask_adjs(obj0hat, flags) if is_adj else mask_x(obj0hat, flags)

                loss = loss_fn(x, obj0hat_masked, **loss_kwargs) if is_adj else loss_fn(obj0hat_masked, adj, **loss_kwargs)
                loss = loss.mean()

                loss.backward()

                obj_grad = obj.grad.detach().clone()
                obj.grad = None

            drift, diffusion = self.sde.sde(adj, t)
            drift = drift - diffusion[:, None, None] ** 2 * score * (0.5 if self.rsde.probability_flow else 1.)
            # -------- Set the diffusion function to zero for ODEs. --------
            diffusion = 0. if self.rsde.probability_flow else diffusion

            obj_mean = obj + drift * dt

            z = gen_noise(obj, flags, sym=is_adj)
            obj = obj_mean + diffusion[:, None, None] * np.sqrt(-dt) * z
            
            if self.guidance_args.lr_guidance_method == 'adaptive':
                obj -= self.guidance_args.lr_guidance / torch.abs(loss) * obj_grad
            else:
                obj -= self.guidance_args.lr_guidance * obj_grad

            return obj, obj_mean
        else:
            raise NotImplementedError(f"guidance method {self.guidance_args.method} not yet supported.")

            
    def update_fn(self, x, adj, flags, t):
        dt = -1. / self.rsde.N
        timestep = (t[0] * (self.rsde.N - 1) / self.rsde.T).long()

        var = x if self.obj == 'x' else adj
            
        if self.guidance_args is not None and \
                timestep > 0:
            var, var_mean = self.guidance(x, adj, flags, t, is_adj=self.obj == 'adj')
        else:
            z = gen_noise(var, flags, sym=self.obj == 'adj')
            drift, diffusion = self.rsde.sde(x, adj, flags, t, is_adj=self.obj == 'adj')
            var_mean = var + drift * dt
            var = var_mean + diffusion[:, None, None] * np.sqrt(-dt) * z
        return var, var_mean


class ReverseDiffusionPredictor(Predictor):
    def __init__(self, obj, sde, score_fn, probability_flow=False, guidance_args=None):
        super().__init__(sde, score_fn, probability_flow)
        self.obj = obj

        self.guidance_args = guidance_args

    def guidance(self, x, adj, flags, t, is_adj):
        obj = adj if is_adj else x

        dt = -1. / self.rsde.N
        timestep = (t * (self.rsde.N - 1) / self.rsde.T).long()

        loss_fn = getattr(losses_guidance, self.guidance_args.loss_fn)
        loss_kwargs = self.guidance_args.get('loss_kwargs', {})
        if "flags" in loss_kwargs:
            loss_kwargs = getattr(losses_guidance, loss_kwargs['flags'])(flags)
        
        if self.guidance_args.method == 'greedy':
            n_traj = self.guidance_args.n_traj

            f, G = self.rsde.discretize(x, adj, flags, t, is_adj=is_adj)
            
            obj_mean = obj - f

            losses = torch.zeros(n_traj, obj.shape[0])
            obj_hats = []
            for i in range(n_traj):
                z = gen_noise(obj, flags, sym=is_adj)
                obj_hat = obj_mean.clone() + G[:, None, None] * z
                obj_hats.append(obj_hat)
                
                score_i = self.score_fn(x, obj_hat, flags, t+dt) if is_adj else self.score_fn(obj_hat, adj, flags, t+dt)

                obj0hat = self.sde.obj0estimation(obj_hat, score_i, timestep)
                obj0hat_masked = mask_adjs(obj0hat, flags) if is_adj else mask_x(obj0hat, flags)
                
                losses[i,:] = loss_fn(x, obj0hat_masked, **loss_kwargs) if is_adj else loss_fn(obj0hat_masked, adj, **loss_kwargs)

            losses_expanded = torch.argmin(losses, dim=0).view(1, obj.shape[0], 1, 1).expand(1, obj.shape[0], obj.shape[1], obj.shape[2]).to(obj.device)

            return torch.gather(torch.stack(obj_hats, dim=0), 0, losses_expanded).squeeze(0), obj_mean

        elif self.guidance_args.method == 'zero':
            n_traj = self.guidance_args.n_traj

            f, G = self.rsde.discretize(x, adj, flags, t, is_adj=is_adj)
            
            obj_mean = obj - f

            score_no_noise = self.score_fn(x, obj_mean.clone(), flags, t) if is_adj else self.score_fn(obj.clone(), adj, flags, t)
            obj0hat_no_noise = self.sde.obj0estimation(obj_mean.clone(), score_no_noise, timestep)
            obj0hat_masked_no_noise = mask_adjs(obj0hat_no_noise, flags) if is_adj else mask_x(obj0hat_no_noise, flags)
            no_noise_loss = loss_fn(x, obj0hat_masked_no_noise, **loss_kwargs) if is_adj else loss_fn(obj0hat_masked_no_noise, adj, **loss_kwargs)

            losses = torch.zeros(n_traj, obj.shape[0], device=obj.device)
            noise_directions = []
            for i in range(n_traj):
                z = gen_noise(obj, flags, sym=is_adj)
                obj_hat = obj_mean.clone() + G[:, None, None] * z
                noise_directions.append(z)
                
                score_i = self.score_fn(x, obj_hat, flags, t+dt) if is_adj else self.score_fn(obj_hat, adj, flags, t+dt)

                obj0hat = self.sde.obj0estimation(obj_hat, score_i, timestep)
                obj0hat_masked = mask_adjs(obj0hat, flags) if is_adj else mask_x(obj0hat, flags)
                
                losses[i,:] = loss_fn(x, obj0hat_masked, **loss_kwargs) if is_adj else loss_fn(obj0hat_masked, adj, **loss_kwargs)

            weights = (losses - no_noise_loss[None,:]) / G[None, :]
            directions = torch.stack(noise_directions, dim=0)
            obj = obj_mean - self.guidance_args.lr_zero * (weights[:,:,None,None] * directions).mean(dim=0)
            return obj, obj_mean

        
        elif self.guidance_args.method == 'loss':

            with torch.enable_grad():

                obj.requires_grad = True

                score = self.score_fn(x, obj, flags, t) if is_adj else self.score_fn(obj, adj, flags, t)

                obj0hat = self.sde.obj0estimation(obj, score, timestep)

                obj0hat_masked = mask_adjs(obj0hat, flags) if is_adj else mask_x(obj0hat, flags)
                
                loss = loss_fn(x, obj0hat_masked, **loss_kwargs) if is_adj else loss_fn(obj0hat_masked, adj, **loss_kwargs)
                loss = loss.mean()

                loss.backward()

                obj_grad = obj.grad.detach().clone()
                obj.grad = None

            f, G = self.rsde.discretize(x, adj, flags, t, is_adj=is_adj)

            obj_mean = obj - f

            z = gen_noise(obj, flags, sym=is_adj)
            obj = obj_mean + G[:, None, None] * z
            
            if self.guidance_args.lr_guidance_method == 'adaptive':
                obj -= self.guidance_args.lr_guidance / torch.abs(loss) * obj_grad
            else:
                obj -= self.guidance_args.lr_guidance * obj_grad

            return obj, obj_mean
        else:
            raise NotImplementedError(f"guidance method {self.guidance_args.method} not yet supported.")


    def update_fn(self, x, adj, flags, t):
        timestep = (t[0] * (self.rsde.N - 1) / self.rsde.T).long()

        var = x if self.obj == 'x' else adj

        if self.guidance_args is not None and \
                timestep > 0:
            var, var_mean = self.guidance(x, adj, flags, t, is_adj=self.obj == 'adj')
        else:
            f, G = self.rsde.discretize(x, adj, flags, t, is_adj=self.obj == 'adj')
            z = gen_noise(var, flags, sym=self.obj == 'adj')
            var_mean = var - f
            var = var_mean + G[:, None, None] * z
        return var, var_mean
    

class NoneCorrector(Corrector):
    """An empty corrector that does nothing."""

    def __init__(self, obj, sde, score_fn, snr, scale_eps, n_steps):
        self.obj = obj
        pass

    def update_fn(self, x, adj, flags, t):
        if self.obj == 'x':
            return x, x
        elif self.obj == 'adj':
            return adj, adj
        else:
            raise NotImplementedError(f"obj {self.obj} not yet supported.")


class LangevinCorrector(Corrector):
    def __init__(self, obj, sde, score_fn, snr, scale_eps, n_steps):
        super().__init__(sde, score_fn, snr, scale_eps, n_steps)
        self.obj = obj

    def update_fn(self, x, adj, flags, t):
        sde = self.sde
        score_fn = self.score_fn
        n_steps = self.n_steps
        target_snr = self.snr
        seps = self.scale_eps

        if isinstance(sde, VPSDE) or isinstance(sde, subVPSDE):
            timestep = (t * (sde.N - 1) / sde.T).long()
            alpha = sde.alphas.to(t.device)[timestep]
        else:
            alpha = torch.ones_like(t)

        if self.obj == 'x':
            for i in range(n_steps):
                grad = score_fn(x, adj, flags, t)
                noise = gen_noise(x, flags, sym=False)
                grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
                noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
                step_size = (target_snr * noise_norm / grad_norm) ** 2 * 2 * alpha
                x_mean = x + step_size[:, None, None] * grad
                x = x_mean + torch.sqrt(step_size * 2)[:, None, None] * noise * seps
            return x, x_mean

        elif self.obj == 'adj':
            for i in range(n_steps):
                grad = score_fn(x, adj, flags, t)
                noise = gen_noise(adj, flags)
                grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
                noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
                step_size = (target_snr * noise_norm / grad_norm) ** 2 * 2 * alpha
                adj_mean = adj + step_size[:, None, None] * grad
                adj = adj_mean + torch.sqrt(step_size * 2)[:, None, None] * noise * seps
            return adj, adj_mean

        else:
            raise NotImplementedError(f"obj {self.obj} not yet supported")


# -------- PC sampler --------
def get_pc_sampler(sde_x, sde_adj, shape_x, shape_adj, predictor='Euler', corrector='None',
                                     guidance_args_adj=None, guidance_args_x=None,
                                     constraint_config_prodigy=None, method_config_prodigy=None,
                                     snr=0.1, scale_eps=1.0, n_steps=1, 
                                     probability_flow=False, continuous=False,
                                     denoise=True, eps=1e-3, device='cuda'):

    def pc_sampler(model_x, model_adj, init_flags):

        score_fn_x = get_score_fn(sde_x, model_x, train=False, continuous=continuous)
        score_fn_adj = get_score_fn(sde_adj, model_adj, train=False, continuous=continuous)

        predictor_fn = ReverseDiffusionPredictor if predictor=='Reverse' else EulerMaruyamaPredictor 
        corrector_fn = LangevinCorrector if corrector=='Langevin' else NoneCorrector

        predictor_obj_x = predictor_fn('x', sde_x, score_fn_x, probability_flow, guidance_args=guidance_args_x)
        corrector_obj_x = corrector_fn('x', sde_x, score_fn_x, snr, scale_eps, n_steps)

        predictor_obj_adj = predictor_fn('adj', sde_adj, score_fn_adj, probability_flow, guidance_args=guidance_args_adj)
        corrector_obj_adj = corrector_fn('adj', sde_adj, score_fn_adj, snr, scale_eps, n_steps)

        with torch.no_grad():
            # -------- Initial sample --------
            x = sde_x.prior_sampling(shape_x).to(device) 
            adj = sde_adj.prior_sampling_sym(shape_adj).to(device) 
            flags = init_flags
            x = mask_x(x, flags)
            adj = mask_adjs(adj, flags)
            diff_steps = sde_adj.N
            timesteps = torch.linspace(sde_adj.T, eps, diff_steps, device=device)

            # -------- Reverse diffusion process --------
            for i in trange(0, (diff_steps), desc = '[Sampling]', position = 1, leave=False):
                t = timesteps[i]
                vec_t = torch.ones(shape_adj[0], device=t.device) * t

                _x = x
                x, x_mean = corrector_obj_x.update_fn(x, adj, flags, vec_t)
                adj, adj_mean = corrector_obj_adj.update_fn(_x, adj, flags, vec_t)

                _x = x
                x, x_mean = predictor_obj_x.update_fn(x, adj, flags, vec_t)
                adj, adj_mean = predictor_obj_adj.update_fn(_x, adj, flags, vec_t)

                if constraint_config_prodigy is not None and method_config_prodigy is not None:
                    x, adj = drifted_project(x, adj, i=i, diff_steps=diff_steps, constraint_config=constraint_config_prodigy, method_config=method_config_prodigy)
                    # x_mean, adj_mean = drifted_project(x_mean, adj_mean, i=i, diff_steps=diff_steps, constraint_config=constraint_config_prodigy, method_config=method_config_prodigy)
            print(' ')
            return (x_mean if denoise else x), (adj_mean if denoise else adj), diff_steps * (n_steps + 1)
    return pc_sampler


# -------- S4 solver --------
def S4_solver(sde_x, sde_adj, shape_x, shape_adj, predictor='None', corrector='None',
                                                guidance_args_adj=None, guidance_args_x=None,
                                                constraint_config_prodigy=None, method_config_prodigy=None,
                                                snr=0.1, scale_eps=1.0, n_steps=1, 
                                                probability_flow=False, continuous=False,
                                                denoise=True, eps=1e-3, device='cuda'):

    def s4_solver(model_x, model_adj, init_flags):

        score_fn_x = get_score_fn(sde_x, model_x, train=False, continuous=continuous)
        score_fn_adj = get_score_fn(sde_adj, model_adj, train=False, continuous=continuous)

        with torch.no_grad():
            # -------- Initial sample --------
            x = sde_x.prior_sampling(shape_x).to(device) 
            adj = sde_adj.prior_sampling_sym(shape_adj).to(device) 
            flags = init_flags
            x = mask_x(x, flags)
            adj = mask_adjs(adj, flags)
            diff_steps = sde_adj.N
            timesteps = torch.linspace(sde_adj.T, eps, diff_steps, device=device)
            dt = -1. / diff_steps

            # -------- Rverse diffusion process --------
            for i in trange(0, (diff_steps), desc = '[Sampling]', position = 1, leave=False):
                t = timesteps[i]
                vec_t = torch.ones(shape_adj[0], device=t.device) * t
                vec_dt = torch.ones(shape_adj[0], device=t.device) * (dt/2) 

                # -------- Score computation --------
                score_x = score_fn_x(x, adj, flags, vec_t)
                score_adj = score_fn_adj(x, adj, flags, vec_t)

                Sdrift_x = -sde_x.sde(x, vec_t)[1][:, None, None] ** 2 * score_x
                Sdrift_adj = -sde_adj.sde(adj, vec_t)[1][:, None, None] ** 2 * score_adj

                # -------- Correction step --------
                timestep = (vec_t * (sde_x.N - 1) / sde_x.T).long()

                noise = gen_noise(x, flags, sym=False)
                grad_norm = torch.norm(score_x.reshape(score_x.shape[0], -1), dim=-1).mean()
                noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
                if isinstance(sde_x, VPSDE):
                    alpha = sde_x.alphas.to(vec_t.device)[timestep]
                else:
                    alpha = torch.ones_like(vec_t)
            
                step_size = (snr * noise_norm / grad_norm) ** 2 * 2 * alpha
                x_mean = x + step_size[:, None, None] * score_x
                x = x_mean + torch.sqrt(step_size * 2)[:, None, None] * noise * scale_eps

                noise = gen_noise(adj, flags)
                grad_norm = torch.norm(score_adj.reshape(score_adj.shape[0], -1), dim=-1).mean()
                noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
                if isinstance(sde_adj, VPSDE):
                    alpha = sde_adj.alphas.to(vec_t.device)[timestep] # VP
                else:
                    alpha = torch.ones_like(vec_t) # VE
                step_size = (snr * noise_norm / grad_norm) ** 2 * 2 * alpha
                adj_mean = adj + step_size[:, None, None] * score_adj
                adj = adj_mean + torch.sqrt(step_size * 2)[:, None, None] * noise * scale_eps

                # -------- Prediction step --------
                x_mean = x
                adj_mean = adj
                mu_x, sigma_x = sde_x.transition(x, vec_t, vec_dt)
                mu_adj, sigma_adj = sde_adj.transition(adj, vec_t, vec_dt)

                if guidance_args_x is not None and \
                        timestep > 0:
                    x = apply_guidance(x, adj, flags, False, vec_t, vec_dt,
                                               sde_x, score_fn_x, mask_x,
                                               guidance_args_x,
                                               mu_x, sigma_x)
                else:
                    x = mu_x + sigma_x[:, None, None] * gen_noise(x, flags, sym=False)
                    x = x + Sdrift_x * dt
                mu_x, sigma_x = sde_x.transition(x, vec_t + vec_dt, vec_dt)
                x_mean = mu_x

                if guidance_args_adj is not None and \
                        timestep > 0:
                    adj = apply_guidance(x, adj, flags, True, vec_t, vec_dt,
                                                    sde_adj, score_fn_adj, mask_adjs,
                                                    guidance_args_adj,
                                                    mu_adj, sigma_adj)
                else:
                    adj = mu_adj + sigma_adj[:, None, None] * gen_noise(adj, flags)
                    adj = adj + Sdrift_adj * dt
                mu_adj, sigma_adj = sde_adj.transition(adj, vec_t + vec_dt, vec_dt)
                adj_mean = mu_adj

                if constraint_config_prodigy is not None and method_config_prodigy is not None:
                    x, adj = drifted_project(x, adj, i=i, diff_steps=diff_steps, constraint_config=constraint_config_prodigy, method_config=method_config_prodigy)
                    x_mean, adj_mean = drifted_project(x_mean, adj_mean, i=i, diff_steps=diff_steps, constraint_config=constraint_config_prodigy, method_config=method_config_prodigy)

            print(' ')
            return (x_mean if denoise else x), (adj_mean if denoise else adj), 0
    return s4_solver

def apply_guidance(x, adj, flags, is_adj, vec_t, vec_dt,
                   sde, score_fn, mask_fn,
                   guidance_args,
                   mu, sigma):

    timestep = vec_t[0].item()
    obj = adj if is_adj else x
    # Guidance
    if guidance_args is not None and guidance_args.method == 'greedy':
        n_traj = guidance_args.n_traj
        loss_fn = getattr(losses_guidance, guidance_args.loss_fn)
        loss_kwargs = guidance_args.get('loss_kwargs', {})
        if "flags" in loss_kwargs:
            loss_kwargs = getattr(losses_guidance, loss_kwargs['flags'])(flags)

        losses = torch.zeros(n_traj, x.shape[0])
        obj_hats = []
        for i in range(n_traj):
            obj_hat = mu + sigma[:, None, None] * gen_noise(obj, flags, sym=is_adj)
            obj_hats.append(obj_hat)
            
            score_i = score_fn(x, obj_hat, flags, vec_t+vec_dt) if is_adj else score_fn(obj_hat, adj, flags, vec_t+vec_dt)

            obj0hat = sde.obj0estimation(obj_hat, score_i, timestep)
            obj0hat_masked = mask_fn(obj0hat, flags)
            
            losses[i,:] = loss_fn(obj0hat_masked, **loss_kwargs)

        losses_expanded = torch.argmin(losses, dim=0).view(1, x.shape[0], 1, 1).expand(1, x.shape[0], x.shape[1], x.shape[2]).to(x.device)

        obj = torch.gather(torch.stack(obj_hats, dim=0), 0, losses_expanded).squeeze(0)

    elif guidance_args.method == 'zero':
        n_traj = guidance_args.n_traj

        score_no_noise = score_fn(x, mu.clone(), flags, vec_t) if is_adj else score_fn(mu.clone(), adj, flags, vec_t)
        obj0hat_no_noise = sde.obj0estimation(mu.clone(), score_no_noise, timestep)
        obj0hat_masked_no_noise = mask_fn(obj0hat_no_noise, flags)
        no_noise_loss = loss_fn(obj0hat_masked_no_noise, **loss_kwargs)

        losses = torch.zeros(n_traj, obj.shape[0])
        noise_directions = []
        for i in range(n_traj):
            z = gen_noise(obj, flags, sym=is_adj)
            obj_hat = mu + sigma[:, None, None] * z
            noise_directions.append(z)
            
            score_i = score_fn(x, obj_hat, flags, vec_t+vec_dt) if is_adj else score_fn(obj_hat, adj, flags, vec_t+vec_dt)

            obj0hat = sde.obj0estimation(obj_hat, score_i, timestep)
            obj0hat_masked = mask_fn(obj0hat, flags)
            
            losses[i,:] = loss_fn(x, obj0hat_masked, **loss_kwargs) if is_adj else loss_fn(obj0hat_masked, adj, **loss_kwargs)

        weights = (losses - no_noise_loss[None,:]) / sigma[None, :]
        directions = torch.stack(noise_directions, dim=0)
        obj = mu - guidance_args.lr_zero * (weights[:,:,None,None] * directions).mean(dim=0)

    elif guidance_args is not None and guidance_args.method == 'loss':
        loss_fn = getattr(losses_guidance, guidance_args.loss_fn)
        loss_kwargs = guidance_args.get('loss_kwargs', {})
        with torch.enable_grad():

            obj.requires_grad = True

            score = score_fn(x, obj_hat, flags, vec_t+vec_dt) if is_adj else score_fn(obj_hat, adj, flags, vec_t+vec_dt)

            obj0hat = sde.obj0estimation(obj, score, timestep)
            obj0hat_masked = mask_fn(obj0hat, flags)
            
            loss = loss_fn(obj0hat_masked, **loss_kwargs).mean()

            loss.backward()

            obj_grad = obj.grad.detach().clone()
            obj.grad = None

        obj = mu + sigma[:, None, None] * gen_noise(obj, flags, sym=False)

        if guidance_args.lr_guidance_method == 'adaptive':
            obj -= guidance_args.lr_guidance / torch.abs(loss) * obj_grad
        else:
            obj -= guidance_args.lr_guidance * obj_grad

    else:
        raise NotImplementedError
    return obj