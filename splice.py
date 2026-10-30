import math

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from base import Synthesizer

CHEB_NORM = math.sqrt(2. / math.pi)

def normal_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def gauss_delta(sigma, eps, sens):
    if sigma <= 0:
        return 1.0
    a = sens / (2.0 * sigma)
    b = eps * sigma / sens
    return normal_cdf(a - b) - math.exp(eps) * normal_cdf(-a - b)

def analytic_gauss_sigma(eps, delta, sens=1.0):
    low = 1e-10
    high = 1.0
    while gauss_delta(high, eps, sens) > delta:
        high = high * 2.0
    for _ in range(200):
        mid = 0.5 * (low + high)
        if gauss_delta(mid, eps, sens) > delta:
            low = mid
        else:
            high = mid
    return high

def sample_dirs(L, d, rng):
    Z = rng.standard_normal(size=(L, d))
    norms = np.linalg.norm(Z, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return Z / norms

def chebyshev_T(x, max_deg):
    x = np.clip(x, -1.0, 1.0)
    angle = np.arccos(x)
    T = np.zeros((max_deg + 1, len(x)))
    for j in range(max_deg + 1):
        T[j] = np.cos(j * angle)
    return T

class TabularData:
    DISCRETE_THRESHOLD = 20
    MAX_LEVELS = 64

    def __init__(self, df, cat_columns=None):
        self.df = df
        self.columns = df.columns.tolist()
        self.n, self.d = df.shape
        self.dtypes = {col: df[col].dtype for col in self.columns}

        # split the columns into categorical and continuous
        self.cat_cols = []
        self.con_cols = []
        for col in self.columns:
            if cat_columns is not None:
                declared = col in cat_columns
            else:
                declared = not pd.api.types.is_numeric_dtype(df[col])

            if declared or df[col].nunique() < self.DISCRETE_THRESHOLD:
                self.cat_cols.append(col)
            else:
                self.con_cols.append(col)

        # continuous columns: remember the min and max
        self.bounds = {}
        for col in self.con_cols:
            self.bounds[col] = (df[col].min(), df[col].max())

        # categorical columns: sorted levels, each given a grid point in [-1, 1]
        self.cat_levels = {}
        self.cat_grid = {}
        for col in self.cat_cols:
            levels = np.sort(pd.unique(df[col]))

            # too many levels: keep only the most frequent ones (still sorted)
            if len(levels) > self.MAX_LEVELS:
                most_common = set(df[col].value_counts().index[: self.MAX_LEVELS])
                levels = np.array([v for v in levels if v in most_common])

            if len(levels) > 1:
                grid = np.linspace(-1.0, 1.0, len(levels))
            else:
                grid = np.array([0.0])

            self.cat_levels[col] = levels
            self.cat_grid[col] = grid

    def encode(self):
        # every column becomes numbers in [-1, 1]
        X = np.zeros((self.n, self.d), dtype=float)
        for j, col in enumerate(self.columns):
            if col in self.con_cols:
                X[:, j] = self.encode_continuous(col)
            else:
                X[:, j] = self.encode_categorical(col)
        return X

    def encode_continuous(self, col):
        # rescale [lo, hi] to [-1, 1]
        lo, hi = self.bounds[col]
        width = hi - lo
        if width == 0:
            width = 1.0
        values = self.df[col].to_numpy(dtype=float)
        return 2.0 * (values - lo) / width - 1.0

    def encode_categorical(self, col):
        # replace each value by the grid point of its level
        levels = self.cat_levels[col]
        grid = self.cat_grid[col]
        level_to_grid = dict(zip(levels, grid))
        levels_are_numeric = np.issubdtype(levels.dtype, np.number)

        encoded = np.zeros(self.n)
        for row, value in enumerate(self.df[col].to_numpy()):
            if value in level_to_grid:
                encoded[row] = level_to_grid[value]
            elif levels_are_numeric:
                # this value's level was dropped: use the nearest kept level
                nearest = np.argmin(np.abs(levels - value))
                encoded[row] = grid[nearest]
            else:
                # dropped non-numeric level: fall back to the first level
                encoded[row] = grid[0]
        return encoded

    def decode_continuous(self, col, values):
        # undo the rescaling: [-1, 1] back to [lo, hi]
        lo, hi = self.bounds[col]
        out = (values + 1.0) / 2.0 * (hi - lo) + lo
        out = np.clip(out, lo, hi)

        dtype = self.dtypes[col]
        if np.issubdtype(dtype, np.integer):
            out = np.round(out)
        return out.astype(dtype)


#### the mechanism

class SPLICE(Synthesizer):
    def __init__(self, eps=1.0, delta=1e-9, K=5, q=2,
                 L=None, max_deg=None, cat_columns=None, iters=2000,
                 lr=0.02, n_restarts=1, ema_decay=0.995, huber_delta=0.5,
                 ridge=1e-4, ent_reg=1e-3, grad_clip=5.0, device=None,
                 seed=None):
        super().__init__(eps=eps, delta=delta, seed=seed)
        self.K = K
        self.q = q
        self.L = L
        self.max_deg = max_deg
        self.cat_columns = cat_columns
        self.iters = iters
        self.lr = lr
        self.n_restarts = n_restarts
        self.ema_decay = ema_decay
        self.huber_delta = huber_delta
        self.ridge = ridge
        # keeps mixture weights away from zero, only matters with > 1 component
        self.ridge_pi = 1.0 if self.K > 1 else 0.0
        self.ent_reg = ent_reg
        self.grad_clip = grad_clip
        self.device = device
        if self.device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device(self.device)
        self.dtype = torch.float64
        self.learned_params = None
        self.data = None
        self._fitted = False

    def sensitivity_calc(self):
        constant_H = float(np.sum(1.0 / np.arange(1, self.max_deg + 1)))
        sens = math.sqrt((8. * self.L * constant_H / (np.pi * self.data.n ** 2)))
        return sens

    def fit(self, df):
        self.data = TabularData(df, cat_columns=self.cat_columns)
        n = self.data.n
        d = self.data.d

        eps_n = self.eps * n
        if self.max_deg is None:
            self.max_deg = max(8, min(30, int(np.ceil(eps_n**0.5))))

        if self.L is None:
            self.L = max(100, min(300, int(np.ceil(2*eps_n**0.5))))

        sens = self.sensitivity_calc()
        sigma = analytic_gauss_sigma(self.eps, self.delta, sens)
        rng = np.random.default_rng(self.seed)
        thetas = sample_dirs(self.L, d, rng)

        X = self.data.encode()

        # we will project the data onto the theta dirs
        proj = thetas @ X.T
        # rescale
        proj = proj / np.sqrt(d)
        # noise vec, noise the higher deg more
        noise = sigma * np.sqrt(np.arange(1, self.max_deg + 1))
        noisy_moments = np.zeros((self.L, self.max_deg))
        for l in range(self.L):
            T = chebyshev_T(proj[l], self.max_deg)
            norm_T_moms = CHEB_NORM * np.mean(T[1:], axis=1)
            noisy_moments[l] = norm_T_moms + noise * rng.standard_normal(self.max_deg)

        # now that we have noisy sliced moments, this func
        # contains everything that is post-processing
        # to recover a matching dist
        self.solve(noisy_moments, thetas)
        self._fitted = True
        return self

    def outer_prod_and_matmul(self, tab, a, b):
        out = a.unsqueeze(-1) * b.unsqueeze(-2)
        out = out.reshape(out.shape[0], out.shape[1], -1)
        tab_2_d = tab.reshape(tab.shape[0], -1)
        return out @ tab_2_d.T

    def solve(self, noisy_moments, thetas):
        col_index = {
            col: idx for idx, col in enumerate(self.data.columns)
        }
        cat_index = [col_index[col] for col in self.data.cat_cols]
        cont_index = [col_index[col] for col in self.data.con_cols]

        n_cat = len(cat_index)
        n_cont = len(cont_index)

        # we need to construct the right convolution
        B = []
        for k in range(self.max_deg + 1):
            B_k = [math.comb(k, p) for p in range(self.max_deg + 1)]
            B.append(torch.tensor(B_k, device=self.device))
        B = torch.stack(B)

        indices = torch.arange(self.max_deg + 1, device=self.device)
        k, p, q = torch.meshgrid(indices, indices, indices, indexing="ij")

        T_vals_binom = torch.where(k == p + q, B[k, p], 0).to(self.dtype)

        # also need chebyshev poly basis
        # (built in float64 so the big coefficients stay exact, cast at the end)
        C = torch.zeros((self.max_deg + 1, self.max_deg + 1), dtype=torch.float64)
        C[0, 0] = 1.0
        C[1, 1] = 1.0
        for j in range(2, self.max_deg + 1):
            for t in range(j + 1):
                if t >= 1:
                    shifted = 2. * C[j - 1, t - 1]
                else:
                    shifted = 0.
                C[j, t] = shifted - C[j - 2, t]
        C = C.to(device=self.device, dtype=self.dtype)

        # closed form gauss raw moments (same float64-then-cast trick)
        G = torch.zeros((self.max_deg + 1, self.max_deg + 1, self.max_deg // 2 + 1), dtype=torch.float64)
        for t in range(self.max_deg + 1):
            for var_pow in range(t // 2 + 1):
                mean_pow = t - 2 * var_pow
                coef = math.factorial(t) // (math.factorial(mean_pow) * 2**var_pow * math.factorial(var_pow))
                G[t, mean_pow, var_pow] = float(coef)
        G = G.to(device=self.device, dtype=self.dtype)

        # ok! now we are ready for the optimization
        tensor_thetas = torch.tensor(thetas, device=self.device, dtype=self.dtype)
        target = torch.tensor(noisy_moments, dtype=self.dtype, device=self.device)

        weights = 1.0 / torch.arange(1, self.max_deg + 1, device=self.device, dtype=self.dtype)

        cat_thetas = tensor_thetas[:, cat_index] / math.sqrt(self.data.d)
        cat_theta_powers = torch.stack([cat_thetas**t for t in range(self.max_deg + 1)], dim=2)
        grid = []
        for col in self.data.cat_cols:
            g = torch.tensor(self.data.cat_grid[col], device=self.device, dtype=self.dtype)
            grid.append(torch.stack([g**t for t in range(self.max_deg + 1)], dim=1))

        con_thetas = tensor_thetas[:, cont_index]

        ### PRED THE MOMENTS
        # needed twice (every training step, and to score the ema params),
        # so it lives in a little function that can see all the tables above
        def pred_moments(mix_weights, cat_weights, mu, U_cov, ind_var):
            rho_categorical = torch.zeros(self.L, self.K, self.max_deg + 1, dtype=self.dtype, device=self.device)
            rho_categorical[:, :, 0] = 1.0

            for a in range(n_cat):
                component_moment = cat_weights[a] @ grid[a]
                col_moments = cat_theta_powers[:, a, :].unsqueeze(1) * component_moment[None]
                rho_categorical = self.outer_prod_and_matmul(T_vals_binom, rho_categorical, col_moments)

            if n_cont == 0:
                rho_continuous = torch.zeros(self.L, self.K, self.max_deg + 1, dtype=self.dtype, device=self.device)
                rho_continuous[:, :, 0] = 1.0
            else:
                mean_proj_cont = (con_thetas @ mu.T) / math.sqrt(self.data.d)
                cov_U_proj = torch.matmul(con_thetas, U_cov).transpose(0, 1)
                var_1d_gauss = ((cov_U_proj ** 2).sum(dim=-1) + (con_thetas ** 2) @ ind_var.T) / self.data.d

                powers_mean_proj_cont = torch.stack([mean_proj_cont ** i for i in range(self.max_deg + 1)], dim=-1)
                var_1d_pows = []
                for i in range(G.shape[2]):
                    var_1d_pows.append(var_1d_gauss ** i)
                var_1d_pows = torch.stack(var_1d_pows, dim=-1)
                rho_continuous = self.outer_prod_and_matmul(G, powers_mean_proj_cont, var_1d_pows)

            rho_full = self.outer_prod_and_matmul(T_vals_binom, rho_categorical, rho_continuous)
            chebyshev_moments = (rho_full @ C.T)[:, :, 1:] * CHEB_NORM
            chebyshev_moments = chebyshev_moments * mix_weights[None, :, None]
            return chebyshev_moments.sum(dim=1)
        ###

        best_loss = float('inf')
        best = None
        for rest in range(self.n_restarts):
            # own random stream per restart, never the one that drew the privacy noise
            if self.seed is None:
                rng = np.random.default_rng()
            else:
                rng = np.random.default_rng([self.seed, 2 + rest])

            mixture_logits = torch.zeros(self.K, device=self.device, dtype=self.dtype, requires_grad=True)
            cat_logits = []
            for col in self.data.cat_cols:
                # we init with a little noise to jumpstart the optimization
                cat_logits.append(torch.tensor(
                    0.1 * rng.standard_normal(size=(self.K, len(self.data.cat_levels[col]))),
                    device=self.device,
                    dtype=self.dtype,
                    requires_grad=True
                ))

            # more warmup
            if n_cont == 0:
                mu_zero = np.zeros((self.K, 0), np.float64)
                U_cov_zero = np.zeros((self.K, 0, self.q), np.float64)
                ind_var_zero = np.zeros((self.K, 0), np.float64)
            else:
                # we can do least squares init with the moms as a nice
                # init : ) (thetas * mean = right_hand_side)
                right_hand_side = noisy_moments[:, 0] * math.sqrt(self.data.d) / CHEB_NORM
                lsq_sol = np.linalg.lstsq(thetas, right_hand_side, rcond=None)[0]
                lsq_cont = lsq_sol[cont_index]
                mu_zero = lsq_cont[None, :] + 0.3 * rng.standard_normal(size=(self.K, n_cont))
                U_cov_zero = 0.05 * rng.standard_normal((self.K, n_cont, self.q))
                ind_var_zero = np.full((self.K, n_cont), math.log(0.1))

            # conv to tensors (note ind_var_zero holds LOG variances)
            mu_zero = torch.tensor(mu_zero, device=self.device, dtype=self.dtype, requires_grad=True)
            U_cov_zero = torch.tensor(U_cov_zero, device=self.device, dtype=self.dtype, requires_grad=True)
            ind_var_zero = torch.tensor(ind_var_zero, device=self.device, dtype=self.dtype, requires_grad=True)

            # params
            model_params = [mixture_logits] + cat_logits + [mu_zero, U_cov_zero, ind_var_zero]

            optimizer = torch.optim.Adam(model_params, lr=self.lr)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.iters)
            # we do the ema trick for stability
            ema = [param.detach().clone() for param in model_params]

            for i in range(self.iters):
                optimizer.zero_grad()
                mix_weights = F.softmax(mixture_logits, dim=0)
                cat_weights = [F.softmax(logs, dim=1) for logs in cat_logits]

                pred = pred_moments(mix_weights, cat_weights, mu_zero, U_cov_zero, torch.exp(ind_var_zero))

                # weighted residuals, with huber so the noisiest
                # high degree moments can't dominate
                residual = weights * (pred - target)
                loss = F.smooth_l1_loss(residual, torch.zeros_like(residual),
                                        beta=self.huber_delta, reduction='sum')

                # light regularization
                if n_cont > 0:
                    ridge_penalty = (mu_zero ** 2).sum() + (U_cov_zero ** 2).sum() + (ind_var_zero ** 2).sum()
                    loss = loss + self.ridge * ridge_penalty
                if self.ridge_pi > 0:
                    loss = loss - self.ridge_pi * torch.log(mix_weights + 1e-30).sum()
                if self.ent_reg > 0 and n_cat > 0:
                    # reward spread out categorical probs
                    entropy = 0.0
                    for probs in cat_weights:
                        entropy = entropy - (probs * torch.log(probs + 1e-10)).sum()
                    loss = loss - self.ent_reg * entropy

                loss.backward()

                # oops nan grads
                has_nan_grad = False
                for param in model_params:
                    if param.grad is not None and torch.isnan(param.grad).any():
                        has_nan_grad = True
                if has_nan_grad:
                    optimizer.zero_grad()
                    continue

                torch.nn.utils.clip_grad_norm_(model_params, self.grad_clip)
                optimizer.step()
                scheduler.step()

                # ema = decay * ema + (1 - decay) * param
                for ema_param, param in zip(ema, model_params):
                    ema_param.mul_(self.ema_decay).add_(param.detach(), alpha=1 - self.ema_decay)

            # score the ema params
            with torch.no_grad():
                ema_mix_weights = F.softmax(ema[0], dim=0)
                ema_cat_weights = [F.softmax(logs, dim=1) for logs in ema[1:1 + n_cat]]
                ema_mu, ema_U_cov, ema_log_var = ema[1 + n_cat:]
                ema_ind_var = torch.exp(ema_log_var)

                pred = pred_moments(ema_mix_weights, ema_cat_weights, ema_mu, ema_U_cov, ema_ind_var)
                final_loss = float(((weights * (pred - target)) ** 2).sum())

            # keep the best restart
            if final_loss < best_loss:
                best_loss = final_loss
                best = {
                    "mix_weights": ema_mix_weights.cpu().numpy(),
                    "cat_weights": [w.cpu().numpy() for w in ema_cat_weights],
                    "mu": ema_mu.cpu().numpy(),
                    "U_cov": ema_U_cov.cpu().numpy(),
                    "ind_var": ema_ind_var.cpu().numpy(),
                }

        self.learned_params = best

    def sample(self, n):
        mix_weights = self.learned_params["mix_weights"].astype(np.float64)
        cat_weights = self.learned_params["cat_weights"]
        mu = self.learned_params["mu"]
        U_cov = self.learned_params["U_cov"]
        ind_var = self.learned_params["ind_var"]

        rng = np.random.default_rng()
        
        # which mixture component does each row come from
        component = rng.choice(self.K, size=n, p=mix_weights / mix_weights.sum())

        out = {}

        # draw a level from that cat component's probs
        for a, col in enumerate(self.data.cat_cols):
            levels = self.data.cat_levels[col]
            values = np.empty(n, dtype=object)
            for h in range(self.K):
                rows = component == h
                count = int(rows.sum())
                if count == 0:
                    continue
                probs = cat_weights[a][h].astype(np.float64)
                probs = probs / probs.sum()
                values[rows] = levels[rng.choice(len(levels), size=count, p=probs)]
            out[col] = pd.Series(values).astype(self.data.dtypes[col])

        # mean + low rank part + diagonal part for cont columns
        n_cont = len(self.data.con_cols)
        if n_cont > 0:
            X_cont = np.zeros((n, n_cont))
            for h in range(self.K):
                rows = component == h
                count = int(rows.sum())
                if count == 0:
                    continue
                low_rank = rng.standard_normal((count, self.q)) @ U_cov[h].T
                diag_noise = rng.standard_normal((count, n_cont)) * np.sqrt(ind_var[h])[None, :]
                X_cont[rows] = mu[h][None, :] + low_rank + diag_noise

            # need to clip cuz gaussias tail
            X_cont = np.clip(X_cont, -1.0, 1.0)
            for i, col in enumerate(self.data.con_cols):
                out[col] = self.data.decode_continuous(col, X_cont[:, i])

        return pd.DataFrame(out, columns=self.data.columns)
