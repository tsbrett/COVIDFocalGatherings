from collections import deque

import numpy as np
import torch
from torch.distributions.binomial import Binomial
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
from torch.distributions.negative_binomial import NegativeBinomial

class GeneralSpatialTransmissionModel:

    def __init__(self,
                 seed_day,
                 seed_counties,
                 seed_size,
                 population_size_tensor,
                 local_tensor,
                 M_out_tensor,
                 unscaled_M_back_tensor,
                 month_start_tensor,
                 location_names,
                 t0,
                 tf,
                 t_eval,
                 numpy_rng,
                 reduction_start_day,
                 n_particles=1,
                 timestep=1,
                 death_delay_weeks=0,
                 mean_floor=1e-5,
                 rw_sd=None,
                 rw_alpha=1.,):

        self.numpy_rng = numpy_rng

        self.timestep = timestep
        self.t_eval = t_eval
        self.t0 = t0
        self.tf = tf

        self.compartment_names = ["S", "E", "I", "R", "C"]
        self.n_compartments = len(self.compartment_names)

        self.seed_day = seed_day
        self.seed_counties = seed_counties
        self.seed_size = seed_size

        self.population_size_tensor = population_size_tensor
        self.local_tensor = local_tensor
        # the day fixed_reduction_coef's reduction starts
        self.reduction_start_day = reduction_start_day
        self.M_out_tensor = M_out_tensor
        self.unscaled_M_back_tensor = unscaled_M_back_tensor
        self.month_start_tensor = month_start_tensor
        self.location_names = location_names

        self.n_c = self.population_size_tensor.shape[0]
        self.n_particles = n_particles
        self.tensor_size = (self.n_particles, self.n_compartments,
                            self.n_c, self.n_c)

        # weeks from removals to reported deaths, and the floor on each
        # county's expected deaths
        self.death_delay_weeks = death_delay_weeks
        self.mean_floor = mean_floor
        # iterated filtering: each parameter's random-walk sd, and its cooling
        self.rw_sd = None if rw_sd is None else torch.tensor(rw_sd)
        self.rw_alpha = rw_alpha
        # each particle's weekly removals by residence, the last
        # death_delay_weeks + 1 weeks, oldest first
        self.removed_history = deque(maxlen=death_delay_weeks + 1)


    def initialise_model(self):
        """
        Initial fully susceptible population with all individuals in their in
         home county. Model must be allowed to relax to DFE before introduction
          of pathogen.
        :param N_vector: Population size for each county
        :return: Initial conditions
        """

        y0 = torch.zeros(size=(self.n_compartments, self.n_c, self.n_c))
        y0[0, :, :] = torch.diag(self.population_size_tensor)

        y0 = torch.unsqueeze(y0, 0)  # Add particles dimension
        y0 = torch.tile(y0, dims=(self.n_particles, 1, 1, 1))

        return y0

    def seed_outbreak(self, y):
        """
        Seed outbreak in population
        :param y: State tensor
        :return: Updated state tensor
        """
        for seed_county in self.seed_counties:
            # Seed equal number of E and I
            seed_tensor = self.seed_size * torch.ones(size=(self.n_particles,),
                                                      dtype=torch.int64)
            y[:, 1, seed_county, seed_county] = seed_tensor
            y[:, 0, seed_county, seed_county] -= seed_tensor
        return y

    def reset_cumulative_variables(self, y, indicies=[4]):

        y[:, indicies] = 0
        return y

    def get_connectivity_tensors(self, m, par, t):
        """
        Make function depend on model parameters.
        :param m: index of month
        :param t: time, for the start of the fixed reduction
        :return:
        """

        mean_trip_duration = par[:, 7]
        local_coef = par[:, 8]
        fixed_reduction_coef = par[:, 9]
        travel_scale = par[:, 11]

        M_out = self.M_out_tensor[m]
        M_back = self.unscaled_M_back_tensor  # Assumed constant throughout

        M_out = torch.tile(M_out[None, :, :], dims=(self.n_particles, 1, 1))
        # each particle's travel rates scaled by its travel_scale
        M_out = M_out * travel_scale[:, None, None]

        M_back = torch.tile(M_back[None, :, :], dims=(self.n_particles, 1, 1))
        M_back *= 1. / mean_trip_duration[:, None, None]  # Multiply by mean
        # trip duration

        local_exponent = (local_coef[:, None] * self.local_tensor[m][None, :]
                          - fixed_reduction_coef[:, None]
                          * float(t >= self.reduction_start_day))
        local_contact = torch.exp(local_exponent)

        return M_out, M_back,  local_contact

    def net_movement_out(self, comp_mat, M_out):

        # Total count of outflow from origin i for each county
        rate_mat_sum = M_out.sum(dim=-1)  # Total outflow rate for each county
        p = 1 - torch.exp(-self.timestep * rate_mat_sum)  # Convert to a prob.

        # Generalise torch.diag for >2-dim tensors
        comp_mat_diag = torch.diagonal(comp_mat, dim1=-2, dim2=-1)
        # comp_mat_diag = torch.diag(comp_mat)
        n_event = Binomial(total_count=comp_mat_diag, probs=p).sample()

        # Probability of traveling from i to j given travel:
        M_out_norm = M_out / rate_mat_sum[..., None]

        #print(n_event)
        #print(M_out_norm)

        if self.n_particles == 1:
            # Optimised sampling, only works for n_particles == 1
            n_event_squeeze = n_event[0]
            M_out_norm_squeeze = M_out_norm[0]
            pos = np.nonzero(n_event > 0)  # Only update these positions
            out = np.zeros(M_out_norm_squeeze.shape)
            out[pos] = self.numpy_rng.multinomial(n=n_event_squeeze[pos],
                                                  pvals=M_out_norm_squeeze[pos])
            out = out[None, ...]
            # out -= np.diag(n_event)  # Total departures
            # out = torch.Tensor(out)

        else:
            out = self.numpy_rng.multinomial(n=n_event,
                                                  pvals=M_out_norm)

        out = torch.Tensor(out)
        n_event_diag = torch.diag_embed(torch.Tensor(n_event))
        out -= n_event_diag

        return out

    def net_movement_back(self, comp_mat, M_back):

        #  Binomial sampling for return movement
        p = 1 - torch.exp(-self.timestep * M_back)

        back = Binomial(total_count=comp_mat, probs=p).sample()

        total_back = torch.diag_embed(torch.sum(back, dim=-1))
        return total_back - back

    def movement_model(self, comp_mat, M_out, M_back):

        return (self.net_movement_out(comp_mat, M_out)
                + self.net_movement_back(comp_mat, M_back))

    def record_removed(self, y):
        """
        Add the week's removals by residence to the history the reported
        deaths are drawn from, the weeks before the first counted as 0.
        :param y: state tensor
        """
        removed = y[:, 4].sum(axis=-1)
        if not self.removed_history:
            self.removed_history.extend(
                [torch.zeros_like(removed)] * self.death_delay_weeks)
        self.removed_history.append(removed)

    def resample_history(self, index):
        """
        Keep the removals history with its particles after resampling.
        :param index: each new particle's old particle
        """
        self.removed_history = deque((h[index] for h in self.removed_history),
                                     maxlen=self.removed_history.maxlen)

    def get_removed(self, y):
        """
        The removals (I to R) by residence whose deaths are reported this
        week: this week's, or death_delay_weeks back (record_removed).
        :param y: state tensor
        :return: (particles, counties) tensor
        """
        if self.death_delay_weeks == 0:
            return y[:, 4].sum(axis=-1)
        return self.removed_history[0]

    def reporting_model(self, y, par):

        rep_prob = par[:, 4][:, None]
        dispersion = par[:, 10][:, None]
        mu = self.get_removed(y) * rep_prob + self.mean_floor

        return NegativeBinomial(total_count=1. / dispersion,
                                probs=1 - 1. / (dispersion * mu + 1.),)

    def reporting_model_prob(self, t, y, par, y_obs, give_log=True):

        log_obs_prob = self.reporting_model(y=y, par=par).log_prob(y_obs)
        log_likelihood = log_obs_prob.sum(axis=-1)  # Sum over counties

        return log_likelihood if give_log else torch.exp(log_likelihood)

    def importance_sampling(self, t, y_prev, par_prev, y_obs):

        # Importance weight
        weights = self.reporting_model_prob(t, y=y_prev, par=par_prev,
                                            y_obs=y_obs, give_log=False)

        conditional_log_likelihood = torch.log(weights.mean())

        resampled_index = Categorical(weights).sample((self.n_particles,))

        y = y_prev[resampled_index]
        par = par_prev[resampled_index]
        self.resample_history(resampled_index)

        return y, par, conditional_log_likelihood

    def parameter_random_walk(self, t, par, rw_sd):

        log_indices = [0, 1, 2, 3]
        logit_indices = [4]

        rw_indices = (rw_sd > 0).nonzero()

        par[:, log_indices] = torch.log(par[:, log_indices])
        par[:, logit_indices] = torch.logit(par[:, logit_indices])

        par[:, rw_indices] = Normal(loc=par[:, rw_indices],
                                    scale=rw_sd[rw_indices]).sample()

        par[:, log_indices] = torch.exp(par[:, log_indices])
        par[:, logit_indices] = torch.special.expit(par[:, logit_indices])

        return par

    def construct_parameter_tensor(self, pars):
        """
        Convert list of parameters to a (n_particle, n_parameters) tensor
        :param pars:
        :return:
        """
        return torch.tensor(pars)[None, :].repeat(self.n_particles, 1)

    def process_model(self, t, X, par, M_out, M_back, local_contact):
        """
        Take present state of system and return updated state
        :param t: time.
        :param X: state of the system at time t.
        :return: Updated state (torch tensor)
        """

        delta = self.timestep
        net_movement = lambda x: self.movement_model(x, M_out=M_out,
                                                     M_back=M_back)

        # Split state tensor into model compartments
        S, E, I, R, C = X.unbind(dim=1)

        # Unpack first 4 parameters of par tensor
        R0, tau_I, tau_E, mean_lifespan, *_ = par.unbind(dim=1)

        mu = 1. / mean_lifespan
        kappa = 1. / tau_E
        gamma = 1. / tau_I

        # Constants
        beta_pre_factor = (kappa + mu) * (gamma + mu) / kappa
        beta = beta_pre_factor * R0

        # Apply movement model
        Sp = S + net_movement(S)
        Ep = E + net_movement(E)
        Ip = I + net_movement(I)
        Rp = R + net_movement(R)

        Np = Sp + Ep + Ip + Rp  # Elementwise addition of compartments

        # Total number of individuals present in j
        N_sum = torch.sum(Np, dim=-2)  # sum over rows
        # Calculate force of infection  (Note: error factor to prevent
        # possible 0 / 0 if no population in location j)
        foi = (beta[:, None] * local_contact * torch.sum(Ip, dim=-2)
               / (N_sum + 1e-15))
        # Same foi experienced by everyone in j:
        foi_tile = torch.unsqueeze(foi, 1)
        foi_tile = torch.tile(foi_tile,  dims=(1, self.n_c, 1))

        # Births only occur in origin county
        # births = torch.diag(torch.ones(size=(self.n_c,), dtype=torch.int64))
        # np.diag(rng.poisson(delta * mu * (1 - v) * N_vector))

        ones_mat = torch.ones(size=(self.n_particles, self.n_c, self.n_c))

        p_S = 1 - torch.exp(-delta * foi_tile)
        p_E = 1 - torch.exp(-delta * kappa[:, None, None] * ones_mat)
        p_I = 1 - torch.exp(-delta * gamma[:, None, None] * ones_mat)
        p_R = 1 - torch.exp(-delta * mu[:, None, None] * ones_mat)


        fromS = Binomial(total_count=Sp, probs=p_S).sample()
        fromE = Binomial(total_count=Ep, probs=p_E).sample()
        fromI = Binomial(total_count=Ip, probs=p_I).sample()
        fromR = Binomial(total_count=Rp, probs=p_R).sample()

        # State updates (multinomial version)
        # S = Sp - fromS.sum(-1) + births
        # E = Ep - fromE.sum(-1) + fromS[:, :, 0]
        # I = Ip - fromI.sum(-1) + fromE[:, :, 0]
        # R = Rp - fromR + fromI[:, :, 0]

        S = Sp - fromS  # + births
        E = Ep - fromE + fromS
        I = Ip - fromI + fromE
        R = Rp - fromR + fromI
        C = C + fromI

        return torch.stack((S, E, I, R, C), dim=1).to(torch.int64)

    def simulate_euler_forward(self, y0, pars, t0=None, tf=None, timestep=None,
                               t_eval=None, y_obs=None, Nmif=None):

        """
        Note: Need to check  t_eval.shape must be equal to y_obs.shape[0]
        :param y0:
        :param pars:
        :param t0:
        :param tf:
        :param timestep:
        :param t_eval:
        :param y_obs:
        :param Nmif:
        :return:
        """

        if t0 is None:
            t0 = self.t0

        if tf is None:
            tf = self.tf

        if t_eval is None:
            t_eval = self.t_eval

        if timestep is None:
            timestep = self.timestep


        par_tensor = self.construct_parameter_tensor(pars=pars)
        if Nmif is not None:
            par_out = torch.zeros(size=(t_eval.shape[0] * Nmif,
                                        par_tensor.shape[-1]),
                                   dtype=torch.float64)

        for n in range(0, Nmif) if Nmif is not None else range(1):
            # Initialise state and output tensor
            t = t0
            y = y0  # self.initialise_model(N_vector=N_vector)
            i = 0

            y_out = torch.zeros(size=(t_eval.shape[0], *y.shape[:-1]),
                               dtype=torch.int64)  # pre-allocate output memory

            obs_out = torch.zeros(size=(t_eval.shape[0], y.shape[0],
                                        *y.shape[2:-1]),
                                 dtype=torch.int64)

            m = 0  # Month counter

            self.removed_history.clear()

            # Initialized using first month's data
            M_out, M_back, local_contact = (
                self.get_connectivity_tensors(m, par_tensor, t))

            outbreak_seeded = False  # logical condition

            while t < tf:

                if (m < self.month_start_tensor.shape[0]
                        and t > self.month_start_tensor[m]):
                    m += 1

                if not outbreak_seeded and t >= self.seed_day:
                    y = self.seed_outbreak(y)
                    outbreak_seeded = True

                M_out, M_back, local_contact = (
                    self.get_connectivity_tensors(m, par_tensor, t))

                y = self.process_model(t=t, X=y, par=par_tensor,
                                       M_out=M_out, M_back=M_back,
                                       local_contact=local_contact)
                t += timestep

                if (i < len(t_eval)) and (t > t_eval[i]):
                    y_reduced = y.sum(dim=-1)
                    y_out[i] = y_reduced
                    if self.death_delay_weeks:
                        self.record_removed(y)
                    obs_out[i] = self.reporting_model(y=y,
                                                      par=par_tensor).sample()

                    if y_obs is not None:
                        imp_sample = self.importance_sampling(t,
                                                              y_prev=y,
                                                              par_prev=par_tensor,
                                                              y_obs=y_obs[i])

                        y, par_tensor, cond_log_likelihood = imp_sample

                        if Nmif is not None:
                            n_obs = len(y_obs)
                            rw_exponent = (i + n_obs * n) / (n_obs * Nmif)
                            rw_sd = self.rw_sd * self.rw_alpha ** rw_exponent
                            par_tensor = self.parameter_random_walk(t=t,
                                                                    par=par_tensor,
                                                                    rw_sd=rw_sd)
                            par_out[i + n_obs * n] = par_tensor.mean(axis=0)

                    y = self.reset_cumulative_variables(y)
                    i += 1

                print(t)

        if Nmif is None:
            return y_out, obs_out
        else:
            return y_out, obs_out, par_out
