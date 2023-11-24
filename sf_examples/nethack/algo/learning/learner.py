from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from sample_factory.algo.learning.learner import Learner
from sample_factory.algo.learning.rnn_utils import build_core_out_from_seq, build_rnn_inputs
from sample_factory.algo.utils.action_distributions import get_action_distribution
from sample_factory.algo.utils.env_info import EnvInfo
from sample_factory.algo.utils.misc import LEARNER_ENV_STEPS, POLICY_ID_KEY, STATS_KEY, TRAIN_STATS, memory_stats
from sample_factory.algo.utils.model_sharing import ParameterServer
from sample_factory.algo.utils.rl_utils import prepare_and_normalize_obs
from sample_factory.algo.utils.shared_buffers import alloc_trajectory_tensors
from sample_factory.algo.utils.tensor_dict import TensorDict
from sample_factory.algo.utils.tensor_utils import ensure_torch_tensor
from sample_factory.algo.utils.torch_utils import masked_select, synchronize, to_scalar
from sample_factory.model.model_utils import get_rnn_size
from sample_factory.utils.attr_dict import AttrDict
from sample_factory.utils.typing import ActionDistribution, Config, InitModelData, PolicyID
from sample_factory.utils.utils import log
from sf_examples.nethack.algo.sampling.rollout_worker import DatasetRolloutWorker
from sf_examples.nethack.datasets.dataset_info import DatasetInfo, extract_dataset_info


class DatasetLearner(Learner):
    def __init__(
        self,
        cfg: Config,
        env_info: EnvInfo,
        policy_versions_tensor: Tensor,
        policy_id: PolicyID,
        param_server: ParameterServer,
    ):
        super().__init__(
            cfg,
            env_info,
            policy_versions_tensor,
            policy_id,
            param_server,
        )

        self.dataset_info: DatasetInfo = None

        self.dataset_idx = 0
        self.dataset_rollout_worker = None
        self.dataset_last_rnn_states = []
        self.dataset_training_batches = []

        self.supervised_loss_func: Optional[Callable] = None
        self.distillation_loss_func: Optional[Callable] = None
        self.kickstarting_loss_func: Optional[Callable] = None

    def init(self) -> InitModelData:
        init_model_data = super().init()

        if self.cfg.use_dataset:
            assert self.cfg.rollout == self.cfg.dataset_rollout

            self.dataset_rollout_worker = DatasetRolloutWorker(self.cfg)
            self.dataset_info = extract_dataset_info(self.dataset_rollout_worker.dataset, self.cfg)

            self.max_batches_to_accumulate = self.cfg.dataset_num_splits

            for i in range(self.max_batches_to_accumulate):
                rnn_size = get_rnn_size(self.cfg)
                training_batch = alloc_trajectory_tensors(
                    self.dataset_info,
                    self.dataset_rollout_worker.dataset.dataset_batch_size,
                    self.cfg.dataset_rollout,
                    rnn_size,
                    self.device,
                    False,
                )
                rnn_states = torch.zeros_like(training_batch["rnn_states"][:, 0])
                self.dataset_last_rnn_states.append(rnn_states)
                self.dataset_training_batches.append(training_batch)

        use_supervised_loss = self.cfg.supervised_loss_coeff > 0.0
        use_distillation_loss = self.cfg.distillation_loss_coeff > 0.0
        use_kickstarting_loss = self.cfg.kickstarting_loss_coeff > 0.0

        assert (
            sum(
                [
                    use_supervised_loss,
                    use_distillation_loss,
                    use_kickstarting_loss,
                ]
            )
            <= 1
        ), "only one regularization loss allowed at the time"

        assert (
            use_supervised_loss or use_distillation_loss
        ) == self.cfg.use_dataset, (
            "If either 'use_supervised_loss' or 'use_distillation_loss' is true, then 'use_dataset' must also be true."
        )

        self.supervised_loss_func = self._supervised_loss if use_supervised_loss else lambda *_: 0.0
        self.distillation_loss_func = self._distillation_loss if use_distillation_loss else lambda *_: 0.0
        self.kickstarting_loss_func = self._kickstarting_loss if use_kickstarting_loss else lambda *_: 0.0

        return init_model_data

    def _supervised_loss(self, mb_results, mb, num_invalids: int):
        result = mb_results["result"]
        valids = mb_results["valids"]
        outputs = result["action_logits"]
        targets = mb["normalized_obs"]["actions_converted"].reshape(-1).long()
        supervised_loss = F.cross_entropy(outputs, targets, reduction="none")
        supervised_loss = masked_select(supervised_loss, valids, num_invalids)
        supervised_loss *= self.cfg.supervised_loss_coeff
        supervised_loss = supervised_loss.mean()

        return supervised_loss

    def _distillation_loss(self, mb_results, mb, num_invalids: int):
        result = mb_results["result"]
        valids = mb_results["valids"]

        # we want to be equivalent to reduction "batchmean",
        # first we will sum div on every single distribution and leave the batch intact
        # after masked_select we will average what is left
        distillation_loss = F.kl_div(
            F.log_softmax(result["action_logits"], dim=-1),
            F.log_softmax(result["kick_action_logits"], dim=-1),
            log_target=True,
            reduction="none",
        ).sum(axis=1)
        distillation_loss = masked_select(distillation_loss, valids, num_invalids)
        distillation_loss *= self.cfg.distillation_loss_coeff
        distillation_loss = distillation_loss.mean()

        return distillation_loss

    def _kickstarting_loss(self, result, valids, num_invalids: int):
        # we want to be equivalent to reduction "batchmean",
        # first we will sum div on every single distribution and leave the batch intact
        # after masked_select we will average what is left
        kickstarting_loss = F.kl_div(
            F.log_softmax(result["action_logits"], dim=-1),
            F.log_softmax(result["kick_action_logits"], dim=-1),
            log_target=True,
            reduction="none",
        ).sum(axis=1)
        kickstarting_loss = masked_select(kickstarting_loss, valids, num_invalids)
        kickstarting_loss *= self.cfg.kickstarting_loss_coeff
        kickstarting_loss = kickstarting_loss.mean()

        return kickstarting_loss

    def _sample_dataset_batch(self):
        batch = self.dataset_training_batches[self.dataset_idx]
        rnn_states = self.dataset_last_rnn_states[self.dataset_idx]
        # TODO: do we want to keep hidden states between rollouts?
        if self.cfg.reset_on_rollout_boundary == True:
            rnn_states[:] = 0
        obs = self.dataset_rollout_worker.sample_batch(self.dataset_idx)

        with torch.no_grad():
            with self.timing.add_time("stack"):
                for key, x in obs.items():
                    obs[key] = ensure_torch_tensor(x)

            num_samples, seq_len = batch["rewards"].shape[0], batch["rewards"].shape[1]

            with self.timing.add_time("obs_to_device_normalize"):
                actor_critic = self.actor_critic
                if actor_critic.training:
                    actor_critic.eval()  # need to call this because we can be in serial mode

                normalized_obs = prepare_and_normalize_obs(actor_critic, obs)
                rnn_states = ensure_torch_tensor(rnn_states).to(self.device).float()
                batch["rnn_states"][:, 0] = rnn_states

            batch["obs"] = normalized_obs
            batch["dones"] = normalized_obs["dones"].bool()[:, :-1]
            batch["policy_id"].fill_(self.policy_id)
            batch["valids"] = normalized_obs["mask"].bool()
            # TODO: maybe also return rewards calculated based on scores

            with self.timing.add_time("forward"):
                for i in range(seq_len):
                    rnn_state = batch["rnn_states"][:, i]
                    i_normalized_obs = {key: value[:, i] for key, value in normalized_obs.items()}
                    policy_outputs = actor_critic(i_normalized_obs, rnn_state)
                    policy_outputs["policy_version"] = torch.empty([num_samples]).fill_(self.train_step)
                    # this workaround results from mismatch between policy_outputs["actions"] and batch["actions"]
                    # in normal code this is resolved with policy_output_tensor, which concatenates all policy_outputs
                    policy_outputs["actions"].unsqueeze_(-1)
                    # reset next-step hidden states to zero if we encountered an episode boundary
                    # not sure if this is the best practice, but this is what everybody seems to be doing
                    not_done = (1.0 - i_normalized_obs["dones"].float()).unsqueeze(-1)
                    batch["rnn_states"][:, i + 1] = (
                        policy_outputs["new_rnn_states"] * not_done * i_normalized_obs["mask"].unsqueeze(-1)
                    )
                    # we don't store new_rnn_states in buffer
                    del policy_outputs["new_rnn_states"]
                    # store policy outputs in batch
                    batch[:, i] = policy_outputs

        self.dataset_last_rnn_states[self.dataset_idx] = batch["rnn_states"][:, i + 1]
        self.dataset_idx = (self.dataset_idx + 1) % self.cfg.dataset_num_splits

        return batch

    def _compute_model_outputs(self, mb: TensorDict, num_invalids: int):
        with torch.no_grad(), self.timing.add_time("losses_init"):
            recurrence: int = self.cfg.recurrence
            valids = mb.valids

        # calculate policy head outside of recurrent loop
        with self.timing.add_time("forward_head"):
            head_outputs = self.actor_critic.forward_head(mb.normalized_obs)
            minibatch_size: int = head_outputs.size(0)

        # initial rnn states
        with self.timing.add_time("bptt_initial"):
            if self.cfg.use_rnn:
                # this is the only way to stop RNNs from backpropagating through invalid timesteps
                # (i.e. experience collected by another policy)
                done_or_invalid = torch.logical_or(mb.dones_cpu, ~valids.cpu()).float()
                head_output_seq, rnn_states, inverted_select_inds = build_rnn_inputs(
                    head_outputs,
                    done_or_invalid,
                    mb.rnn_states,
                    recurrence,
                )
            else:
                rnn_states = mb.rnn_states[::recurrence]

        # calculate RNN outputs for each timestep in a loop
        with self.timing.add_time("bptt"):
            if self.cfg.use_rnn:
                with self.timing.add_time("bptt_forward_core"):
                    core_output_seq, _ = self.actor_critic.forward_core(head_output_seq, rnn_states)
                core_outputs = build_core_out_from_seq(core_output_seq, inverted_select_inds)
                del core_output_seq
            else:
                core_outputs, _ = self.actor_critic.forward_core(head_outputs, rnn_states)

            del head_outputs

        num_trajectories = minibatch_size // recurrence
        assert core_outputs.shape[0] == minibatch_size

        with self.timing.add_time("tail"):
            # calculate policy tail outside of recurrent loop
            result = self.actor_critic.forward_tail(core_outputs, values_only=False, sample_actions=False)
            action_distribution = self.actor_critic.action_distribution()
            log_prob_actions = action_distribution.log_prob(mb.actions)
            ratio = torch.exp(log_prob_actions - mb.log_prob_actions)  # pi / pi_old

            # super large/small values can cause numerical problems and are probably noise anyway
            ratio = torch.clamp(ratio, 0.05, 20.0)

            values = result["values"].squeeze()

            del core_outputs

        # these computations are not the part of the computation graph
        with torch.no_grad(), self.timing.add_time("advantages_returns"):
            if self.cfg.with_vtrace:
                # V-trace parameters
                rho_hat = torch.Tensor([self.cfg.vtrace_rho])
                c_hat = torch.Tensor([self.cfg.vtrace_c])

                ratios_cpu = ratio.cpu()
                values_cpu = values.cpu()
                rewards_cpu = mb.rewards_cpu
                dones_cpu = mb.dones_cpu

                vtrace_rho = torch.min(rho_hat, ratios_cpu)
                vtrace_c = torch.min(c_hat, ratios_cpu)

                vs = torch.zeros((num_trajectories * recurrence))
                adv = torch.zeros((num_trajectories * recurrence))

                next_values = values_cpu[recurrence - 1 :: recurrence] - rewards_cpu[recurrence - 1 :: recurrence]
                next_values /= self.cfg.gamma
                next_vs = next_values

                for i in reversed(range(self.cfg.recurrence)):
                    rewards = rewards_cpu[i::recurrence]
                    dones = dones_cpu[i::recurrence]
                    not_done = 1.0 - dones
                    not_done_gamma = not_done * self.cfg.gamma

                    curr_values = values_cpu[i::recurrence]
                    curr_vtrace_rho = vtrace_rho[i::recurrence]
                    curr_vtrace_c = vtrace_c[i::recurrence]

                    delta_s = curr_vtrace_rho * (rewards + not_done_gamma * next_values - curr_values)
                    adv[i::recurrence] = curr_vtrace_rho * (rewards + not_done_gamma * next_vs - curr_values)
                    next_vs = curr_values + delta_s + not_done_gamma * curr_vtrace_c * (next_vs - next_values)
                    vs[i::recurrence] = next_vs

                    next_values = curr_values

                targets = vs.to(self.device)
                adv = adv.to(self.device)
            else:
                # using regular GAE
                adv = mb.advantages
                targets = mb.returns

            adv_std, adv_mean = torch.std_mean(masked_select(adv, valids, num_invalids))
            adv = (adv - adv_mean) / torch.clamp_min(adv_std, 1e-7)  # normalize advantage

        return dict(
            result=result,
            ratio=ratio,
            adv=adv,
            adv_std=adv_std,
            adv_mean=adv_mean,
            values=values,
            targets=targets,
            valids=valids,
        )

    def _calculate_losses(
        self,
        mb: AttrDict,
        num_invalids: int,
        dataset_mb: Optional[AttrDict] = None,
        dataset_num_invalids: Optional[AttrDict] = None,
    ) -> Tuple[ActionDistribution, Tensor, Tensor | float, Optional[Tensor], Tensor | float, Tensor, Dict]:
        # PPO clipping
        clip_ratio_high = 1.0 + self.cfg.ppo_clip_ratio  # e.g. 1.1
        # this still works with e.g. clip_ratio = 2, while PPO's 1-r would give negative ratio
        clip_ratio_low = 1.0 / clip_ratio_high
        clip_value = self.cfg.ppo_clip_value

        mb_results = self._compute_model_outputs(mb, num_invalids)
        # we want action distribution (last) of the same shape as mb
        action_distribution = self.actor_critic.action_distribution()

        dataset_mb_results = self._compute_model_outputs(dataset_mb, dataset_num_invalids) if dataset_mb else None

        with self.timing.add_time("losses"):
            if not self.cfg.behavioral_clone:
                ratio = mb_results["ratio"]
                adv = mb_results["adv"]
                adv_mean = mb_results["adv_mean"]
                adv_std = mb_results["adv_std"]
                valids = mb_results["valids"]
                values = mb_results["values"]
                targets = mb_results["targets"]

                # noinspection PyTypeChecker
                policy_loss = self._policy_loss(ratio, adv, clip_ratio_low, clip_ratio_high, valids, num_invalids)
                exploration_loss = self.exploration_loss_func(action_distribution, valids, num_invalids)
                kl_old, kl_loss = self.kl_loss_func(
                    self.actor_critic.action_space, mb.action_logits, action_distribution, valids, num_invalids
                )
                old_values = mb["values"]
                value_loss = self._value_loss(values, old_values, targets, clip_value, valids, num_invalids)
            else:
                ratio = torch.tensor([0.0])
                policy_loss = 0.0
                exploration_loss = 0.0
                kl_old, kl_loss = torch.tensor([0.0]), 0.0
                value_loss = 0.0
                kickstarting_loss = 0.0
                adv, adv_mean, adv_std = 0.0, 0.0, 0.0
                values = torch.tensor([0.0]).to(self.device)

        with self.timing.add_time("regularizer_losses"):
            supervised_loss = self.supervised_loss_func(
                dataset_mb_results,
                dataset_mb,
                dataset_num_invalids,
            )
            distillation_loss = self.distillation_loss_func(
                dataset_mb_results,
                dataset_mb,
                dataset_num_invalids,
            )
            kickstarting_loss = self.kickstarting_loss_func(
                mb_results["result"],
                mb_results["valids"],
                num_invalids,
            )

        loss_summaries = dict(
            ratio=ratio,
            clip_ratio_low=clip_ratio_low,
            clip_ratio_high=clip_ratio_high,
            values=values,
            adv=adv,
            adv_std=adv_std,
            adv_mean=adv_mean,
        )
        regularizer_loss = supervised_loss + distillation_loss + kickstarting_loss
        regularizer_loss_summaries = dict(
            supervised_loss=to_scalar(supervised_loss),
            distillation_loss=to_scalar(distillation_loss),
            kickstarting_loss=to_scalar(kickstarting_loss),
        )

        return (
            action_distribution,
            policy_loss,
            exploration_loss,
            kl_old,
            kl_loss,
            value_loss,
            loss_summaries,
            regularizer_loss,
            regularizer_loss_summaries,
        )

    def _train(
        self,
        gpu_buffer: TensorDict,
        batch_size: int,
        experience_size: int,
        num_invalids: int,
        ds_gpu_buffer: Optional[TensorDict] = None,
        ds_batch_size: Optional[int] = None,
        ds_experience_size: Optional[int] = None,
        ds_num_invalids: Optional[int] = None,
    ) -> Optional[AttrDict]:
        timing = self.timing
        with torch.no_grad():
            early_stopping_tolerance = 1e-6
            early_stop = False
            prev_epoch_actor_loss = 1e9
            epoch_actor_losses = [0] * self.cfg.num_batches_per_epoch

            # recent mean KL-divergences per minibatch, this used by LR schedulers
            recent_kls = []

            if self.cfg.with_vtrace:
                assert (
                    self.cfg.recurrence == self.cfg.rollout and self.cfg.recurrence > 1
                ), "V-trace requires to recurrence and rollout to be equal"

            num_sgd_steps = 0
            stats_and_summaries: Optional[AttrDict] = None

            # When it is time to record train summaries, we randomly sample epoch/batch for which the summaries are
            # collected to get equal representation from different stages of training.
            # Half the time, we record summaries from the very large step of training. There we will have the highest
            # KL-divergence and ratio of PPO-clipped samples, which makes this data even more useful for analysis.
            # Something to consider: maybe we should have these last-batch metrics in a separate summaries category?
            with_summaries = self._should_save_summaries()
            if np.random.rand() < 0.5:
                summaries_epoch = np.random.randint(0, self.cfg.num_epochs)
                summaries_batch = np.random.randint(0, self.cfg.num_batches_per_epoch)
            else:
                summaries_epoch = self.cfg.num_epochs - 1
                summaries_batch = self.cfg.num_batches_per_epoch - 1

            assert self.actor_critic.training

        for epoch in range(self.cfg.num_epochs):
            with timing.add_time("epoch_init"):
                if early_stop:
                    break

                force_summaries = False
                minibatches = self._get_minibatches(batch_size, experience_size)
                ds_minibatches = (
                    self._get_minibatches(ds_batch_size, ds_experience_size) if ds_experience_size else None
                )

            for batch_num in range(len(minibatches)):
                with torch.no_grad(), timing.add_time("minibatch_init"):
                    indices = minibatches[batch_num]
                    ds_indices = ds_minibatches[batch_num] if ds_minibatches else None

                    # current minibatch consisting of short trajectory segments with length == recurrence
                    mb = self._get_minibatch(gpu_buffer, indices)
                    ds_mb = self._get_minibatch(ds_gpu_buffer, ds_indices) if ds_gpu_buffer else None

                    # enable syntactic sugar that allows us to access dict's keys as object attributes
                    mb = AttrDict(mb)
                    ds_mb = AttrDict(ds_mb) if ds_mb else None

                with timing.add_time("calculate_losses"):
                    (
                        action_distribution,
                        policy_loss,
                        exploration_loss,
                        kl_old,
                        kl_loss,
                        value_loss,
                        loss_summaries,
                        regularizer_loss,
                        regularizer_loss_summaries,
                    ) = self._calculate_losses(mb, num_invalids, ds_mb, ds_num_invalids)

                with timing.add_time("losses_postprocess"):
                    # noinspection PyTypeChecker
                    actor_loss: Tensor = policy_loss + exploration_loss + kl_loss
                    critic_loss = value_loss
                    loss: Tensor = actor_loss + critic_loss + regularizer_loss

                    epoch_actor_losses[batch_num] = float(actor_loss)

                    high_loss = 30.0
                    if torch.abs(loss) > high_loss:
                        log.warning(
                            "High loss value: l:%.4f pl:%.4f vl:%.4f exp_l:%.4f kl_l:%.4f (recommended to adjust the --reward_scale parameter)",
                            to_scalar(loss),
                            to_scalar(policy_loss),
                            to_scalar(value_loss),
                            to_scalar(exploration_loss),
                            to_scalar(kl_loss),
                        )

                        # perhaps something weird is happening, we definitely want summaries from this step
                        force_summaries = True

                with torch.no_grad(), timing.add_time("kl_divergence"):
                    # if kl_old is not None it is already calculated above
                    if kl_old is None:
                        # calculate KL-divergence with the behaviour policy action distribution
                        old_action_distribution = get_action_distribution(
                            self.actor_critic.action_space,
                            mb.action_logits,
                        )
                        kl_old = action_distribution.kl_divergence(old_action_distribution)
                        kl_old = masked_select(kl_old, mb.valids, num_invalids)

                    kl_old_mean = float(kl_old.mean().item())
                    recent_kls.append(kl_old_mean)
                    if kl_old.numel() > 0 and kl_old.max().item() > 100:
                        log.warning(f"KL-divergence is very high: {kl_old.max().item():.4f}")

                # update the weights
                with timing.add_time("update"):
                    # following advice from https://youtu.be/9mS1fIYj1So set grad to None instead of optimizer.zero_grad()
                    for p in self.actor_critic.parameters():
                        p.grad = None

                    loss.backward()

                    if self.cfg.max_grad_norm > 0.0:
                        with timing.add_time("clip"):
                            torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.cfg.max_grad_norm)

                    curr_policy_version = self.train_step  # policy version before the weight update

                    actual_lr = self.curr_lr
                    if num_invalids > 0:
                        # if we have masked (invalid) data we should reduce the learning rate accordingly
                        # this prevents a situation where most of the data in the minibatch is invalid
                        # and we end up doing SGD with super noisy gradients
                        actual_lr = self.curr_lr * (experience_size - num_invalids) / experience_size
                    self._apply_lr(actual_lr)

                    with self.param_server.policy_lock:
                        self.optimizer.step()

                    num_sgd_steps += 1

                with torch.no_grad(), timing.add_time("after_optimizer"):
                    self._after_optimizer_step()

                    if self.lr_scheduler.invoke_after_each_minibatch():
                        self.curr_lr = self.lr_scheduler.update(self.curr_lr, recent_kls)

                    # collect and report summaries
                    should_record_summaries = with_summaries
                    should_record_summaries &= epoch == summaries_epoch and batch_num == summaries_batch
                    should_record_summaries |= force_summaries
                    if should_record_summaries:
                        # hacky way to collect all of the intermediate variables for summaries
                        summary_vars = {**locals(), **loss_summaries}
                        stats_and_summaries = self._record_summaries(AttrDict(summary_vars))
                        # dont have a better way to do this then to modify record summaries
                        for key, value in regularizer_loss_summaries.items():
                            stats_and_summaries[key] = value

                        del summary_vars
                        force_summaries = False

                    # make sure everything (such as policy weights) is committed to shared device memory
                    synchronize(self.cfg, self.device)
                    # this will force policy update on the inference worker (policy worker)
                    self.policy_versions_tensor[self.policy_id] = self.train_step

            # end of an epoch
            if self.lr_scheduler.invoke_after_each_epoch():
                self.curr_lr = self.lr_scheduler.update(self.curr_lr, recent_kls)

            new_epoch_actor_loss = float(np.mean(epoch_actor_losses))
            loss_delta_abs = abs(prev_epoch_actor_loss - new_epoch_actor_loss)
            if loss_delta_abs < early_stopping_tolerance:
                early_stop = True
                log.debug(
                    "Early stopping after %d epochs (%d sgd steps), loss delta %.7f",
                    epoch + 1,
                    num_sgd_steps,
                    loss_delta_abs,
                )
                break

            prev_epoch_actor_loss = new_epoch_actor_loss

        return stats_and_summaries

    def train(self, batch: TensorDict) -> Optional[Dict]:
        if self.cfg.save_milestones_ith > 0 and self.env_steps // self.cfg.save_milestones_ith > self.checkpoint_steps:
            self.save_milestone()
            self.checkpoint_steps = self.env_steps // self.cfg.save_milestones_ith

        with self.timing.add_time("misc"):
            self._maybe_update_cfg()
            self._maybe_load_policy()

        with self.timing.add_time("sample_dataset_batch"):
            dataset_batch = self._sample_dataset_batch() if self.cfg.use_dataset else None

        with self.timing.add_time("prepare_batch"):
            buff, experience_size, num_invalids = self._prepare_batch(batch)
            if dataset_batch:
                ds_buff, ds_experience_size, ds_num_invalids = self._prepare_batch(dataset_batch)
            else:
                ds_buff, ds_experience_size, ds_num_invalids = None, None, None

        if num_invalids >= experience_size:
            if self.cfg.with_pbt:
                log.warning("No valid samples in the batch, with PBT this must mean we just replaced weights")
            else:
                log.error(f"Learner {self.policy_id=} received an entire batch of invalid data, skipping...")
            return None
        else:
            with self.timing.add_time("train"):
                train_stats = self._train(
                    buff,
                    self.cfg.batch_size,
                    experience_size,
                    num_invalids,
                    ds_buff,
                    self.cfg.dataset_batch_size,
                    ds_experience_size,
                    ds_num_invalids,
                )

            # multiply the number of samples by frameskip so that FPS metrics reflect the number
            # of environment steps actually simulated
            if self.cfg.behavioral_clone:
                self.env_steps += ds_experience_size
            else:
                if self.cfg.summaries_use_frameskip:
                    self.env_steps += experience_size * self.env_info.frameskip
                else:
                    self.env_steps += experience_size

            stats = {LEARNER_ENV_STEPS: self.env_steps, POLICY_ID_KEY: self.policy_id}
            if train_stats is not None:
                if train_stats is not None:
                    stats[TRAIN_STATS] = train_stats
                stats[STATS_KEY] = memory_stats("learner", self.device)

            return stats
