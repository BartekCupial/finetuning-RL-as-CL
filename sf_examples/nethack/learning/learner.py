from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from sample_factory.algo.learning.learner import Learner
from sample_factory.algo.utils.action_distributions import get_action_distribution
from sample_factory.algo.utils.env_info import EnvInfo
from sample_factory.algo.utils.model_sharing import ParameterServer
from sample_factory.algo.utils.rl_utils import prepare_and_normalize_obs
from sample_factory.algo.utils.shared_buffers import init_tensor
from sample_factory.algo.utils.tensor_dict import TensorDict
from sample_factory.algo.utils.tensor_utils import ensure_torch_tensor
from sample_factory.algo.utils.torch_utils import masked_select, synchronize, to_scalar
from sample_factory.model.model_utils import get_rnn_size
from sample_factory.utils.attr_dict import AttrDict
from sample_factory.utils.typing import Config, InitModelData, PolicyID
from sample_factory.utils.utils import log
from sf_examples.nethack.datasets.dataset import TtyrecDatasetWorker


class NetHackLearner(Learner):
    def __ini__(
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

        self.dataset = None
        self.dataset_rnn_states = None

        self.supervised_loss_func: Optional[Callable] = None
        self.kickstarting_loss_func: Optional[Callable] = None

    def init(self) -> InitModelData:
        init_model_data = super().init()

        supervised_learning = self.cfg.supervised_loss_coeff != 0
        # TODO: distillation
        self.use_dataset = supervised_learning

        if self.use_dataset:
            self.dataset = TtyrecDatasetWorker(self.cfg)

            rnn_size = get_rnn_size(self.cfg)
            sampling_device = "cpu"  # TODO: check if we can use gpu here
            dataset_num_traj = self.cfg.dataset_batch_size // self.cfg.dataset_rollout
            self.dataset_rnn_states = []
            for _ in range(self.cfg.dataset_num_splits):
                hs = init_tensor([dataset_num_traj], torch.float32, [rnn_size], sampling_device, False)
                self.dataset_rnn_states.append(hs)

        if self.cfg.supervised_loss_coeff == 0.0:
            self.supervised_loss_func = lambda dataset_mb: 0.0
        else:
            self.supervised_loss_func = self._supervised_loss

        if self.cfg.kickstarting_loss_coeff == 0.0:
            self.kickstarting_loss_func = lambda mb, valids, num_invalids: 0.0
        else:
            self.kickstarting_loss_func = self._kickstarting_loss

        return init_model_data

    def _sample_dataset(self):
        # TODO: split this code into pieces, for loop should be inside model, only keep the loss related stuff
        dataset_mb = self.dataset.result()
        idx = self.dataset.idx

        obs = {
            key: value
            for key, value in dataset_mb.items()
            if key in ["tty_chars", "tty_colors", "tty_cursor", "screen_image"]
        }
        normalized_obs = prepare_and_normalize_obs(self.actor_critic, obs)
        dataset_mb["normalized_obs"] = normalized_obs

        rnn_states = self.dataset_rnn_states[idx]
        # TODO: do we want to keep hidden states between rollouts?
        rnn_states = torch.zeros_like(rnn_states)
        rnn_states = ensure_torch_tensor(rnn_states).to(self.device).float()

        dones = dataset_mb["dones"]
        for i in range(len(dones)):
            step_dones = dones[i]
            # Reset rnn_state to zero whenever an episode ended
            # Make sure it is broadcastable with rnn_states
            rnn_states = rnn_states * (~step_dones).float().view(-1, 1)
            step_normalized_obs = {key: value[i] for key, value in dataset_mb["normalized_obs"].items()}

            policy_outputs = self.actor_critic(step_normalized_obs, rnn_states)
            rnn_states = policy_outputs["new_rnn_states"]

            for key, value in policy_outputs.items():
                if key not in dataset_mb:
                    dataset_mb[key] = []
                dataset_mb[key].append(value)

        for key in policy_outputs.keys():
            dataset_mb[key] = torch.concatenate(dataset_mb[key])

        # store rnn_states, for next rollout
        self.dataset_rnn_states[idx] = rnn_states
        return dataset_mb

    def _kickstarting_loss(self, mb, valids, num_invalids: int):
        student_logits = mb.action_logits
        teacher_logits = mb.kick_action_logits

        kickstarting_loss = F.kl_div(
            F.log_softmax(student_logits, dim=-1),
            F.log_softmax(teacher_logits, dim=-1),
            log_target=True,
            reduction="none",
        )
        kickstarting_loss = masked_select(kickstarting_loss, valids, num_invalids)
        kickstarting_loss = kickstarting_loss.mean()
        kickstarting_loss *= self.cfg.kickstarting_loss_coeff

        return kickstarting_loss

    def _supervised_loss(self, dataset_mb):
        outputs = dataset_mb["action_logits"]
        targets = dataset_mb["actions_converted"].reshape(-1)
        supervised_loss = F.cross_entropy(outputs, targets)

        supervised_loss = supervised_loss.mean()
        supervised_loss *= self.cfg.supervised_loss_coeff

        return supervised_loss

    def _regularizer_losses(self, mb: AttrDict, num_invalids: int, dataset_mb: AttrDict):
        supervised_loss = self.supervised_loss_func(dataset_mb)
        valids = mb.valids
        kickstarting_loss = self.kickstarting_loss_func(mb, valids, num_invalids)
        loss_summaries = dict(
            supervised_loss=to_scalar(supervised_loss),
            kickstarting_loss=to_scalar(kickstarting_loss),
        )
        regularized_loss = supervised_loss + kickstarting_loss

        return regularized_loss, loss_summaries

    def _train(
        self, gpu_buffer: TensorDict, batch_size: int, experience_size: int, num_invalids: int
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

            for batch_num in range(len(minibatches)):
                with torch.no_grad(), timing.add_time("minibatch_init"):
                    indices = minibatches[batch_num]

                    # current minibatch consisting of short trajectory segments with length == recurrence
                    mb = self._get_minibatch(gpu_buffer, indices)

                    # enable syntactic sugar that allows us to access dict's keys as object attributes
                    mb = AttrDict(mb)

                with timing.add_time("calculate_losses"):
                    (
                        action_distribution,
                        policy_loss,
                        exploration_loss,
                        kl_old,
                        kl_loss,
                        value_loss,
                        loss_summaries,
                    ) = self._calculate_losses(mb, num_invalids)

                with timing.add_time("sample_dataset"):
                    if self.use_dataset:
                        dataset_mb = self._sample_dataset()
                    else:
                        dataset_mb = None

                with timing.add_time("regularizer_losses"):
                    regularizer_loss, regularizer_loss_summaries = self._regularizer_losses(
                        mb, num_invalids, dataset_mb
                    )

                if self.use_dataset:
                    # Only call step when you are done with dataset data - it may get overwritten
                    self.dataset.step()

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
