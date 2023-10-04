from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from sample_factory.algo.utils.tensor_dict import TensorDict
from sample_factory.model.actor_critic import ActorCritic


class KickStarter(nn.Module):
    def __init__(self, student: ActorCritic, teacher: ActorCritic, always_run_teacher=True, run_teacher_hs=False):
        super().__init__()
        self.student = student
        self.teacher = teacher
        self.teacher.eval()
        self._k = "kick_"
        self.always_run_teacher = always_run_teacher
        self.run_teacher_hs = run_teacher_hs
        self.num_models = 2

        self.action_space = self.student.action_space
        self.returns_normalizer = self.student.returns_normalizer

    def get_action_parameterization(self, decoder_output_size: int):
        self.student.get_action_parameterization(decoder_output_size)

    def model_to_device(self, device):
        self.student.model_to_device(device)
        self.teacher.model_to_device(device)

    def device_for_input_tensor(self, input_tensor_name: str) -> torch.device:
        return self.student.device_for_input_tensor(input_tensor_name)

    def type_for_input_tensor(self, input_tensor_name: str) -> torch.dtype:
        return self.student.type_for_input_tensor(input_tensor_name)

    def initialize_weights(self, layer):
        return self.student.initialize_weights(layer)

    def normalize_obs(self, obs: Dict[str, Tensor]) -> Dict[str, Tensor]:
        return self.student.normalize_obs(obs)

    def summaries(self) -> Dict:
        return self.student.summaries()

    def action_distribution(self):
        return self.student.action_distribution()

    def _maybe_sample_actions(self, sample_actions: bool, result: TensorDict) -> None:
        return self.student(sample_actions, result)

    def forward(self, normalized_obs_dict, rnn_states, values_only=False) -> TensorDict:
        rnn_states_split = rnn_states.chunk(self.num_models, dim=1)
        student_rnn_states, teacher_rnn_states = rnn_states_split[0], rnn_states_split[1]

        if self.run_teacher_hs:
            student_rnn_states = teacher_rnn_states

        student_outputs = self.student(normalized_obs_dict, student_rnn_states, values_only)

        if self.always_run_teacher or not self.training:
            with torch.no_grad():
                teacher_outputs = self.teacher(normalized_obs_dict, teacher_rnn_states, values_only)

            new_rnn_states = [student_outputs["new_rnn_states"], teacher_outputs["new_rnn_states"]]
            new_rnn_states = torch.cat(new_rnn_states, dim=1)
            student_outputs["new_rnn_states"] = new_rnn_states
            if not values_only:
                student_outputs[self._k + "action_logits"] = teacher_outputs["action_logits"]

        return student_outputs

    def forward_head(self, normalized_obs_dict: Dict):
        # this function is only used in learner, to reduce computation we only process student
        return self.student.forward_head(normalized_obs_dict)

    def forward_core(self, head_output, rnn_states):
        # in forward_head we process only student head so head output is already good
        # but we need to chunk rnn_states to process only student rnn_states
        rnn_states_split = rnn_states.chunk(self.num_models, dim=1)
        return self.student.forward_core(head_output, rnn_states_split[0])

    def forward_tail(self, core_output, values_only: bool, sample_actions: bool) -> TensorDict:
        # we can pass whole core_output to teacher since we only returned student core_output in forward_core
        return self.student.forward_tail(core_output, values_only, sample_actions)

    def policy_output_shapes(self, num_actions, num_action_distribution_parameters) -> List[Tuple[str, List]]:
        # policy outputs, this matches the expected output of the actor-critic
        policy_outputs = self.student.policy_output_shapes(num_actions, num_action_distribution_parameters)
        policy_outputs.append((self._k + "action_logits", [num_action_distribution_parameters]))
        return policy_outputs
