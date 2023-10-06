from typing import Dict, Iterator, List, Optional, Set, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from sample_factory.algo.utils.tensor_dict import TensorDict
from sample_factory.model.actor_critic import ActorCritic


class KickStarter(nn.Module):
    def __init__(self, student: ActorCritic, teacher: ActorCritic, always_run_teacher=True, run_teacher_hs=False):
        super().__init__()
        self.student = student
        self.teacher = teacher
        self.teacher.eval()
        self.teacher.requires_grad_(False)
        self._k = "kick_"
        self.always_run_teacher = always_run_teacher
        self.run_teacher_hs = run_teacher_hs
        self.num_models = 2

        self.action_space = self.student.action_space
        self.returns_normalizer = self.student.returns_normalizer

    def named_modules(self, memo: Set[Module] | None = None, prefix: str = "", remove_duplicate: bool = True):
        # we override only named_modules, since named_buffers, named_parameters and parameters are dependent
        # returning only student parameters will speed up the training when using kickstarting
        # since optimizer will not have to keep track of teacher paramterers
        return self.student.named_modules(memo, prefix, remove_duplicate)

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
        return self.student._maybe_sample_actions(sample_actions, result)

    def policy_output_shapes(self, num_actions, num_action_distribution_parameters) -> List[Tuple[str, List]]:
        return self.student.policy_output_shapes(num_actions, num_action_distribution_parameters)

    def forward(self, normalized_obs_dict, rnn_states, values_only=False) -> TensorDict:
        rnn_states_split = rnn_states.chunk(self.num_models, dim=1)

        student_outputs = self.student.forward(normalized_obs_dict, rnn_states_split[0], values_only)
        teacher_outputs = self.teacher.forward(normalized_obs_dict, rnn_states_split[1], values_only)

        new_rnn_states = torch.cat([student_outputs["new_rnn_states"], teacher_outputs["new_rnn_states"]], dim=1)
        student_outputs["new_rnn_states"] = new_rnn_states

        return student_outputs

    def forward_head(self, normalized_obs_dict: Dict):
        student_head_output = self.student.forward_head(normalized_obs_dict)
        teacher_head_output = self.teacher.forward_head(normalized_obs_dict)

        return torch.cat([student_head_output, teacher_head_output], dim=1)

    def forward_core(self, head_output, rnn_states):
        unpacked_head_output, lengths = pad_packed_sequence(head_output)
        unpacked_head_output_split = unpacked_head_output.chunk(self.num_models, dim=2)
        student_head_output = pack_padded_sequence(unpacked_head_output_split[0], lengths, enforce_sorted=False)
        teacher_head_output = pack_padded_sequence(unpacked_head_output_split[1], lengths, enforce_sorted=False)

        student_rnn_states, teacher_rnn_states = rnn_states.chunk(self.num_models, dim=1)

        student_output, student_new_rnn_state = self.student.forward_core(student_head_output, student_rnn_states)
        teacher_output, teacher_new_rnn_state = self.teacher.forward_core(teacher_head_output, teacher_rnn_states)

        unpacked_student_output, lengths = pad_packed_sequence(student_output)
        unpacked_teacher_output, lengths = pad_packed_sequence(teacher_output)
        unpacked_outputs = torch.cat([unpacked_student_output, unpacked_teacher_output], dim=2)
        outputs = pack_padded_sequence(unpacked_outputs, lengths, enforce_sorted=False)

        new_rnn_states = torch.cat([student_new_rnn_state, teacher_new_rnn_state], dim=1)

        return outputs, new_rnn_states

    def forward_tail(self, core_output, values_only: bool, sample_actions: bool) -> TensorDict:
        core_outputs_split = core_output.chunk(self.num_models, dim=1)

        student_outputs = self.student.forward_tail(core_outputs_split[0], values_only, sample_actions)
        teacher_outputs = self.teacher.forward_tail(core_outputs_split[1], values_only, sample_actions)

        student_outputs[self._k + "action_logits"] = teacher_outputs["action_logits"]

        return student_outputs

    # def policy_output_shapes(self, num_actions, num_action_distribution_parameters) -> List[Tuple[str, List]]:
    #     # policy outputs, this matches the expected output of the actor-critic
    #     policy_outputs = self.student.policy_output_shapes(num_actions, num_action_distribution_parameters)
    #     policy_outputs.append((self._k + "action_logits", [num_action_distribution_parameters]))
    #     return policy_outputs
