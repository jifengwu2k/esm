from abc import ABC, abstractmethod

import attr
import torch
from tqdm import tqdm

from esm.models.esm3 import ESM3
from esm.sdk.api import (
    ESMProtein,
    ESMProteinError,
    ESMProteinTensor,
    SamplingConfig,
    SamplingTrackConfig,
)


class GuidedDecodingScoringFunction(ABC):
    @abstractmethod
    def __call__(self, protein: ESMProtein) -> float:
        pass


class ESM3GuidedDecoding:
    """This class can be used to perform derivative-free guided decoding, based on
    the method described in "Derivative-Free Guidance in Continuous and Discrete Diffusion Models with Soft Value-Based Decoding"
    https://arxiv.org/abs/2408.08252
    """

    def __init__(
            self,
            client: ESM3,
            scoring_function: GuidedDecodingScoringFunction,
    ):
        self.client = client
        self.tokenizers = client.tokenizers
        self.scoring_function = scoring_function

    def guided_generate(
            self,
            protein: ESMProtein,
            num_decoding_steps: int,
            num_samples_per_step: int,
            denoised_prediction_temperature: float = 0.0,
            track: str = "sequence",
            verbose: bool = True,
    ) -> ESMProtein:
        protein_tensor = self.client.encode(protein)

        assert not isinstance(protein_tensor, ESMProteinError)

        if track == "structure":
            protein_tensor = self.maybe_add_default_structure_tokens(protein_tensor)

        num_masked_positions = self.get_number_of_masked_positions(
            protein_tensor, track=track
        )
        num_positions_to_unmask = num_masked_positions // num_decoding_steps

        current_score = -1

        if verbose:
            pbar = tqdm(range(num_decoding_steps), desc="Current score: -1")
        else:
            pbar = range(num_decoding_steps)

        for step in pbar:
            if step == num_decoding_steps - 1:
                # At the last step, unmask all remaining positions
                num_positions_to_unmask = self.get_number_of_masked_positions(
                    protein_tensor, track=track
                )

            samples = []
            scores = []
            for _ in range(num_samples_per_step):
                sample = self.randomly_unmask_positions(
                    protein_tensor, num_positions_to_unmask, track=track
                )
                scores.append(
                    self.reward_function(
                        sample,
                        denoised_prediction_temperature=denoised_prediction_temperature,
                    )
                )
                samples.append(sample)

            # Select best scoring sample
            best_sample = samples[scores.index(max(scores))]
            current_score = max(scores)
            protein_tensor = best_sample

            if verbose:
                pbar.set_description(f"Current score: {current_score:.2f}")  # type: ignore

        # Fully predict and decode final protein
        protein_tensor_output = self.client.forward_and_sample(
            protein_tensor,
            SamplingConfig(
                sequence=SamplingTrackConfig(temperature=0.0),
                structure=SamplingTrackConfig(temperature=0.0),
            ),
        )

        assert not isinstance(protein_tensor_output, ESMProteinError)
        protein_tensor = protein_tensor_output.protein_tensor

        decoded_protein = self.client.decode(protein_tensor)
        assert not isinstance(decoded_protein, ESMProteinError)
        return decoded_protein

    def reward_function(
            self,
            protein_tensor: ESMProteinTensor,
            denoised_prediction_temperature: float = 0.0,
    ) -> float:
        denoised_protein = self.predict_denoised(
            protein_tensor, temperature=denoised_prediction_temperature
        )
        return self.scoring_function(denoised_protein)

    def get_number_of_masked_positions(
            self, protein_tensor: ESMProteinTensor, track: str = "sequence"
    ) -> int:
        assert isinstance(protein_tensor, ESMProteinTensor)
        track_tensor = getattr(protein_tensor, track)
        track_tokenizer = getattr(self.tokenizers, track)
        is_mask = track_tensor == track_tokenizer.mask_token_id
        return is_mask.sum().item()  # type: ignore

    def randomly_unmask_positions(
            self,
            protein_tensor: ESMProteinTensor,
            num_positions_to_unmask: int,
            temperature: float = 1.0,
            track: str = "sequence",
    ) -> ESMProteinTensor:
        track_tensor = getattr(protein_tensor, track)
        assert track_tensor is not None
        protein_tensor = attr.evolve(protein_tensor)
        setattr(protein_tensor, track, track_tensor.clone())

        track_tensor = getattr(protein_tensor, track)
        track_tokenizer = getattr(self.tokenizers, track)

        is_mask = track_tensor == track_tokenizer.mask_token_id
        num_masked_positions = is_mask.sum().item()

        if num_positions_to_unmask > num_masked_positions:
            num_positions_to_unmask = num_masked_positions  # type: ignore

        mask_indices = is_mask.nonzero(as_tuple=False)
        mask_indices = mask_indices[torch.randperm(mask_indices.size(0))]
        mask_indices = mask_indices[:num_positions_to_unmask]

        sampling_config = SamplingConfig()
        setattr(sampling_config, track, SamplingTrackConfig(temperature=temperature))

        denoised_protein_tensor_output = self.client.forward_and_sample(
            protein_tensor, sampling_configuration=sampling_config
        )
        assert not isinstance(denoised_protein_tensor_output, ESMProteinError)
        denoised_protein_tensor = denoised_protein_tensor_output.protein_tensor
        output_track_tensor = getattr(denoised_protein_tensor, track)
        assert output_track_tensor is not None
        track_tensor[mask_indices] = output_track_tensor[mask_indices]
        setattr(protein_tensor, track, track_tensor)

        return protein_tensor

    def predict_denoised(
            self, protein_tensor: ESMProteinTensor, temperature: float = 0.0
    ) -> ESMProtein:
        denoised_protein_tensor_output = self.client.forward_and_sample(
            protein_tensor,
            sampling_configuration=SamplingConfig(
                sequence=SamplingTrackConfig(temperature=temperature),
                structure=SamplingTrackConfig(temperature=temperature),
            ),
        )
        assert not isinstance(denoised_protein_tensor_output, ESMProteinError)
        denoised_protein_tensor = denoised_protein_tensor_output.protein_tensor
        denoised_protein = self.client.decode(denoised_protein_tensor)
        assert not isinstance(denoised_protein, ESMProteinError)
        return denoised_protein

    def maybe_add_default_structure_tokens(
            self, protein_tensor: ESMProteinTensor
    ) -> ESMProteinTensor:
        empty_protein_tensor = ESMProteinTensor.empty(
            length=len(protein_tensor) - 2,
            tokenizers=self.tokenizers,
            device=protein_tensor.device,
        )
        if protein_tensor.structure is None:
            setattr(protein_tensor, "structure", empty_protein_tensor.structure)
        else:
            print("Warning: structure already exists in protein_tensor")
        return protein_tensor
