import torch

from esm.pretrained import ESM3_sm_open_v0
from esm.sdk.api import ESMProtein, GenerationConfig


# This will instantiate the model on your machine.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ESM3_sm_open_v0(
    device=device,
    esm3_structure_encoder_v0_pth_path='models--EvolutionaryScale--esm3-sm-open-v1/snapshots/2feda652c51c57f9797fc1b01f8d6181b38fb488/data/weights/esm3_structure_encoder_v0.pth',
    esm3_structure_decoder_v0_pth_path='models--EvolutionaryScale--esm3-sm-open-v1/snapshots/2feda652c51c57f9797fc1b01f8d6181b38fb488/data/weights/esm3_structure_decoder_v0.pth',
    esm3_function_decoder_v0_pth_path='models--EvolutionaryScale--esm3-sm-open-v1/snapshots/2feda652c51c57f9797fc1b01f8d6181b38fb488/data/weights/esm3_function_decoder_v0.pth',
    keyword_vocabulary_path='models--EvolutionaryScale--esm3-sm-open-v1/snapshots/2feda652c51c57f9797fc1b01f8d6181b38fb488/data/keyword_vocabulary_safety_filtered_58641.txt',
    lsh_hyperplanes_path='models--EvolutionaryScale--esm3-sm-open-v1/snapshots/2feda652c51c57f9797fc1b01f8d6181b38fb488/data/hyperplanes_8bit_58641.npz',
    residue_annotations_csv_path='models--EvolutionaryScale--esm3-sm-open-v1/snapshots/2feda652c51c57f9797fc1b01f8d6181b38fb488/data/uniref90_and_mgnify90_residue_annotations_gt_1k_proteins.csv',
    esm3_sm_open_v1_pth_path='models--EvolutionaryScale--esm3-sm-open-v1/snapshots/2feda652c51c57f9797fc1b01f8d6181b38fb488/data/weights/esm3_sm_open_v1.pth',
)

if device.type != "cpu":
    model = model.to(torch.bfloat16)

# Generate a completion for a partial Carbonic Anhydrase (2vvb)
prompt = "___________________________________________________DQATSLRILNNGHAFNVEFDDSQDKAVLKGGPLDGTYRLIQFHFHWGSLDGQGSEHTVDKKKYAAELHLVHWNTKYGDFGKAVQQPDGLAVLGIFLKVGSAKPGLQKVVDVLDSIKTKGKSADFTNFDPRGLLPESLDYWTYPGSLTTPP___________________________________________________________"
protein = ESMProtein(sequence=prompt)
# Generate the sequence, then the structure. This will iteratively unmask the sequence track.
protein = model.generate(protein, GenerationConfig(track="sequence", num_steps=8, temperature=0.7))
# We can show the predicted structure for the generated sequence.
protein = model.generate(protein, GenerationConfig(track="structure", num_steps=8))
protein.to_pdb("./generation.pdb")
# Then we can do a round trip design by inverse folding the sequence and recomputing the structure
protein.sequence = None
protein = model.generate(protein, GenerationConfig(track="sequence", num_steps=8))
protein.coordinates = None
protein = model.generate(protein, GenerationConfig(track="structure", num_steps=8))
protein.to_pdb("./round_tripped.pdb")
