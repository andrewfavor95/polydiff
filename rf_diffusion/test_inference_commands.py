import os
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from .inference import utils as iu
from . import aa_model


CONFIG_DIR = Path(__file__).resolve().parent / "config" / "inference"
DEFAULT_CKPT = "weights/train_session2024-07-08_1720455712_BFF_3.00.pt"


SCENARIOS = [
    (
        "rna_unconditional",
        [
            "diffuser.T=50",
            "inference.num_designs=1",
            "contigmap.contigs=['9']",
            "contigmap.polymer_chains=['rna']",
            "inference.output_prefix=demo_outputs/RNA_uncond_standard_settings",
        ],
    ),
    (
        "multi_polymer_unconditional",
        [
            "diffuser.T=50",
            "inference.num_designs=1",
            "contigmap.contigs=['3 3 3']",
            "contigmap.polymer_chains=['dna','rna','protein']",
            "inference.output_prefix=test_outputs/basic_uncond_test01",
        ],
    ),
    (
        "dna_binder_unconditional",
        [
            "diffuser.T=50",
            "inference.num_designs=1",
            "contigmap.contigs=['2 2 5']",
            "contigmap.polymer_chains=['dna','dna','protein']",
            "inference.output_prefix=demo_outputs/DNA_prot_uncond_standard_settings",
        ],
    ),
    (
        "rna_secondary_structure",
        [
            "diffuser.T=50",
            "inference.num_designs=1",
            "contigmap.contigs=['9']",
            "contigmap.polymer_chains=['rna']",
            "scaffoldguided.target_ss_string=555...333",
        ],
    ),
    (
        "motif_scaffolding_v1",
        [
            "diffuser.T=50",
            "inference.num_designs=1",
            "contigmap.contigs=['1,D8-10,1,B8-10,1 1,B18-20,1,D18-20,1 A1-3,0 C1-3,0']",
            "contigmap.polymer_chains=['dna','dna','protein','protein']",
            "inference.ij_visible=bce-adf",
            "inference.input_pdb=test_data/combo_DBP009_DBP010_DBP011_with_DNA_v2.pdb",
            "inference.output_prefix=demo_outputs/DNA_binders_scaffolding_test1_standard_settings",
        ],
    ),
    (
        "motif_scaffolding_v2",
        [
            "diffuser.T=50",
            "inference.num_designs=1",
            "contigmap.contigs=['1,D8-10,1,B8-10,1 1,B18-20,1,D18-20,1 A1-3,3,C1-3,0']",
            "contigmap.polymer_chains=['dna','dna','protein']",
            "scaffoldguided.target_ss_pairs=['A1-24,B1-24']",
            "inference.ij_visible=bce-adf",
            "inference.input_pdb=test_data/combo_DBP009_DBP010_DBP011_with_DNA_v2.pdb",
            "inference.output_prefix=demo_outputs/DNA_binders_scaffolding_test2_standard_settings",
        ],
    ),
    (
        "dna_pair_specification",
        [
            "diffuser.T=50",
            "inference.num_designs=1",
            "contigmap.contigs=['6 6 6 6']",
            "contigmap.polymer_chains=['dna','dna','dna','dna']",
            "scaffoldguided.target_ss_pairs=['A1-2,B1-2','A3-4,C3-4','A5-6,D5-6','B3-4,D3-4','B5-6,C5-6','C1-2,D1-2']",
            "inference.symmetry=d2",
            "inference.output_prefix=demo_outputs/DNA_origami_standard_settings",
        ],
    ),
]


@pytest.mark.parametrize("name, overrides", SCENARIOS)
def test_multi_polymer_configs(name, overrides):
    os.environ.setdefault("RFDPOLY_CKPT_PATH", DEFAULT_CKPT)
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        cfg = compose(config_name="multi_polymer", overrides=overrides)

    assert cfg.inference.num_designs == 1
    assert cfg.diffuser.T == 50
    assert len(cfg.contigmap.contigs) >= 1


def test_process_target_resolution():
    pdb_rel_path = "test_data/DBP035.pdb"
    result = iu.process_target(pdb_rel_path, center=False)
    assert "xyz_27" in result
    assert result["xyz_27"].shape[-1] == 3

    with pytest.raises(FileNotFoundError):
        iu.process_target("non_existent/example.pdb")


def test_make_indep_path_resolution():
    indep = aa_model.make_indep("test_data/DBP035.pdb")
    assert indep.xyz.shape[-1] == 3
    with pytest.raises(FileNotFoundError):
        aa_model.make_indep("missing_dir/missing.pdb")
