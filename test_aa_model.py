import copy
import os
import sys
import torch
import unittest

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'RF2-allatom'))
import rf2aa
from aa_model import AtomizeResidues, Indep, Model, make_indep
import atomize

class AAModelTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.model =  Model(conf=None)
        self.indep = make_indep(pdb="benchmark/input/gaa.pdb")
        self.L = self.indep.xyz.shape[0]
        return super().setUp()

    def test_atomize_residues_Nerm(self):
        # Test N term motif 
        input_str_mask = torch.zeros(self.L).bool()
        input_str_mask[:3] = 1

        self.check_protein_atomization_constraints(self.indep, input_str_mask, terminus=True, discontiguous=False)

    def test_atomize_residues_Cterm(self):
        # Test C term motif 
        # indep modified in place need to recreate (should move to setUp)
        
        input_str_mask = torch.zeros(self.L).bool()
        input_str_mask[-3:] = 1

        self.check_protein_atomization_constraints(self.indep, input_str_mask, terminus=True, discontiguous=False)

    def test_atomize_residues_discontig_motif(self):
        # Test motif in the middle of the protein
        input_str_mask = torch.zeros(self.L).bool()
        input_str_mask[[20,55,88]] = 1

        self.check_protein_atomization_constraints(self.indep, input_str_mask, terminus=False, discontiguous=True)

    def check_protein_atomization_constraints(self, indep, input_str_mask, terminus=False, discontiguous=True):
        """
        checks the output values of indep and masks1d given an input indep and str_mask
        """
        indep.xyz = indep.xyz[:, :14]
        L = indep.xyz.shape[0]
        input_indep = copy.deepcopy(indep)
        input_mask = copy.deepcopy(input_str_mask)

        atom_mask = rf2aa.util.allatom_mask[indep.seq]
        atom_mask[:, 14:] = False # no Hs
        indep, atomizer = atomize.atomize(indep, input_str_mask)
        num_atoms = torch.sum(atom_mask[input_mask])

        # original length, remove the residue nodes that are popped and add new atom nodes for those residues
        new_protein_L = L - torch.sum(input_mask)
        correct_new_L = new_protein_L + num_atoms  
        self.assertEqual(indep.seq.shape[0], correct_new_L.item(), msg="atomized indep has the wrong size seq")
        self.assertEqual(indep.xyz.shape[0], correct_new_L.item(), msg="atomized indep has the wrong size xyz")
        self.assertEqual(indep.idx.shape[0], correct_new_L.item(), msg="atomized indep has the wrong size idx")
        self.assertEqual(indep.bond_feats.shape[:2], (correct_new_L.item(), correct_new_L.item()), msg="atomized indep has the wrong size bond_feats")
        self.assertEqual(indep.atom_frames.shape[0], torch.sum(indep.is_sm), msg="atom frames dimension does not match number of atom nodes")
        self.assertEqual(indep.same_chain.shape[:2], (correct_new_L.item(), correct_new_L.item()), msg="atomized indep has the wrong size same_chain")
        self.assertEqual(torch.sum(indep.is_sm), torch.sum(input_indep.is_sm)+num_atoms, msg="atomized indep has the wrong size number of atom nodes in is_sm")

        # assert some edge cases are handled correctly in same_chain and bond_feats
        residue_atom_bonds = (indep.bond_feats == 6)
        residue_atom_bonds_indices = residue_atom_bonds.nonzero()
        atomized_residue_lengths = torch.sum(atom_mask[input_mask], dim=-1)
        if terminus:
            self.assertEqual(residue_atom_bonds_indices.shape[0], 2, msg="terminal contiguous motifs should only have 1 bond to protein ((i, j), (j, i))")
        elif discontiguous:
            num_residue_atom_bonds = 4*atomized_residue_lengths.shape[0] # 2 bonds for each atomized residue and then two orderings (i, j), (j, i)
            self.assertEqual(residue_atom_bonds_indices.shape[0], num_residue_atom_bonds, msg="nonterminal dicontiguous motif has correct bonds to protein")
        rf2aa.tensor_util.assert_equal(residue_atom_bonds, residue_atom_bonds.T)
        
        if not discontiguous:
            for i in range(atomized_residue_lengths.shape[0]-1):
                # should be one bond between contiguous residues
                # indexes out the bonds between each atomized residue and its following neighbor to assert there is a bond placed
                self.assertEqual(torch.sum(indep.bond_feats[new_protein_L+ torch.sum(atomized_residue_lengths[:i]): \
                                new_protein_L+ torch.sum(atomized_residue_lengths[:i+1]), \
                                new_protein_L+ torch.sum(atomized_residue_lengths[:i+1]): \
                                new_protein_L+ torch.sum(atomized_residue_lengths[:i+2])]), 1, msg="Incorrect bond placement between neighboring residues")
        
        self.assertTrue(torch.all(indep.same_chain == 1), msg="all nodes should be on the same chain because there is no small molecule")

if __name__ == '__main__':
        unittest.main()
