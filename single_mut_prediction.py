import csv
import torch
from spurs.inference import get_SPURS, parse_pdb, get_SPURS_from_hub

model, cfg = get_SPURS_from_hub()
pdb_name = '7QQA_dimer_chainB_renamed'
pdb_path = './data/inference_example/' + pdb_name + '.pdb'
chain = 'A'
pdb = parse_pdb(pdb_path, pdb_name, chain, cfg)
wt_seq = 'TRKVAIYGKGGIGKSTTTQNTAAALAYFHDKKVFIHGCDPKADSTRLILGGKPQETLMDMLRDKGAEKITNDDVIKKGFLDIQCVESGGPEPGVGCAGRGVITAIDLMEENGAYTDDLDFVFFDVLGDVVCGGFAMPIRDGKAQEVYIVASGEMMAIYAANNICKGLVKYAKQSGVRLGGIICNSRKVDGEREFLEEFTAAIGTKMIHFVPRDNIVQKAEFNKKTVTEFAPEENQAKEYGELARKIIENDEFVIPKPLTMDQLEDMVVKYGIATRKVAIYGKGGIGKSTTTQNTAAALAYFHDKKVFIHGCDPKADSTRLILGGKPQETLMDMLRDKGAEKITNDDVIKKGFLDIQCVESGGPEPGVGCAGRGVITAIDLMEENGAYTDDLDFVFFDVLGDVVCGGFAMPIRDGKAQEVYIVASGEMMAIYAANNICKGLVKYAKQSGVRLGGIICNSRKVDGEREFLEEFTAAIGTKMIHFVPRDNIVQKAEFNKKTVTEFAPEENQAKEYGELARKIIENDEFVIPKPLTMDQLEDMVVKYG'
start_pos = 2 # residue number of first amino acid in pdb file

with torch.no_grad():
  result = model(pdb,return_logist=True)

amino_acids = ['A','C','D','E','F','G','H','I','K','L',
               'M','N','P','Q','R','S','T','V','W','Y']

print(f"Result shape: {result.shape}")  # should be (L, 20)

with open(f"{pdb_name}_spurs_ddG.csv", "w", newline="") as outfile:
    writer = csv.writer(outfile)
    writer.writerow(["Position", "WT_AA", "Mut_AA", "ddG_kcal_per_mol"])

    for i in range(result.shape[0]):
        wt_aa = wt_seq[i]  # the actual letter
        wt_idx = amino_acids.index(wt_aa)
        for m_idx, m_aa in enumerate(amino_acids):
            if m_aa == wt_aa:
                continue
            ddG = result[i, m_idx].item() - result[i, wt_idx].item()
            writer.writerow([i + start_pos, wt_aa, m_aa, ddG])