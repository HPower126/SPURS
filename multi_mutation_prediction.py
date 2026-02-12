from spurs.inference import parse_pdb, get_SPURS_multi_from_hub, parse_pdb_for_mutation
import torch

pdb_name = '7QQA_dimer'
pdb_path = './data/inference_example/' + pdb_name + '.pdb'
chain = ['A']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model, cfg = get_SPURS_multi_from_hub()

pdb = parse_pdb(pdb_path, pdb_name, chain, cfg)

mut_info_dict = {
  'V1': ['K69N','N112L','L168I','T200A','A201K','K224R','T228I','E234H','G241R','E252M','Q263E'],
  'V2':  ['K69N','N112L','L168I','T200A','A201K','T228V','E234H','G241R','E252M','Q263E'],
  'V3':  ['K69N','N112L','L168I','T200A','T228V','E234H','G241R','E252M','Q263E'],
  'V4':  ['N112L', 'L168I', 'T200A', 'T228I', 'E234H', 'G241R'],
  'V5':  ['N112L', 'T200A', 'T228V', 'E234H', 'G241R'],
  'V6':  ['T200A', 'T228V', 'E234H'],
  'V7':  ['T200A', 'T228V', 'G241R'],
  'V8':  ['T228V']
}

for var, mutations in mut_info_dict.items():
  mut_info_list = [mutations]

  mut_ids, append_tensors = parse_pdb_for_mutation(mut_info_list)
  pdb['mut_ids'] = mut_ids
  pdb['append_tensors'] = append_tensors.to(device)
  model.eval()
  with torch.no_grad():
    ddg = model(pdb)[0]
  print(f"{var}: {ddg}") # ddg[i] for mut_info_list[i]

  with open(f"{pdb_name}_variants_spurs_results.txt", "a") as f:
    f.write(f"{var}: {ddg}\n")