"""
All featurization methods used in the benchmark.

Each featurizer takes a list of SMILES or protein sequences and returns
a numpy array of shape (n_samples, feature_dim).

Naming convention used in diary:
  Ligand:  ecfp2_1024, ecfp4_1024, ecfp6_1024, rdkit_200, maccs_167,
           smiles_char_onehot, chemberta_600
  Protein: aac_20, dipeptide_400, kmer3_8000, esm2_320, esm2_480,
           protbert_1024
"""

import numpy as np
import pandas as pd
from typing import List

import torch

# ─────────────────────────────────────────────────────────────────────────────
# Ligand featurizers
# ─────────────────────────────────────────────────────────────────────────────

def ecfp(smiles_list: List[str], radius: int = 2, nbits: int = 1024) -> np.ndarray:
    """Morgan / ECFP fingerprint."""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            fps.append(np.zeros(nbits, dtype=np.float32))
        else:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
            fps.append(np.array(fp, dtype=np.float32))
    return np.stack(fps)


def maccs_keys(smiles_list: List[str]) -> np.ndarray:
    """MACCS 167-bit keys."""
    from rdkit import Chem
    from rdkit.Chem import MACCSkeys
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            fps.append(np.zeros(167, dtype=np.float32))
        else:
            fp = MACCSkeys.GenMACCSKeys(mol)
            fps.append(np.array(fp, dtype=np.float32))
    return np.stack(fps)


def rdkit_descriptors(smiles_list: List[str]) -> np.ndarray:
    """200 RDKit 2D physicochemical descriptors (normalized)."""
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    from rdkit.ML.Descriptors import MoleculeDescriptors
    names = [d[0] for d in Descriptors._descList[:200]]
    calc  = MoleculeDescriptors.MolecularDescriptorCalculator(names)
    rows  = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            rows.append([0.0] * len(names))
        else:
            try:
                rows.append(list(calc.CalcDescriptors(mol)))
            except Exception:
                rows.append([0.0] * len(names))
    arr = np.array(rows, dtype=np.float32)
    # Replace NaN/inf with 0
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


def smiles_char_onehot(smiles_list: List[str], max_len: int = 100) -> np.ndarray:
    """
    Character-level one-hot encoding of SMILES strings.
    Vocabulary: printable ASCII subset commonly found in SMILES.
    Output shape: (n, max_len * vocab_size) — flattened.
    """
    # Explicit fixed vocab — covers all SMILES characters in BindingDB
    # 39 chars: organic atoms, aromatic atoms, digits, bonds, brackets, misc
    vocab = ['C','N','O','S','P','F','I','B','r','l','c','n','o','s',
             '1','2','3','4','5','6','7','8','9','0',
             '(',')','+','-','=','#','@','\\','/','.',',','[',']','%','H']
    char2idx = {c: i for i, c in enumerate(vocab)}
    vocab_size = len(vocab)
    out = np.zeros((len(smiles_list), max_len * vocab_size), dtype=np.float32)
    for row_idx, smi in enumerate(smiles_list):
        for col_idx, ch in enumerate(smi[:max_len]):
            if ch in char2idx:
                out[row_idx, col_idx * vocab_size + char2idx[ch]] = 1.0
    return out


# def chemberta_embeddings(smiles_list: List[str], batch_size: int = 64) -> np.ndarray:
#     """
#     Mean-pooled embeddings from ChemBERTa-2 (77M params).
#     Requires: transformers, torch
#     Output dim: 600
#     """
#     from transformers import AutoTokenizer, AutoModel
#     import torch
#     model_name = "seyonec/ChemBERTa-zinc-base-v1"
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model     = AutoModel.from_pretrained(model_name)
#     device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model     = model.to(device).eval()

#     all_embs = []
#     for i in range(0, len(smiles_list), batch_size):
#         batch = smiles_list[i : i + batch_size]
#         enc   = tokenizer(batch, padding=True, truncation=True,
#                           max_length=128, return_tensors="pt")
#         enc   = {k: v.to(device) for k, v in enc.items()}
#         with torch.no_grad():
#             out = model(**enc)
#         # Mean pooling over token dimension
#         mask     = enc["attention_mask"].unsqueeze(-1).float()
#         emb      = (out.last_hidden_state * mask).sum(1) / mask.sum(1)
#         all_embs.append(emb.cpu().numpy())
#         if (i // batch_size) % 10 == 0:
#             print(f"  [chemberta] {i+len(batch)}/{len(smiles_list)}")

#     del model
#     torch.cuda.empty_cache()
#     return np.concatenate(all_embs, axis=0)


# def chemberta_embeddings(smiles_list: List[str], model_name: str = "seyonec/ChemBERTa-zinc-base-v1",
#                          batch_size: int = 64) -> np.ndarray:
#     # model_name can be:
#     #   "DeepChem/ChemBERTa-5M-MLM"
#     #   "DeepChem/ChemBERTa-77M-MLM"
#     #   "DeepChem/ChemBERTa-100M-MLM"
#     #   "seyonec/ChemBERTa-zinc-base-v1" (original)
#     from transformers import AutoTokenizer, AutoModel
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = AutoModel.from_pretrained(model_name)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = model.to(device).eval()
#     all_embs = []
#     for i in range(0, len(smiles_list), batch_size):
#         batch = smiles_list[i:i+batch_size]
#         enc = tokenizer(batch, padding=True, truncation=True,
#                         max_length=128, return_tensors="pt")
#         enc = {k: v.to(device) for k, v in enc.items()}
#         with torch.no_grad():
#             out = model(**enc)
#         mask = enc["attention_mask"].unsqueeze(-1).float()
#         emb = (out.last_hidden_state * mask).sum(1) / mask.sum(1)
#         all_embs.append(emb.cpu().numpy())
#     return np.concatenate(all_embs, axis=0)

# def molformer_embeddings(smiles_list: List[str], model_name: str = "DeepChem/MoLFormer-c3-100M",
#                          batch_size: int = 32) -> np.ndarray:
#     # models: "DeepChem/MoLFormer-c3-100M", "DeepChem/MoLFormer-c3-550M"
#     from transformers import AutoTokenizer, AutoModel
#     tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
#     model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = model.to(device).eval()
#     # MoLFormer uses special tokens; default max_length 512
#     all_embs = []
#     for i in range(0, len(smiles_list), batch_size):
#         batch = smiles_list[i:i+batch_size]
#         enc = tokenizer(batch, padding=True, truncation=True,
#                         max_length=512, return_tensors="pt")
#         enc = {k: v.to(device) for k, v in enc.items()}
#         with torch.no_grad():
#             out = model(**enc)
#         # Use CLS token embedding (or mean pool)
#         emb = out.last_hidden_state[:, 0, :].cpu().numpy()
#         all_embs.append(emb)
#     return np.concatenate(all_embs, axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# Protein featurizers
# ─────────────────────────────────────────────────────────────────────────────

_AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")

def amino_acid_composition(sequences: List[str]) -> np.ndarray:
    """20-dim amino acid frequency vector (AAC)."""
    aa2idx = {aa: i for i, aa in enumerate(_AMINO_ACIDS)}
    out = np.zeros((len(sequences), 20), dtype=np.float32)
    for row_idx, seq in enumerate(sequences):
        seq = seq.upper()
        total = max(len(seq), 1)
        for ch in seq:
            if ch in aa2idx:
                out[row_idx, aa2idx[ch]] += 1
        out[row_idx] /= total
    return out


def dipeptide_composition(sequences: List[str]) -> np.ndarray:
    """400-dim dipeptide frequency vector (DPC)."""
    pairs = [a + b for a in _AMINO_ACIDS for b in _AMINO_ACIDS]
    pair2idx = {p: i for i, p in enumerate(pairs)}
    out = np.zeros((len(sequences), 400), dtype=np.float32)
    for row_idx, seq in enumerate(sequences):
        seq   = seq.upper()
        total = max(len(seq) - 1, 1)
        for j in range(len(seq) - 1):
            dp = seq[j:j+2]
            if dp in pair2idx:
                out[row_idx, pair2idx[dp]] += 1
        out[row_idx] /= total
    return out


def kmer_frequency(sequences: List[str], k: int = 3, max_features: int = 8000) -> np.ndarray:
    """
    k-mer frequency bag-of-words for protein sequences.
    Uses sklearn's HashingVectorizer for memory efficiency.
    """
    from sklearn.feature_extraction.text import HashingVectorizer
    def _kmerize(seq):
        seq = seq.upper()
        return " ".join(seq[i:i+k] for i in range(len(seq) - k + 1))

    vectorizer = HashingVectorizer(n_features=max_features, norm="l2", alternate_sign=False)
    corpus     = [_kmerize(s) for s in sequences]
    return vectorizer.transform(corpus).toarray().astype(np.float32)



def prot_electra_embeddings(sequences: List[str], batch_size: int = 8,
                             max_len: int = 512) -> np.ndarray:
    """
    Mean-pooled embeddings from ProtElectra (RTD-based protein LM).
    Model: Rostlab/prot_electra_generator_bfd  (hidden_dim = 256)
    Sequences must have spaces between amino acids (ESM-style not needed;
    ProtTrans models expect space-separated single-letter codes).
    Requires: transformers, torch
    Output dim: 256
    """
    from transformers import AutoTokenizer, AutoModel
    import torch

    model_name = "Rostlab/prot_electra_generator_bfd"
    tokenizer  = AutoTokenizer.from_pretrained(model_name, do_lower_case=False)
    model      = AutoModel.from_pretrained(model_name)
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model      = model.to(device).eval()

    # ProtTrans convention: space-separate amino acids, truncate long sequences
    def _prep(seq):
        seq = " ".join(list(seq.upper().replace("-", "L")))
        return seq[:max_len * 2]   # each AA becomes "X " = 2 chars, so 2×max_len

    all_embs = []
    for i in range(0, len(sequences), batch_size):
        batch = [_prep(s) for s in sequences[i : i + batch_size]]
        enc   = tokenizer(batch, return_tensors="pt", padding=True,
                          truncation=True, max_length=max_len)
        enc   = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc)
        mask = enc["attention_mask"].unsqueeze(-1).float()
        emb  = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1)
        all_embs.append(emb.cpu().numpy())
        if (i // batch_size) % 5 == 0:
            print(f"  [prot_electra] {i+len(batch)}/{len(sequences)}")

    del model
    torch.cuda.empty_cache()
    return np.concatenate(all_embs, axis=0)


def smiles_distmat(smiles_list: List[str], max_atoms: int = 100) -> np.ndarray:
    """
    Topological distance matrix flattened to a 1D vector for flat-feature models.
    Output shape: (N, max_atoms * max_atoms)  — use with distmat_cnn directly
    for 2D models (reshape inside model), or as a flat feature with MLP.
    """
    from models.distmat_cnn import smiles_list_to_distmat
    mats = smiles_list_to_distmat(smiles_list, max_atoms=max_atoms)
    return mats.reshape(len(smiles_list), -1)



# ----- ESM-2 larger variants (150M, 650M) -----
def esm2_embeddings(sequences: List[str], model_size: str = "8M",
                    batch_size: int = 16, max_len: int = 1200) -> np.ndarray:
    model_map = {
        "8M":   ("facebook/esm2_t6_8M_UR50D",   320),
        "35M":  ("facebook/esm2_t12_35M_UR50D", 480),
        "150M": ("facebook/esm2_t30_150M_UR50D", 640),
        "650M": ("facebook/esm2_t33_650M_UR50D", 1280),
    }
    if model_size not in model_map:
        raise ValueError(f"Unknown ESM-2 size: {model_size}")
    model_name, dim = model_map[model_size]
    from transformers import AutoTokenizer, EsmModel
    import torch
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = EsmModel.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    seqs = [s[:max_len] for s in sequences]
    all_embs = []
    for i in range(0, len(seqs), batch_size):
        if i % (batch_size * 10) == 0:
            print(f"  [esm2_{model_size}] Processing batch {i//batch_size + 1}/{(len(seqs)-1)//batch_size + 1}")

        batch = seqs[i:i+batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True,
                        truncation=True, max_length=max_len)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc)
        mask = enc["attention_mask"].unsqueeze(-1).float()
        emb = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1)
        all_embs.append(emb.cpu().numpy())
    return np.concatenate(all_embs, axis=0)


def esmc_embeddings(sequences: List[str], model_name: str = "esmc_300m",
                    batch_size: int = 8, max_len: int = 512) -> np.ndarray:
    """
    Generate ESM-C protein embeddings using the official esm library.
    model_name can be "esmc_300m" or "esmc_600m".
    """
    from esm.models.esmc import ESMC
    from esm.sdk.api import ESMProtein, LogitsConfig
    import torch
    import numpy as np

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[esmc] Loading model {model_name} on {device}...")
    model = ESMC.from_pretrained(model_name).to(device).eval()
    print("[esmc] Model loaded.")

    all_embs = []
    total = len(sequences)
    # Process in batches to avoid OOM
    for i in range(0, total, batch_size):
        batch = sequences[i:i+batch_size]
        print(f"[esmc] Batch {i//batch_size + 1}/{(total-1)//batch_size + 1}")
        batch_embs = []
        for seq in batch:
            # Truncate long sequences
            if len(seq) > max_len:
                seq = seq[:max_len]
            protein = ESMProtein(sequence=seq)
            # Encode to tensor
            protein_tensor = model.encode(protein).to(device)
            # Get embeddings
            with torch.no_grad():
                logits_output = model.logits(
                    protein_tensor,
                    LogitsConfig(sequence=False, return_embeddings=True)
                )
                # embeddings shape: (1, seq_len, hidden_dim)
                # Mean pool over sequence length
                emb = logits_output.embeddings.squeeze(0).mean(dim=0).cpu().numpy()
            batch_embs.append(emb)
        all_embs.extend(batch_embs)

    return np.array(all_embs)

def chemberta_embeddings(smiles_list: List[str], model_name: str = "seyonec/ChemBERTa-zinc-base-v1",
                         batch_size: int = 64) -> np.ndarray:
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    all_embs = []
    for i in range(0, len(smiles_list), batch_size):
        batch = smiles_list[i:i+batch_size]
        enc = tokenizer(batch, padding=True, truncation=True,
                        max_length=128, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc)
        mask = enc["attention_mask"].unsqueeze(-1).float()
        emb = (out.last_hidden_state * mask).sum(1) / mask.sum(1)
        all_embs.append(emb.cpu().numpy())
    return np.concatenate(all_embs, axis=0)



# ─────────────────────────────────────────────────────────────────────────────
# Sequence featurizers (for BiCA v2 — true sequence inputs)
# ─────────────────────────────────────────────────────────────────────────────

def chemberta_per_token_padded(
    smiles_list: List[str],
    model_name: str = "seyonec/ChemBERTa-zinc-base-v1",
    batch_size: int = 64,
    max_len: int = 128,
) -> tuple:
    """
    Per-token ChemBERTa embeddings with padding.

    Strips [CLS] (position 0) and [SEP] / padding tokens, keeping only the
    real SMILES subword tokens.  This gives variable-length sequences of
    contextualised chemical token embeddings — richer than RDKit atom features
    because ChemBERTa was pretrained on 77M SMILES strings.

    Returns:
        embeddings: torch.FloatTensor  (N, max_tok_len, hidden_dim)   padded
        mask:       torch.LongTensor   (N, max_tok_len)                1=real 0=pad
    where max_tok_len = min(max(actual token lengths), max_len - 2).

    ChemBERTa hidden_dim = 600 for seyonec/ChemBERTa-zinc-base-v1.
    """
    from transformers import AutoTokenizer, AutoModel

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    cb_model  = AutoModel.from_pretrained(model_name)
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cb_model  = cb_model.to(device).eval()

    hidden_dim = cb_model.config.hidden_size   # 600

    per_sample = []   # list of (L_i, hidden_dim) tensors
    for i in range(0, len(smiles_list), batch_size):
        if i % (batch_size * 10) == 0:
            print(f"  [chemberta_seq] {i}/{len(smiles_list)}")
        batch = smiles_list[i : i + batch_size]
        enc = tokenizer(batch, padding=True, truncation=True,
                        max_length=max_len, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = cb_model(**enc)
        last_hidden  = out.last_hidden_state   # (B, L_tok, H)
        attn_mask    = enc["attention_mask"]   # (B, L_tok) 1=real

        for j in range(len(batch)):
            real_len = attn_mask[j].sum().item()   # includes CLS + SEP
            # Strip CLS (index 0) and SEP (last real token)
            tok_emb = last_hidden[j, 1:real_len - 1]   # (L_real_tokens, H)
            per_sample.append(tok_emb.cpu())

    del cb_model
    torch.cuda.empty_cache()

    lengths  = [e.shape[0] for e in per_sample]
    max_L    = max(lengths)
    n        = len(per_sample)
    padded   = torch.zeros(n, max_L, hidden_dim, dtype=torch.float32)
    mask_t   = torch.zeros(n, max_L, dtype=torch.long)
    for i, emb in enumerate(per_sample):
        L = emb.shape[0]
        padded[i, :L, :] = emb
        mask_t[i, :L]    = 1

    return padded, mask_t   # (N, max_L, hidden_dim), (N, max_L)

def esm2_per_residue_padded(
    sequences: List[str],
    model_size: str = "35M",
    batch_size: int = 8,
    max_len: int = 1200,
) -> tuple:
    """
    Per-residue ESM-2 embeddings with padding.

    Returns:
        embeddings: torch.FloatTensor  (N, max_seq_len, dim)  — padded
        mask:       torch.LongTensor   (N, max_seq_len)        — 1=real, 0=pad
    where max_seq_len = min(max(actual lengths), max_len).

    ESM-2 model_size → hidden_dim:
        "8M"  → 320,  "35M" → 480,  "150M" → 640,  "650M" → 1280
    """
    model_map = {
        "8M":   ("facebook/esm2_t6_8M_UR50D",    320),
        "35M":  ("facebook/esm2_t12_35M_UR50D",  480),
        "150M": ("facebook/esm2_t30_150M_UR50D", 640),
        "650M": ("facebook/esm2_t33_650M_UR50D", 1280),
    }
    if model_size not in model_map:
        raise ValueError(f"Unknown ESM-2 size: {model_size}")
    model_name, dim = model_map[model_size]

    from transformers import AutoTokenizer, EsmModel
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    esm_model = EsmModel.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    esm_model = esm_model.to(device).eval()

    seqs = [s[:max_len] for s in sequences]
    # Collect per-sample tensors (variable length, no special tokens)
    per_sample = []   # list of (L_i, dim) arrays
    for i in range(0, len(seqs), batch_size):
        if i % (batch_size * 5) == 0:
            print(f"  [esm2_{model_size}_seq] {i}/{len(seqs)}")
        batch = seqs[i : i + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True,
                        truncation=True, max_length=max_len)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = esm_model(**enc)
        last_hidden = out.last_hidden_state   # (B, L_tok, dim)
        attn_mask   = enc["attention_mask"]   # (B, L_tok) — 1=real, 0=pad

        for j in range(len(batch)):
            # Exclude special tokens [CLS] (position 0) and [EOS]:
            # keep positions 1 .. seq_len (where attn_mask[j,pos]==1 and pos>0)
            tok_mask = attn_mask[j]                     # (L_tok,)
            real_len = tok_mask.sum().item()             # includes CLS + EOS
            # Strip CLS (index 0) and EOS (last real token)
            seq_emb  = last_hidden[j, 1:real_len - 1]   # (L_aa, dim)
            per_sample.append(seq_emb.cpu())

    del esm_model
    torch.cuda.empty_cache()

    # Pad to longest sequence in the batch, capped at max_len
    # (max_len already truncated raw strings; cap here prevents OOM from
    #  rare edge cases where tokenization produces slightly longer outputs)
    lengths   = [e.shape[0] for e in per_sample]
    max_L     = min(max(lengths), max_len)
    n         = len(per_sample)
    padded    = torch.zeros(n, max_L, dim, dtype=torch.float32)
    mask_t    = torch.zeros(n, max_L, dtype=torch.long)
    for i, emb in enumerate(per_sample):
        L = min(emb.shape[0], max_L)
        padded[i, :L, :] = emb[:L]
        mask_t[i, :L]    = 1

    return padded, mask_t   # (N, max_L, dim), (N, max_L)


def mol_atom_features_padded(
    smiles_list: List[str],
    max_atoms: int = 100,
) -> tuple:
    """
    Per-atom RDKit feature vectors with padding.

    Feature vector per atom (78-dim, same as GNN atom featurizer):
        one-hot atomic num (44), degree (6), H count (5), formal charge (1),
        radical electrons (1), hybridization (5), aromaticity (1),
        ring membership (1), chirality (4)  → 44+6+5+1+1+5+1+1+4 = 68 + extra = 78

    Returns:
        features: torch.FloatTensor  (N, max_atoms, 78)
        mask:     torch.LongTensor   (N, max_atoms)  — 1=real atom, 0=pad
    """
    from rdkit import Chem
    from rdkit.Chem import rdchem

    ATOM_LIST = [1, 5, 6, 7, 8, 9, 15, 16, 17, 35, 53,
                 # common metals / rare atoms → bucket as 'other'
                 0]   # 0 = other
    DEGREE_LIST   = [0, 1, 2, 3, 4, 5]
    HCOUNT_LIST   = [0, 1, 2, 3, 4]
    HYBRID_LIST   = [
        rdchem.HybridizationType.SP,
        rdchem.HybridizationType.SP2,
        rdchem.HybridizationType.SP3,
        rdchem.HybridizationType.SP3D,
        rdchem.HybridizationType.SP3D2,
    ]
    CHIRALITY_LIST = [
        rdchem.ChiralType.CHI_UNSPECIFIED,
        rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        rdchem.ChiralType.CHI_OTHER,
    ]

    def _one_hot(val, choices):
        vec = [0] * len(choices)
        try:
            vec[choices.index(val)] = 1
        except ValueError:
            vec[-1] = 1   # bucket unknown → last slot
        return vec

    def _atom_features(atom):
        feats = []
        feats += _one_hot(atom.GetAtomicNum(), ATOM_LIST)           # 12
        feats += _one_hot(atom.GetDegree(),    DEGREE_LIST)          # 6
        feats += _one_hot(atom.GetTotalNumHs(), HCOUNT_LIST)         # 5
        feats += [float(atom.GetFormalCharge())]                     # 1
        feats += [float(atom.GetNumRadicalElectrons())]              # 1
        feats += _one_hot(atom.GetHybridization(), HYBRID_LIST)      # 5
        feats += [float(atom.GetIsAromatic())]                       # 1
        feats += [float(atom.IsInRing())]                            # 1
        feats += _one_hot(atom.GetChiralTag(), CHIRALITY_LIST)       # 4
        # pad to 78 if needed
        while len(feats) < 78:
            feats.append(0.0)
        return feats[:78]

    n       = len(smiles_list)
    padded  = torch.zeros(n, max_atoms, 78, dtype=torch.float32)
    mask_t  = torch.zeros(n, max_atoms, dtype=torch.long)

    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        atoms = mol.GetAtoms()
        for j, atom in enumerate(atoms):
            if j >= max_atoms:
                break
            padded[i, j, :] = torch.tensor(_atom_features(atom), dtype=torch.float32)
            mask_t[i, j]    = 1

    return padded, mask_t   # (N, max_atoms, 78), (N, max_atoms)


# ─────────────────────────────────────────────────────────────────────────────
# Fusion helpers
# ─────────────────────────────────────────────────────────────────────────────

def concat(lig_features: np.ndarray, prot_features: np.ndarray) -> np.ndarray:
    """Simple concatenation of ligand and protein feature vectors."""
    return np.concatenate([lig_features, prot_features], axis=1)
