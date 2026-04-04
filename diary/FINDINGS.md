# Binding Affinity Benchmark — Findings & Analysis

**Dataset:** BindingDB_filtered (BALM-benchmark)  
**Split:** Bemis-Murcko scaffold split, seed=42  
**Train / Val / Test:** 17,296 / 2,466 / 4,938  
**Task:** pKd regression — metrics: RMSE ↓, Pearson R ↑, Spearman R ↑  
**Total experiments:** 313 (duplicates deduplicated, last run kept)  
**Date:** 2026-03-30

---
## 1. Overall Leaderboard (Top 20, sorted by Test RMSE)

                     experiment_id model_family    ligand_repr     protein_repr  test_rmse  test_pearson_r  test_spearman_r  train_time_sec  n_params
                      rf_ecfp4_aac         tree     ecfp4_1024           aac_20     1.0065          0.7467           0.6737            23.4       NaN
                rf_ecfp4_dipeptide         tree     ecfp4_1024    dipeptide_400     1.0192          0.7387           0.6636            62.1       NaN
        xgb_chemberta_5M_esm2_650M         tree   chemberta_5M        esm2_650M     1.0434          0.7242           0.6478            12.1       NaN
     xgb_chemberta_5M_esm2_35M_480         tree   chemberta_5M     esm2_35M_480     1.0473          0.7220           0.6517             6.4       NaN
                 xgb_ecfp4_esm2_8M         tree     ecfp4_1024      esm2_8M_320     1.0483          0.7207           0.6309             4.3       NaN
        xgb_ecfp4_esm2_8M__seed456         tree     ecfp4_1024      esm2_8M_320     1.0510          0.6913           0.6283             4.6       NaN
             rf_ecfp4_aac__seed456         tree     ecfp4_1024           aac_20     1.0516          0.6941           0.6276            46.9       NaN
       xgb_chemberta_600_esm2_650M         tree  chemberta_600        esm2_650M     1.0522          0.7180           0.6390            15.2       NaN
                     xgb_ecfp4_aac         tree     ecfp4_1024           aac_20     1.0523          0.7184           0.6399             2.5       NaN
                    lgbm_ecfp4_aac         tree     ecfp4_1024           aac_20     1.0528          0.7197           0.6395             5.8       NaN
      xgb_chemberta_5M_esm2_8M_320         tree   chemberta_5M      esm2_8M_320     1.0554          0.7169           0.6469             5.7       NaN
 xgb_chemberta_5M_prot_electra_256         tree   chemberta_5M prot_electra_256     1.0566          0.7152           0.6446            11.7       NaN
    xgb_chemberta_600_esm2_35M_480         tree  chemberta_600     esm2_35M_480     1.0577          0.7147           0.6418             9.0       NaN
        xgb_chemberta_prot_electra         tree  chemberta_600 prot_electra_256     1.0580          0.7142           0.6380            14.6       NaN
xgb_chemberta_600_prot_electra_256         tree  chemberta_600 prot_electra_256     1.0580          0.7142           0.6380            13.9       NaN
        xgb_chemberta_5M_esm2_150M         tree   chemberta_5M        esm2_150M     1.0587          0.7150           0.6397             7.8       NaN
        xgb_chemberta_5M_esmc_300M         tree   chemberta_5M        esmc_300M     1.0599          0.7133           0.6398             9.0       NaN
       xgb_chemberta_600_esmc_300M         tree  chemberta_600        esmc_300M     1.0627          0.7113           0.6341            11.4       NaN
       xgb_chemberta_77M_esm2_650M         tree  chemberta_77M        esm2_650M     1.0661          0.7098           0.6384            11.8       NaN
      xgb_chemberta_100M_esm2_650M         tree chemberta_100M        esm2_650M     1.0662          0.7094           0.6238            14.1       NaN


---
## 2. Best Experiment Per Model Family

                group                                  experiment  test_rmse  pearson_r  spearman_r  train_sec
   baselines (linear)             ridge_chemberta_5M_esm2_35M_480     1.2544     0.5951      0.5255        0.2
                trees                                rf_ecfp4_aac     1.0065     0.7467      0.6737       23.4
           mlp (flat)                mlp_chemberta_100M_esm2_650M     1.1007     0.7027      0.6225       36.1
pretrained (MLP/tree)               xgb_chemberta_5M_esm2_35M_480     1.0473     0.7220      0.6517        6.4
      cnn (1D SMILES)                 cnn_smiles_onehot_esm2_150M     1.3243     0.5213      0.4302       42.8
     distmat_cnn (2D)        distmat_cnn_distmat_100_esm2_35M_480     1.1092     0.6973      0.6307     1330.4
                 lstm       lstm_smiles_bpe_1000_protein_bpe_1000     1.1456     0.6595      0.5812     1082.8
      transformer_seq    transformer_seq_smiles_atom_protein_char     1.2096     0.6061      0.5138      647.4
   transformer (flat) transformer_chemberta_100M_prot_electra_256     1.1194     0.6791      0.6019       77.6
                  gcn                  gcn_mol_graph_esm2_35M_480     1.2024     0.6156      0.5504       73.9
                  gat                     gat_mol_graph_esm2_650M     1.1939     0.6434      0.5840      151.1
                 bica                 bica_chemberta_5M_esmc_300M     1.1020     0.7020      0.6310       44.5


---
## 3. Structural Ligand Representation Comparison

Comparing models that encode molecular structure beyond fingerprints:

                           experiment_id model_family   ligand_repr     protein_repr  test_rmse  test_pearson_r  test_spearman_r  train_time_sec  n_params
    distmat_cnn_distmat_100_esm2_35M_480  distmat_cnn   distmat_100     esm2_35M_480     1.1092          0.6973           0.6307          1330.4 1520865.0
                     distmat_cnn_esm2_8M  distmat_cnn   distmat_100      esm2_8M_320     1.1147          0.6902           0.6195          1235.2 1438945.0
       distmat_cnn_distmat_100_esm2_150M  distmat_cnn   distmat_100        esm2_150M     1.1265          0.6768           0.6053           926.5 1602785.0
       distmat_cnn_distmat_100_esmc_300M  distmat_cnn   distmat_100        esmc_300M     1.1423          0.6731           0.6119           885.1 1766625.0
                  distmat_cnn_kmer3_8000  distmat_cnn   distmat_100       kmer3_8000     1.1448          0.6737           0.6037           761.5 5371105.0
       distmat_cnn_distmat_100_esm2_650M  distmat_cnn   distmat_100        esm2_650M     1.1548          0.6541           0.5810           602.6 1930465.0
distmat_cnn_distmat_100_prot_electra_256  distmat_cnn   distmat_100 prot_electra_256     1.1590          0.6645           0.5890           863.9 1930465.0
            distmat_cnn_prot_electra_256  distmat_cnn   distmat_100 prot_electra_256     1.1604          0.6626           0.5959          1088.9 1930465.0
     distmat_cnn_distmat_100_esm2_8M_320  distmat_cnn   distmat_100      esm2_8M_320     1.1873          0.6496           0.5787           747.1 1438945.0
                 gat_mol_graph_esm2_650M          gat     mol_graph        esm2_650M     1.1939          0.6434           0.5840           151.1  897665.0
                 gat_ecfp_esm2_8M_ranked          gat     mol_graph      esm2_8M_320     1.1963          0.6434           0.5830           208.9  406145.0
              gcn_mol_graph_esm2_35M_480          gcn     mol_graph     esm2_35M_480     1.2024          0.6156           0.5504            73.9  487297.0
                 gcn_mol_graph_esmc_300M          gcn     mol_graph        esmc_300M     1.2073          0.6276           0.5652            88.7  733057.0
                 gcn_mol_graph_esm2_150M          gcn     mol_graph        esm2_150M     1.2086          0.6340           0.5920           129.9  569217.0
                 gat_mol_graph_esmc_300M          gat     mol_graph        esmc_300M     1.2088          0.6326           0.5705            93.9  733825.0
                gcn_mol_graph_kmer3_8000          gcn     mol_graph       kmer3_8000     1.2141          0.6201           0.5558           138.2 4337537.0
              gat_mol_graph_esm2_35M_480          gat     mol_graph     esm2_35M_480     1.2148          0.6349           0.5841           151.7  488065.0
               gcn_mol_graph_esm2_8M_320          gcn     mol_graph      esm2_8M_320     1.2156          0.6360           0.5908           141.7  405377.0
                        gcn_ecfp_esm2_8M          gcn     mol_graph      esm2_8M_320     1.2158          0.6400           0.5964            98.6  405377.0
                gat_mol_graph_kmer3_8000          gat     mol_graph       kmer3_8000     1.2161          0.6407           0.5842           167.5 4338305.0
                  gcn_ecfp_esm2_8M_recon          gcn     mol_graph      esm2_8M_320     1.2187          0.6132           0.5610            91.0  431951.0
          gcn_mol_graph_prot_electra_256          gcn     mol_graph prot_electra_256     1.2255          0.6149           0.5608           178.8  896897.0
          gat_mol_graph_prot_electra_256          gat     mol_graph prot_electra_256     1.2274          0.6224           0.5582           153.4  897665.0
                 gat_mol_graph_esm2_150M          gat     mol_graph        esm2_150M     1.2285          0.6233           0.5664           165.5  569985.0
                 gcn_mol_graph_esm2_650M          gcn     mol_graph        esm2_650M     1.2329          0.6006           0.5583           107.4  896897.0
                        gat_ecfp_esm2_8M          gat     mol_graph      esm2_8M_320     1.2551          0.5978           0.5623            93.8  406145.0
                  gat_ecfp_esm2_8M_recon          gat     mol_graph      esm2_8M_320     1.2563          0.6148           0.5660           190.6  432719.0
               gat_mol_graph_esm2_8M_320          gat     mol_graph      esm2_8M_320     1.2680          0.5969           0.5466           118.1  406145.0
                         distmat_cnn_aac  distmat_cnn   distmat_100           aac_20     1.3030          0.5112           0.4726           866.8 1285345.0
             cnn_smiles_onehot_esm2_150M          cnn smiles_onehot        esm2_150M     1.3243          0.5213           0.4302            42.8  184833.0
           cnn_smiles_onehot_esm2_8M_320          cnn smiles_onehot      esm2_8M_320     1.3286          0.5162           0.4328            51.8  184833.0
             cnn_smiles_onehot_esmc_300M          cnn smiles_onehot        esmc_300M     1.3329          0.5240           0.4183            27.5  184833.0
                distmat_cnn_protein_char  distmat_cnn   distmat_100     protein_char     1.3434          0.4857           0.4228           633.0 1285345.0
                            gcn_ecfp_aac          gcn     mol_graph           aac_20     1.3472          0.5171           0.4694            48.9  251777.0
          cnn_smiles_onehot_protein_char          cnn smiles_onehot     protein_char     1.3524          0.5119           0.4501            39.4  184833.0
                            gat_ecfp_aac          gat     mol_graph           aac_20     1.3525          0.5089           0.4499           101.4  252545.0
            cnn_smiles_onehot_kmer3_8000          cnn smiles_onehot       kmer3_8000     1.3561          0.5060           0.4302            39.7  184833.0
              gat_mol_graph_protein_char          gat     mol_graph     protein_char     1.3572          0.5162           0.4794           180.5  252545.0
      cnn_smiles_onehot_prot_electra_256          cnn smiles_onehot prot_electra_256     1.3680          0.4855           0.4027            38.3  184833.0
             cnn_smiles_onehot_esm2_650M          cnn smiles_onehot        esm2_650M     1.3769          0.4916           0.4188            36.8  184833.0
                   cnn_smiles_onehot_aac          cnn smiles_onehot           aac_20     1.3787          0.5005           0.4371            24.7  184833.0
              gcn_mol_graph_protein_char          gcn     mol_graph     protein_char     1.3877          0.5025           0.4456            82.9  251777.0
          cnn_smiles_onehot_esm2_35M_480          cnn smiles_onehot     esm2_35M_480     1.3893          0.4968           0.4124            65.6  184833.0
              gat_ecfp_esm2_8M__leakypdb          gat     mol_graph      esm2_8M_320     1.5839          0.4544           0.4331            36.4  406145.0
              gcn_ecfp_esm2_8M__leakypdb          gcn     mol_graph      esm2_8M_320     1.5969          0.4299           0.4184            20.6  405377.0


---
## 4. BiCA Bidirectional Cross-Attention Analysis

                        experiment_id model_family    ligand_repr     protein_repr  test_rmse  test_pearson_r  test_spearman_r  train_time_sec  n_params
          bica_chemberta_5M_esmc_300M         bica   chemberta_5M        esmc_300M     1.1020          0.7020           0.6310            44.5 1433345.0
          bica_chemberta_5M_esm2_650M         bica   chemberta_5M        esm2_650M     1.1068          0.6922           0.6091            81.3 1515265.0
           bica_chemberta_esm2_8M_dsm         bica  chemberta_600      esm2_8M_320     1.1244          0.6850           0.6142            75.2 1367809.0
        bica_chemberta_5M_esm2_8M_320         bica   chemberta_5M      esm2_8M_320     1.1468          0.6748           0.6047            61.3 1269505.0
         bica_chemberta_77M_esm2_650M         bica  chemberta_77M        esm2_650M     1.1560          0.6615           0.5765            60.0 1515265.0
        bica_chemberta_600_kmer3_8000         bica  chemberta_600       kmer3_8000     1.1661          0.6511           0.5804            46.8 3333889.0
          bica_chemberta_5M_esm2_150M         bica   chemberta_5M        esm2_150M     1.1673          0.6484           0.5751            51.7 1351425.0
      bica_chemberta_esm2_8M__seed123         bica  chemberta_600      esm2_8M_320     1.1729          0.6289           0.5799            69.6 1367809.0
               bica_chemberta_esm2_8M         bica  chemberta_600      esm2_8M_320     1.1764          0.6551           0.5806            60.2 1367809.0
  bica_chemberta_600_prot_electra_256         bica  chemberta_600 prot_electra_256     1.1775          0.6541           0.5877            81.5 1613569.0
        bica_chemberta_100M_esm2_650M         bica chemberta_100M        esm2_650M     1.1791          0.6446           0.5650            59.7 1613569.0
        bica_chemberta_esm2_8M_ranked         bica  chemberta_600      esm2_8M_320     1.1798          0.6436           0.5643            44.5 1367809.0
   bica_chemberta_5M_prot_electra_256         bica   chemberta_5M prot_electra_256     1.1810          0.6407           0.5730            50.2 1515265.0
      bica_chemberta_600_esm2_35M_480         bica  chemberta_600     esm2_35M_480     1.1839          0.6411           0.5738            57.1 1408769.0
         bica_chemberta_600_esm2_650M         bica  chemberta_600        esm2_650M     1.1839          0.6442           0.5797            65.1 1613569.0
          bica_chemberta_prot_electra         bica  chemberta_600 prot_electra_256     1.1858          0.6310           0.5756            43.0 1613569.0
              bica_ecfp4_aac__seed456         bica     ecfp4_1024           aac_20     1.1860          0.5900           0.5208            94.6 1356545.0
         bica_chemberta_600_esmc_300M         bica  chemberta_600        esmc_300M     1.1867          0.6518           0.5801            39.9 1531649.0
         bica_chemberta_600_esm2_150M         bica  chemberta_600        esm2_150M     1.1871          0.6273           0.5391            49.3 1449729.0
        bica_chemberta_100M_esm2_150M         bica chemberta_100M        esm2_150M     1.1874          0.6369           0.5529            67.3 1449729.0
       bica_chemberta_600_esm2_8M_320         bica  chemberta_600      esm2_8M_320     1.1887          0.6324           0.5598            54.0 1367809.0
      bica_chemberta_esm2_8M__seed456         bica  chemberta_600      esm2_8M_320     1.1901          0.5814           0.5024            65.5 1367809.0
       bica_chemberta_77M_esm2_8M_320         bica  chemberta_77M      esm2_8M_320     1.1932          0.6529           0.5907            64.4 1269505.0
     bica_chemberta_100M_esm2_35M_480         bica chemberta_100M     esm2_35M_480     1.2033          0.6281           0.5451            62.4 1408769.0
         bica_chemberta_77M_esm2_150M         bica  chemberta_77M        esm2_150M     1.2040          0.6264           0.5579            45.5 1351425.0
      bica_chemberta_prot_electra_dsm         bica  chemberta_600 prot_electra_256     1.2115          0.6277           0.5528            84.6 1613569.0
           bica_ecfp6_1024_kmer3_8000         bica     ecfp6_1024       kmer3_8000     1.2144          0.6334           0.5706            43.5 3399425.0
        bica_chemberta_100M_esmc_300M         bica chemberta_100M        esmc_300M     1.2243          0.6223           0.5474            47.0 1531649.0
       bica_chemberta_5M_esm2_35M_480         bica   chemberta_5M     esm2_35M_480     1.2271          0.5991           0.5504            42.2 1310465.0
  bica_chemberta_77M_prot_electra_256         bica  chemberta_77M prot_electra_256     1.2297          0.5852           0.5032            40.8 1515265.0
          bica_smiles_char_kmer3_8000         bica    smiles_char       kmer3_8000     1.2339          0.5841           0.5056            43.1 4135681.0
         bica_chemberta_77M_esmc_300M         bica  chemberta_77M        esmc_300M     1.2441          0.5973           0.5201            33.8 1433345.0
      bica_chemberta_77M_esm2_35M_480         bica  chemberta_77M     esm2_35M_480     1.2485          0.5990           0.5377            43.8 1310465.0
      bica_chemberta_100M_esm2_8M_320         bica chemberta_100M      esm2_8M_320     1.2491          0.5662           0.4854            39.7 1367809.0
      bica_chemberta_600_protein_char         bica  chemberta_600     protein_char     1.2558          0.5788           0.5084            55.9 1291009.0
 bica_chemberta_100M_prot_electra_256         bica chemberta_100M prot_electra_256     1.2710          0.6081           0.5319           101.4 1613569.0
     bica_ecfp6_1024_prot_electra_256         bica     ecfp6_1024 prot_electra_256     1.2774          0.5859           0.5442            46.5 1679105.0
         bica_ecfp6_1024_protein_char         bica     ecfp6_1024     protein_char     1.2942          0.5592           0.5143            43.2 1356545.0
                       bica_ecfp4_aac         bica     ecfp4_1024           aac_20     1.3081          0.5587           0.4896            38.0 1356545.0
              bica_ecfp4_aac__seed123         bica     ecfp4_1024           aac_20     1.3109          0.5089           0.4638            54.7 1356545.0
    bica_smiles_char_prot_electra_256         bica    smiles_char prot_electra_256     1.3288          0.5398           0.4647            39.4 2415361.0
        bica_smiles_char_protein_char         bica    smiles_char     protein_char     1.3370          0.4780           0.3874            37.4 2092801.0
bica_chemberta_prot_electra__leakypdb         bica  chemberta_600 prot_electra_256     1.5825          0.5236           0.4965            18.7 1613569.0
     bica_chemberta_esm2_8M__leakypdb         bica  chemberta_600      esm2_8M_320     1.5955          0.4464           0.4252            24.8 1367809.0


Comparison: BiCA vs MLP with same representations:

            experiment model  test_rmse  pearson_r
        bica_ecfp4_aac  BiCA     1.3081     0.5587
 mlp_shallow_ecfp4_aac   MLP     1.3285     0.5880
bica_chemberta_esm2_8M  BiCA     1.1764     0.6551
 mlp_chemberta_esm2_8M   MLP     1.1931     0.6472


---
## 5. Tokenization Strategy Comparison

### 5a. LSTM — effect of tokenization
                        experiment_id     ligand_repr          protein_repr  test_rmse  test_pearson_r  test_spearman_r
lstm_smiles_bpe_1000_protein_bpe_1000 smiles_bpe_1000      protein_bpe_1000     1.1456          0.6595           0.5812
    lstm_smiles_atom_protein_bpe_1000     smiles_atom      protein_bpe_1000     1.1456          0.6566           0.5679
        lstm_smiles_atom_protein_char     smiles_atom          protein_char     1.1654          0.6525           0.5660
    lstm_smiles_char_protein_bpe_1000     smiles_char      protein_bpe_1000     1.1907          0.6310           0.5320
    lstm_smiles_bpe512_protein_bpe512  smiles_bpe_512       protein_bpe_512     1.1947          0.6460           0.5730
  lstm_smiles_bpe1000_protein_bpe1000 smiles_bpe_1000      protein_bpe_1000     1.2227          0.6321           0.5880
        lstm_smiles_char_protein_char     smiles_char          protein_char     1.2462          0.6257           0.5372
    lstm_smiles_bpe_1000_protein_char smiles_bpe_1000          protein_char     1.2524          0.5902           0.5215
lstm_smiles_atom_protein_wordpiece512     smiles_atom protein_wordpiece_512     1.2608          0.5726           0.4523

### 5b. TransformerSeq — effect of tokenization
                                   experiment_id     ligand_repr          protein_repr  test_rmse  test_pearson_r  test_spearman_r
        transformer_seq_smiles_atom_protein_char     smiles_atom          protein_char     1.2096          0.6061           0.5138
    transformer_seq_smiles_char_protein_bpe_1000     smiles_char      protein_bpe_1000     1.2108          0.6050           0.5240
        transformer_seq_smiles_char_protein_char     smiles_char          protein_char     1.2215          0.5985           0.5097
    transformer_seq_smiles_atom_protein_bpe_1000     smiles_atom      protein_bpe_1000     1.2247          0.6161           0.5188
    transformer_seq_smiles_bpe512_protein_bpe512  smiles_bpe_512       protein_bpe_512     1.2282          0.5861           0.5342
transformer_seq_smiles_atom_protein_wordpiece512     smiles_atom protein_wordpiece_512     1.3360          0.4881           0.3535
transformer_seq_smiles_bpe_1000_protein_bpe_1000 smiles_bpe_1000      protein_bpe_1000     1.3698          0.5448           0.5124
    transformer_seq_smiles_bpe_1000_protein_char smiles_bpe_1000          protein_char     1.3730          0.5421           0.4843
  transformer_seq_smiles_bpe1000_protein_bpe1000 smiles_bpe_1000      protein_bpe_1000     1.5254          0.5091           0.5009

### 5c. BPE vs Character vs WordPiece (averaged across LSTM + TransformerSeq)
              test_rmse  test_pearson_r  test_spearman_r
tok_strategy                                            
atom_level      1.18750         0.62930          0.53990
bpe_512         1.21145         0.61605          0.55360
char            1.23385         0.61210          0.52345
bpe_1000        1.26607         0.59865          0.53310
wordpiece       1.29840         0.53035          0.40290


---
## 6. Ligand Representation Comparison (avg across all models)

                 test_rmse_mean  test_rmse_min  test_rmse_count  test_pearson_r_mean  test_pearson_r_min  test_pearson_r_count  test_spearman_r_mean  test_spearman_r_min  test_spearman_r_count
ligand_repr                                                                                                                                                                                     
chemberta_5M             1.1601         1.0434               30               0.6591              0.5484                    30                0.5920               0.5021                     30
distmat_100              1.1769         1.1092               11               0.6399              0.4857                    11                0.5737               0.4228                     11
chemberta_77M            1.1871         1.0661               30               0.6384              0.5091                    30                0.5670               0.4481                     30
chemberta_100M           1.2071         1.0662               30               0.6343              0.5028                    30                0.5557               0.4222                     30
smiles_atom              1.2188         1.1456                8               0.6024              0.4881                     8                0.5085               0.3535                      8
smiles_bpe_512           1.2190         1.1947                3               0.6208              0.5861                     3                0.5606               0.5342                      3
ecfp6_1024               1.2296         1.0691               19               0.6178              0.4099                    19                0.5489               0.3671                     19
smiles_bpe_1000          1.2663         1.1456                9               0.6018              0.5091                     9                0.5446               0.4843                      9
ecfp4_1024               1.2733         1.0065               38               0.5910              0.3177                    38                0.5364               0.2842                     38
mol_graph                1.2733         1.1939               25               0.5912              0.4299                    25                0.5413               0.4184                     25
maccs_167                1.2778         1.1213                4               0.4977              0.3932                     4                0.4372               0.3000                      4
chemberta_600            1.2827         1.0522               74               0.6272              0.1350                    74                0.5619               0.1046                     74
rdkit_200                1.3498         1.2919                3               0.4219              0.3641                     3                0.3647               0.3070                      3
smiles_onehot            1.3564         1.3243                9               0.5060              0.4855                     9                0.4258               0.4027                      9
smiles_char              1.3917         1.1293               20               0.5451              0.2396                    20                0.4793               0.2673                     20


---
## 7. Protein Representation Comparison (avg across all models)

                       test_rmse_mean  test_rmse_min  test_rmse_count  test_pearson_r_mean  test_pearson_r_min  test_pearson_r_count  test_spearman_r_mean  test_spearman_r_min  test_spearman_r_count
protein_repr                                                                                                                                                                                          
dipeptide_400                  1.1589         1.0192                7               0.6630              0.5243                     7                0.5957               0.4634                      7
esmc_300M                      1.1865         1.0599               23               0.6422              0.5079                    23                0.5702               0.4183                     23
esm2_650M                      1.1876         1.0434               24               0.6445              0.4916                    24                0.5744               0.4188                     24
esm2_150M                      1.1905         1.0587               24               0.6403              0.5213                    24                0.5680               0.4302                     24
esm2_35M_480                   1.1944         1.0473               26               0.6382              0.4968                    26                0.5681               0.4124                     26
protein_bpe_512                1.2190         1.1947                3               0.6208              0.5861                     3                0.5606               0.5342                      3
protein_bpe_1000               1.2363         1.1456               11               0.6127              0.5091                    11                0.5440               0.5009                     11
kmer3_8000                     1.2417         1.0687               23               0.6216              0.3431                    23                0.5509               0.3562                     23
prot_electra_256               1.2430         1.0566               47               0.6180              0.3459                    47                0.5541               0.3689                     47
protein_wordpiece_512          1.2984         1.2608                2               0.5304              0.4881                     2                0.4029               0.3535                      2
aac_20                         1.3012         1.0065               38               0.5453              0.3177                    38                0.4914               0.2842                     38
protein_char                   1.3155         1.0893               28               0.5580              0.2396                    28                0.4920               0.2673                     28
esm2_8M_320                    1.3235         1.0483               57               0.6140              0.1350                    57                0.5523               0.1046                     57


---
## 8. Compute Efficiency (Test RMSE vs Training Time)

                                   experiment_id    model_family  test_rmse  train_time_sec  rmse_per_minute
                                    rf_ecfp4_aac            tree     1.0065            23.4           2.5808
                              rf_ecfp4_dipeptide            tree     1.0192            62.1           0.9847
                      xgb_chemberta_5M_esm2_650M            tree     1.0434            12.1           5.1739
                   xgb_chemberta_5M_esm2_35M_480            tree     1.0473             6.4           9.8184
                               xgb_ecfp4_esm2_8M            tree     1.0483             4.3          14.6274
                      xgb_ecfp4_esm2_8M__seed456            tree     1.0510             4.6          13.7087
                           rf_ecfp4_aac__seed456            tree     1.0516            46.9           1.3453
                     xgb_chemberta_600_esm2_650M            tree     1.0522            15.2           4.1534
                                   xgb_ecfp4_aac            tree     1.0523             2.5          25.2552
                                  lgbm_ecfp4_aac            tree     1.0528             5.8          10.8910
                    xgb_chemberta_5M_esm2_8M_320            tree     1.0554             5.7          11.1095
               xgb_chemberta_5M_prot_electra_256            tree     1.0566            11.7           5.4185
                  xgb_chemberta_600_esm2_35M_480            tree     1.0577             9.0           7.0513
                      xgb_chemberta_prot_electra            tree     1.0580            14.6           4.3479
              xgb_chemberta_600_prot_electra_256            tree     1.0580            13.9           4.5669
                      xgb_chemberta_5M_esm2_150M            tree     1.0587             7.8           8.1438
                      xgb_chemberta_5M_esmc_300M            tree     1.0599             9.0           7.0660
                     xgb_chemberta_600_esmc_300M            tree     1.0627            11.4           5.5932
                     xgb_chemberta_77M_esm2_650M            tree     1.0661            11.8           5.4208
                    xgb_chemberta_100M_esm2_650M            tree     1.0662            14.1           4.5370
                   xgb_chemberta_600_esm2_8M_320            tree     1.0670             8.1           7.9037
                           xgb_chemberta_esm2_8M            tree     1.0670             8.1           7.9037
                    xgb_chemberta_600_kmer3_8000            tree     1.0687            23.6           2.7170
                       xgb_ecfp6_1024_kmer3_8000            tree     1.0691            19.1           3.3584
                                 xgb_ecfp6_kmer3            tree     1.0691            18.0           3.5637
                  xgb_chemberta_esm2_8M__seed456            tree     1.0693             8.8           7.2907
                     xgb_chemberta_600_esm2_150M            tree     1.0694            10.8           5.9411
                    xgb_chemberta_100M_esmc_300M            tree     1.0716            11.2           5.7407
                 xgb_chemberta_100M_esm2_35M_480            tree     1.0717             8.5           7.5649
                  xgb_chemberta_100M_esm2_8M_320            tree     1.0718             8.0           8.0385
                   xgb_chemberta_77M_esm2_8M_320            tree     1.0722             5.5          11.6967
                 xgb_ecfp6_1024_prot_electra_256            tree     1.0726            11.3           5.6952
                           rf_ecfp4_aac__seed123            tree     1.0728            46.9           1.3725
                    mlp_chemberta_5M_esm2_8M_320             mlp     1.0744            41.2           1.5647
                    xgb_chemberta_100M_esm2_150M            tree     1.0770            10.8           5.9833
                  xgb_chemberta_77M_esm2_35M_480            tree     1.0771             6.2          10.4235
             xgb_chemberta_100M_prot_electra_256            tree     1.0775            13.7           4.7190
             xgb_chemberta_prot_electra__seed123            tree     1.0778            15.3           4.2267
                     xgb_chemberta_77M_esmc_300M            tree     1.0780             8.9           7.2674
             xgb_chemberta_prot_electra__seed456            tree     1.0795            14.8           4.3764
                   mlp_chemberta_77M_esm2_8M_320             mlp     1.0796            27.3           2.3727
                  mlp_chemberta_77M_esm2_35M_480             mlp     1.0801            42.4           1.5284
              xgb_chemberta_77M_prot_electra_256            tree     1.0816            11.4           5.6926
                      xgb_ecfp4_esm2_8M__seed123            tree     1.0818             4.6          14.1104
                     xgb_chemberta_77M_esm2_150M            tree     1.0822             7.6           8.5437
                          xgb_ecfp4_aac__seed456            tree     1.0838             2.5          26.0112
                  xgb_chemberta_esm2_8M__seed123            tree     1.0882             8.3           7.8665
                               xgb_chemberta_aac            tree     1.0893             5.7          11.4663
                  xgb_chemberta_600_protein_char            tree     1.0893             6.1          10.7144
                     mlp_chemberta_600_esm2_650M             mlp     1.0901            40.7           1.6070
                         lgbm_ecfp4_aac__seed456            tree     1.0906             8.5           7.6984
                      xgb_chemberta_aac__seed456            tree     1.0918             6.1          10.7390
                     xgb_ecfp6_1024_protein_char            tree     1.0920             2.5          26.2080
              mlp_chemberta_600_prot_electra_256             mlp     1.0942            52.4           1.2529
                     mlp_chemberta_600_esmc_300M             mlp     1.0978            20.5           3.2131
                    mlp_chemberta_100M_esm2_650M             mlp     1.1007            36.1           1.8294
                     bica_chemberta_5M_esmc_300M            bica     1.1020            44.5           1.4858
                   mlp_chemberta_5M_esm2_35M_480             mlp     1.1036            35.5           1.8652
                    mlp_chemberta_100M_esmc_300M             mlp     1.1062            25.0           2.6549
                     bica_chemberta_5M_esm2_650M            bica     1.1068            81.3           0.8168
                      mlp_medium_ecfp4_dipeptide             mlp     1.1082            23.9           2.7821
                   mlp_chemberta_600_esm2_8M_320             mlp     1.1091            34.6           1.9233
            distmat_cnn_distmat_100_esm2_35M_480     distmat_cnn     1.1092          1330.4           0.0500
                     mlp_chemberta_77M_esm2_650M             mlp     1.1115            34.5           1.9330
                          mlp_ecfp4_prot_electra             mlp     1.1129            49.2           1.3572
                             distmat_cnn_esm2_8M     distmat_cnn     1.1147          1235.2           0.0541
                     mlp_chemberta_77M_esmc_300M             mlp     1.1172            15.3           4.3812
                         lgbm_ecfp4_aac__seed123            tree     1.1174             8.2           8.1761
             mlp_chemberta_prot_electra__seed123             mlp     1.1179            47.4           1.4151
     transformer_chemberta_100M_prot_electra_256     transformer     1.1194            77.6           0.8655
                   mlp_deep_ecfp6_kmer3__seed456             mlp     1.1212            49.3           1.3645
             mlp_chemberta_prot_electra__seed456             mlp     1.1213            56.6           1.1887
                        mlp_wide_maccs_dipeptide             mlp     1.1213            32.6           2.0637
                      mlp_chemberta_5M_esm2_650M             mlp     1.1216            28.4           2.3696
                    mlp_chemberta_600_kmer3_8000             mlp     1.1227            24.7           2.7272
                      mlp_ecfp4_esm2_8M__seed456             mlp     1.1233            25.2           2.6745
                      mlp_chemberta_5M_esm2_150M             mlp     1.1239            28.8           2.3415
                      bica_chemberta_esm2_8M_dsm            bica     1.1244            75.2           0.8971
                         mlp_chemberta_dipeptide             mlp     1.1246            34.2           1.9730
               distmat_cnn_distmat_100_esm2_150M     distmat_cnn     1.1265           926.5           0.0730
               mlp_chemberta_5M_prot_electra_256             mlp     1.1266            31.5           2.1459
                      xgb_smiles_char_kmer3_8000            tree     1.1293            20.5           3.3053
                               mlp_ecfp4_esm2_8M             mlp     1.1293            15.1           4.4873
              transformer_chemberta_5M_esm2_150M     transformer     1.1296            94.7           0.7157
                          xgb_ecfp4_aac__seed123            tree     1.1302             2.4          28.2550
                xgb_smiles_char_prot_electra_256            tree     1.1309            13.7           4.9528
                      xgb_chemberta_aac__seed123            tree     1.1324             6.0          11.3240
                    xgb_smiles_char_protein_char            tree     1.1329             4.8          14.1612
                  mlp_chemberta_600_esm2_35M_480             mlp     1.1334            29.9           2.2744
                       mlp_ecfp6_1024_kmer3_8000             mlp     1.1335            43.4           1.5671
                      mlp_chemberta_5M_esmc_300M             mlp     1.1378            19.2           3.5556
              mlp_chemberta_77M_prot_electra_256             mlp     1.1405            26.6           2.5726
               distmat_cnn_distmat_100_esmc_300M     distmat_cnn     1.1423           885.1           0.0774
                    mlp_chemberta_esm2_8M_ranked             mlp     1.1432            34.4           1.9940
                    mlp_chemberta_100M_esm2_150M             mlp     1.1440            27.0           2.5422
             mlp_chemberta_100M_prot_electra_256             mlp     1.1442            35.3           1.9448
                          distmat_cnn_kmer3_8000     distmat_cnn     1.1448           761.5           0.0902
           lstm_smiles_bpe_1000_protein_bpe_1000            lstm     1.1456          1082.8           0.0635
               lstm_smiles_atom_protein_bpe_1000            lstm     1.1456          1284.1           0.0535
                      mlp_chemberta_prot_electra             mlp     1.1466            64.8           1.0617
                   bica_chemberta_5M_esm2_8M_320            bica     1.1468            61.3           1.1225
         transformer_chemberta_100M_esm2_35M_480     transformer     1.1540            97.5           0.7102
               distmat_cnn_distmat_100_esm2_650M     distmat_cnn     1.1548           602.6           0.1150
              transformer_chemberta_5M_esmc_300M     transformer     1.1550            60.4           1.1474
                    bica_chemberta_77M_esm2_650M            bica     1.1560            60.0           1.1560
             transformer_chemberta_77M_esm2_150M     transformer     1.1568            78.5           0.8842
             mlp_medium_ecfp4_dipeptide__seed123             mlp     1.1570            38.0           1.8268
                     mlp_chemberta_600_esm2_150M             mlp     1.1580            33.8           2.0556
          mamba_smiles_bpe_1000_protein_bpe_1000           mamba     1.1581           558.8           0.1243
                   transformer_chemberta_esm2_8M     transformer     1.1583            46.9           1.4818
          transformer_chemberta_100M_esm2_8M_320     transformer     1.1589           104.4           0.6660
        distmat_cnn_distmat_100_prot_electra_256     distmat_cnn     1.1590           863.9           0.0805
             mlp_medium_ecfp4_dipeptide__seed456             mlp     1.1591            24.5           2.8386
                     mlp_chemberta_77M_esm2_150M             mlp     1.1592            22.1           3.1471
                    distmat_cnn_prot_electra_256     distmat_cnn     1.1604          1088.9           0.0639
             transformer_chemberta_77M_esm2_650M     transformer     1.1625            71.7           0.9728
                  mlp_chemberta_esm2_8M__seed456             mlp     1.1627            35.1           1.9875
                   lstm_smiles_atom_protein_char            lstm     1.1654           765.4           0.0914
                   bica_chemberta_600_kmer3_8000            bica     1.1661            46.8           1.4950
                 mlp_chemberta_100M_esm2_35M_480             mlp     1.1665            28.9           2.4218
                     bica_chemberta_5M_esm2_150M            bica     1.1673            51.7           1.3547
                          mlp_chemberta_esm2_35M             mlp     1.1692            26.4           2.6573
              mamba_smiles_bpe_1000_protein_char           mamba     1.1722           573.3           0.1227
                 bica_chemberta_esm2_8M__seed123            bica     1.1729            69.6           1.0111
                  mamba_smiles_char_protein_char           mamba     1.1755           778.1           0.0906
                          bica_chemberta_esm2_8M            bica     1.1764            60.2           1.1725
             bica_chemberta_600_prot_electra_256            bica     1.1775            81.5           0.8669
            mamba_smiles_bpe1000_protein_bpe1000           mamba     1.1778           697.5           0.1013
                   bica_chemberta_100M_esm2_650M            bica     1.1791            59.7           1.1850
             transformer_chemberta_77M_esmc_300M     transformer     1.1791            54.4           1.3005
                   bica_chemberta_esm2_8M_ranked            bica     1.1798            44.5           1.5907
            transformer_chemberta_100M_esmc_300M     transformer     1.1805            98.4           0.7198
                  mamba_smiles_atom_protein_char           mamba     1.1808           756.0           0.0937
                  mlp_chemberta_100M_esm2_8M_320             mlp     1.1810            20.5           3.4566
              bica_chemberta_5M_prot_electra_256            bica     1.1810            50.2           1.4116
                  mlp_chemberta_esm2_8M__seed123             mlp     1.1820            49.9           1.4212
                 bica_chemberta_600_esm2_35M_480            bica     1.1839            57.1           1.2440
                    bica_chemberta_600_esm2_650M            bica     1.1839            65.1           1.0912
      transformer_chemberta_77M_prot_electra_256     transformer     1.1841            55.6           1.2778
                     bica_chemberta_prot_electra            bica     1.1858            43.0           1.6546
                         bica_ecfp4_aac__seed456            bica     1.1860            94.6           0.7522
             transformer_chemberta_600_esm2_650M     transformer     1.1864            96.2           0.7400
                    bica_chemberta_600_esmc_300M            bica     1.1867            39.9           1.7845
                    bica_chemberta_600_esm2_150M            bica     1.1871            49.3           1.4447
             distmat_cnn_distmat_100_esm2_8M_320     distmat_cnn     1.1873           747.1           0.0954
                   bica_chemberta_100M_esm2_150M            bica     1.1874            67.3           1.0586
            transformer_chemberta_600_kmer3_8000     transformer     1.1883           330.8           0.2155
                  bica_chemberta_600_esm2_8M_320            bica     1.1887            54.0           1.3208
                 bica_chemberta_esm2_8M__seed456            bica     1.1901            65.5           1.0902
              transformer_chemberta_5M_esm2_650M     transformer     1.1906            46.6           1.5330
               lstm_smiles_char_protein_bpe_1000            lstm     1.1907          1213.0           0.0589
            transformer_chemberta_100M_esm2_650M     transformer     1.1909            54.4           1.3135
          transformer_chemberta_600_esm2_35M_480     transformer     1.1912            67.5           1.0588
                  transformer_chemberta_esm2_35M     transformer     1.1914            27.4           2.6089
                            mlp_deep_ecfp6_kmer3             mlp     1.1926            72.5           0.9870
                           mlp_chemberta_esm2_8M             mlp     1.1931            20.2           3.5439
                  bica_chemberta_77M_esm2_8M_320            bica     1.1932            64.4           1.1117
                         gat_mol_graph_esm2_650M             gat     1.1939           151.1           0.4741
                   mlp_deep_ecfp6_kmer3__seed123             mlp     1.1940            60.7           1.1802
               lstm_smiles_bpe512_protein_bpe512            lstm     1.1947           621.7           0.1153
                         gat_ecfp_esm2_8M_ranked             gat     1.1963           208.9           0.3436
            transformer_chemberta_100M_esm2_150M     transformer     1.1970            62.9           1.1418
      transformer_chemberta_600_prot_electra_256     transformer     1.1976            61.0           1.1780
           transformer_chemberta_77M_esm2_8M_320     transformer     1.1979            68.2           1.0539
         transformer_ecfp6_1024_prot_electra_256     transformer     1.1980            71.2           1.0096
       transformer_chemberta_5M_prot_electra_256     transformer     1.2020            56.8           1.2697
                      gcn_mol_graph_esm2_35M_480             gcn     1.2024            73.9           0.9762
             transformer_chemberta_600_esm2_150M     transformer     1.2029            56.1           1.2865
                               mlp_chemberta_aac             mlp     1.2032            21.3           3.3893
                bica_chemberta_100M_esm2_35M_480            bica     1.2033            62.4           1.1570
                 mlp_ecfp6_1024_prot_electra_256             mlp     1.2036            48.6           1.4859
                    bica_chemberta_77M_esm2_150M            bica     1.2040            45.5           1.5877
                         gcn_mol_graph_esmc_300M             gcn     1.2073            88.7           0.8167
                         gcn_mol_graph_esm2_150M             gcn     1.2086           129.9           0.5582
                         gat_mol_graph_esmc_300M             gat     1.2088            93.9           0.7724
                           transformer_ecfp4_aac     transformer     1.2095            46.2           1.5708
        transformer_seq_smiles_atom_protein_char transformer_seq     1.2096           647.4           0.1121
           transformer_chemberta_600_esm2_8M_320     transformer     1.2099            88.0           0.8249
    transformer_seq_smiles_char_protein_bpe_1000 transformer_seq     1.2108           710.6           0.1022
                 bica_chemberta_prot_electra_dsm            bica     1.2115            84.6           0.8592
           transformer_chemberta_5M_esm2_35M_480     transformer     1.2127            52.1           1.3966
                      mlp_ecfp4_esm2_8M__seed123             mlp     1.2141            42.0           1.7344
                        gcn_mol_graph_kmer3_8000             gcn     1.2141           138.2           0.5271
                      bica_ecfp6_1024_kmer3_8000            bica     1.2144            43.5           1.6750
                      gat_mol_graph_esm2_35M_480             gat     1.2148           151.7           0.4805
               transformer_ecfp6_1024_kmer3_8000     transformer     1.2153           280.9           0.2596
                       gcn_mol_graph_esm2_8M_320             gcn     1.2156           141.7           0.5147
                                gcn_ecfp_esm2_8M             gcn     1.2158            98.6           0.7398
                        gat_mol_graph_kmer3_8000             gat     1.2161           167.5           0.4356
                      mlp_smiles_char_kmer3_8000             mlp     1.2170            27.5           2.6553
                          gcn_ecfp_esm2_8M_recon             gcn     1.2187            91.0           0.8035
                     mlp_ecfp6_1024_protein_char             mlp     1.2192            29.1           2.5138
        transformer_seq_smiles_char_protein_char transformer_seq     1.2215           689.9           0.1062
          transformer_chemberta_77M_esm2_35M_480     transformer     1.2216            58.8           1.2465
             lstm_smiles_bpe1000_protein_bpe1000            lstm     1.2227           530.0           0.1384
                   bica_chemberta_100M_esmc_300M            bica     1.2243            47.0           1.5629
    transformer_seq_smiles_atom_protein_bpe_1000 transformer_seq     1.2247           776.4           0.0946
            transformer_chemberta_5M_esm2_8M_320     transformer     1.2255            49.7           1.4795
                  gcn_mol_graph_prot_electra_256             gcn     1.2255           178.8           0.4112
                  bica_chemberta_5M_esm2_35M_480            bica     1.2271            42.2           1.7447
                  gat_mol_graph_prot_electra_256             gat     1.2274           153.4           0.4801
              mamba_smiles_atom_protein_bpe_1000           mamba     1.2279           481.1           0.1531
    transformer_seq_smiles_bpe512_protein_bpe512 transformer_seq     1.2282           473.4           0.1557
                         gat_mol_graph_esm2_150M             gat     1.2285           165.5           0.4454
             bica_chemberta_77M_prot_electra_256            bica     1.2297            40.8           1.8084
        transformer_smiles_char_prot_electra_256     transformer     1.2299           143.1           0.5157
                         gcn_mol_graph_esm2_650M             gcn     1.2329           107.4           0.6888
                     bica_smiles_char_kmer3_8000            bica     1.2339            43.1           1.7177
              mamba_smiles_bpe512_protein_bpe512           mamba     1.2340           647.8           0.1143
             transformer_chemberta_600_esmc_300M     transformer     1.2352            36.3           2.0417
              transformer_smiles_char_kmer3_8000     transformer     1.2410           352.7           0.2111
                    bica_chemberta_77M_esmc_300M            bica     1.2441            33.8           2.2085
                   lstm_smiles_char_protein_char            lstm     1.2462           880.2           0.0849
                 bica_chemberta_77M_esm2_35M_480            bica     1.2485            43.8           1.7103
          transformer_chemberta_600_protein_char     transformer     1.2487            49.8           1.5045
                 bica_chemberta_100M_esm2_8M_320            bica     1.2491            39.7           1.8878
               lstm_smiles_bpe_1000_protein_char            lstm     1.2524           829.4           0.0906
                 ridge_chemberta_5M_esm2_35M_480          linear     1.2544             0.2         376.3200
                    ridge_chemberta_5M_esm2_150M          linear     1.2545             0.2         376.3500
                                gat_ecfp_esm2_8M             gat     1.2551            93.8           0.8028
                 bica_chemberta_600_protein_char            bica     1.2558            55.9           1.3479
                          gat_ecfp_esm2_8M_recon             gat     1.2563           190.6           0.3955
                  mlp_chemberta_600_protein_char             mlp     1.2599            23.3           3.2444
           lstm_smiles_atom_protein_wordpiece512            lstm     1.2608           226.8           0.3335
                  ridge_chemberta_5M_esm2_8M_320          linear     1.2673             0.2         380.1900
                       gat_mol_graph_esm2_8M_320             gat     1.2680           118.1           0.6442
            bica_chemberta_100M_prot_electra_256            bica     1.2710           101.4           0.7521
                bica_ecfp6_1024_prot_electra_256            bica     1.2774            46.5           1.6483
             transformer_ecfp6_1024_protein_char     transformer     1.2806            76.0           1.0110
                                 ridge_rdkit_aac          linear     1.2919             0.0              inf
                    bica_ecfp6_1024_protein_char            bica     1.2942            43.2           1.7975
                mlp_smiles_char_prot_electra_256             mlp     1.2957            21.0           3.7020
                    ridge_chemberta_5M_esm2_650M          linear     1.3004             0.4         195.0600
                                 distmat_cnn_aac     distmat_cnn     1.3030           866.8           0.0902
                                 ridge_maccs_aac          linear     1.3079             0.1         784.7400
                                  bica_ecfp4_aac            bica     1.3081            38.0           2.0654
                         bica_ecfp4_aac__seed123            bica     1.3109            54.7           1.4379
                   ridge_chemberta_77M_esm2_150M          linear     1.3173             0.3         263.4600
                    ridge_chemberta_5M_esmc_300M          linear     1.3217             0.4         198.2550
                     cnn_smiles_onehot_esm2_150M             cnn     1.3243            42.8           1.8565
                 ridge_chemberta_77M_esm2_8M_320          linear     1.3248             0.2         397.4400
                           mlp_shallow_ecfp4_aac             mlp     1.3285            30.6           2.6049
                   cnn_smiles_onehot_esm2_8M_320             cnn     1.3286            51.8           1.5389
               bica_smiles_char_prot_electra_256            bica     1.3288            39.4           2.0236
                     cnn_smiles_onehot_esmc_300M             cnn     1.3329            27.5           2.9081
                        ridge_maccs_aac__seed456          linear     1.3356             0.1         801.3600
transformer_seq_smiles_atom_protein_wordpiece512 transformer_seq     1.3360           135.9           0.5898
                ridge_chemberta_77M_esm2_35M_480          linear     1.3368             0.2         401.0400
                   bica_smiles_char_protein_char            bica     1.3370            37.4           2.1449
                        distmat_cnn_protein_char     distmat_cnn     1.3434           633.0           0.1273
            transformer_smiles_char_protein_char     transformer     1.3450            86.3           0.9351
                   ridge_chemberta_77M_esm2_650M          linear     1.3460             0.4         201.9000
                        ridge_maccs_aac__seed123          linear     1.3462             0.1         807.7200
                                    gcn_ecfp_aac             gcn     1.3472            48.9           1.6530
             ridge_chemberta_5M_prot_electra_256          linear     1.3504             0.4         202.5600
                  cnn_smiles_onehot_protein_char             cnn     1.3524            39.4           2.0595
                                    gat_ecfp_aac             gat     1.3525           101.4           0.8003
                    cnn_smiles_onehot_kmer3_8000             cnn     1.3561            39.7           2.0495
                      gat_mol_graph_protein_char             gat     1.3572           180.5           0.4511
                        ridge_rdkit_aac__seed456          linear     1.3670             0.1         820.2000
              cnn_smiles_onehot_prot_electra_256             cnn     1.3680            38.3           2.1431
transformer_seq_smiles_bpe_1000_protein_bpe_1000 transformer_seq     1.3698           614.6           0.1337
    transformer_seq_smiles_bpe_1000_protein_char transformer_seq     1.3730           669.9           0.1230
                     cnn_smiles_onehot_esm2_650M             cnn     1.3769            36.8           2.2449
            ridge_chemberta_77M_prot_electra_256          linear     1.3785             0.5         165.4200
                           cnn_smiles_onehot_aac             cnn     1.3787            24.7           3.3491
                   ridge_chemberta_77M_esmc_300M          linear     1.3871             0.3         277.4200
                      gcn_mol_graph_protein_char             gcn     1.3877            82.9           1.0044
                ridge_chemberta_600_esm2_35M_480          linear     1.3880             0.3         277.6000
                  cnn_smiles_onehot_esm2_35M_480             cnn     1.3893            65.6           1.2707
                        ridge_rdkit_aac__seed123          linear     1.3904             0.1         834.2400
                   ridge_chemberta_600_esm2_150M          linear     1.3948             0.6         139.4800
                  ridge_chemberta_600_kmer3_8000          linear     1.3953             7.0          11.9597
                  ridge_chemberta_100M_esm2_150M          linear     1.4138             0.3         282.7600
               ridge_chemberta_100M_esm2_35M_480          linear     1.4187             0.3         283.7400
                   ridge_chemberta_600_esm2_650M          linear     1.4206             0.6         142.0600
                ridge_chemberta_100M_esm2_8M_320          linear     1.4217             0.3         284.3400
                           ridge_ecfp4_dipeptide          linear     1.4227             0.3         284.5400
                    mlp_smiles_char_protein_char             mlp     1.4272            46.4           1.8455
                 ridge_chemberta_600_esm2_8M_320          linear     1.4336             0.4         215.0400
           ridge_chemberta_100M_prot_electra_256          linear     1.4448             0.8         108.3600
            ridge_chemberta_600_prot_electra_256          linear     1.4497             0.7         124.2600
                  ridge_chemberta_100M_esmc_300M          linear     1.4509             0.4         217.6350
                ridge_chemberta_600_protein_char          linear     1.4536             0.2         436.0800
               ridge_ecfp6_1024_prot_electra_256          linear     1.4669             0.8         110.0175
                  ridge_chemberta_100M_esm2_650M          linear     1.4699             0.6         146.9900
                  mlp_shallow_ecfp4_aac__seed456             mlp     1.4882            42.7           2.0911
                     ridge_ecfp6_1024_kmer3_8000          linear     1.4886             8.0          11.1645
                        ridge_ecfp4_aac__seed456          linear     1.5051             0.4         225.7650
                  mlp_shallow_ecfp4_aac__seed123             mlp     1.5131            63.2           1.4365
                         xgb_ecfp4_aac__leakypdb            tree     1.5199             2.4          37.9975
                        lgbm_ecfp4_aac__leakypdb            tree     1.5247             8.9          10.2789
  transformer_seq_smiles_bpe1000_protein_bpe1000 transformer_seq     1.5254           598.0           0.1531
                 xgb_chemberta_esm2_8M__leakypdb            tree     1.5287             9.1          10.0793
                     xgb_ecfp4_esm2_8M__leakypdb            tree     1.5421             4.7          19.6864
                     xgb_chemberta_aac__leakypdb            tree     1.5488             6.9          13.4678
                                 ridge_ecfp4_aac          linear     1.5498             0.3         309.9600
                 mlp_chemberta_esm2_8M__leakypdb             mlp     1.5563            16.7           5.5915
                          rf_ecfp4_aac__leakypdb            tree     1.5585            25.9           3.6104
                   ridge_ecfp6_1024_protein_char          linear     1.5608             0.4         234.1200
           bica_chemberta_prot_electra__leakypdb            bica     1.5825            18.7           5.0775
                      gat_ecfp_esm2_8M__leakypdb             gat     1.5839            36.4           2.6108
                bica_chemberta_esm2_8M__leakypdb            bica     1.5955            24.8           3.8601
                      gcn_ecfp_esm2_8M__leakypdb             gcn     1.5969            20.6           4.6512
                mlp_ecfp4_prot_electra__leakypdb             mlp     1.6120            13.8           7.0087
                        ridge_ecfp4_aac__seed123          linear     1.6137             0.4         242.0550
            mlp_chemberta_prot_electra__leakypdb             mlp     1.6361            19.3           5.0863
                     mlp_ecfp4_esm2_8M__leakypdb             mlp     1.6450            18.6           5.3065
                       ridge_ecfp4_aac__leakypdb          linear     1.6862             0.3         337.2400
                    ridge_smiles_char_kmer3_8000          linear     2.1670            13.4           9.7030
              ridge_smiles_char_prot_electra_256          linear     2.1715             3.0          43.4300
                  ridge_smiles_char_protein_char          linear     2.4017             1.9          75.8432
                       mlp_chemberta_esm2_8M_dsm             mlp     6.8504            26.3          15.6283


---
## 9. Key Findings

1. **Best overall model:** `rf_ecfp4_aac` — Test RMSE=1.0065, Pearson=0.7467, Spearman=0.6737

2. **Tree models remain the best efficiency trade-off:** `rf_ecfp4_aac` RMSE=1.0065 in 23s — competitive with deep learning models costing 10–100× more compute.

3. **GNNs underperform expectations on this dataset:** Best GNN `gat_mol_graph_esm2_650M` achieves RMSE=1.1939, worse than RF+ECFP4 (RMSE=1.0065). Likely causes: (a) scaffold split penalises topology-based methods heavily; (b) GNN benefits more from 3D conformer features (not used here); (c) 78-dim node features vs 1024-bit ECFP may lose global substructure info. ESM-2 protein significantly boosts GNN: `gcn_ecfp_aac` RMSE=1.3472 → `gcn_ecfp_esm2_8M` RMSE=1.2158.

4. **Distance matrix CNN is the weakest structural encoder:** `distmat_cnn_distmat_100_esm2_35M_480` RMSE=1.1092 — worse than GNN and ECFP-based models. The 2D topological matrix loses atom-type and bond-type information that GNNs retain as node/edge features. Also very slow to train (1330s) due to large 100×100 input tensors.

5. **BiCA cross-attention adds no benefit over MLP on flat vectors:** Best BiCA `bica_chemberta_5M_esmc_300M` RMSE=1.1020 — similar to `mlp_chemberta_esm2_8M` (RMSE=1.1931). Cross-attention on seq_len=1 flat vectors degenerates to a linear transform; BiCA needs true sequence inputs (atom-level graphs, residue sequences) to leverage its bidirectional attention mechanism.

6. **ProtElectra (RTD) is on par with ESM-2 8M for flat-feature models:** `bica_chemberta_prot_electra` RMSE=1.1858 vs `bica_chemberta_esm2_8M` RMSE=1.1764. ProtElectra's discriminative RTD pre-training yields comparable representations to ESM-2's MLM despite being a smaller model (256-dim vs 320-dim).

7. **ESM-2 protein embeddings remain the single biggest signal boost:** Every model family improves ~0.05–0.1 pKd RMSE when swapping AAC → ESM-2. This holds for GNN, distmat CNN, and BiCA — the protein encoder is the bottleneck, not the ligand encoder architecture.

8. **XGBoost + pre-trained embeddings is the best efficiency trade-off:** `xgb_chemberta_esm2_8M` RMSE=1.0670 in 8s — best or near-best result at a fraction of the compute of any deep learning model.

9. **Scaffold split is hard for all structural encoders:** Best Pearson R across all models is ~0.57, best RMSE ~1.44 pKd units. GNNs and distmat CNN both score worse than fingerprint-based models, confirming that structural similarity alone does not generalise across scaffold boundaries. Pre-trained protein representations help more than ligand architecture choice.

10. **Tokenization: BPE-512 is still best for sequence models.** Best tok strategy avg RMSE: 1.1875 (atom_level), worst: 1.2984 (wordpiece). WordPiece consistently underperforms — designed for NLP, not biochemical sequences.