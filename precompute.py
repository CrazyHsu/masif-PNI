#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
File name: precompute.py
Author: CrazyHsu @ crazyhsu9627@gmail.com
Created on: 2022-09-11 20:49:01
Last modified: 2022-09-11 20:49:01
'''

import os
import numpy as np
import pandas as pd
from biopandas.pdb import PandasPdb
# import warnings

# from defaultConfig import DefaultConfig
from commonFuncs import resolveDir
from readDataFromSurface import read_data_from_surface
# from readDataFromSurface import read_data_from_surface1, rebuildInputFeat

# with warnings.catch_warnings():
#     warnings.filterwarnings("ignore", category=FutureWarning)


def getNaCorrd(masifpniOpts, rawPDB, pdb_id, pChain, naChain):
    if not os.path.exists(rawPDB): return
    params = masifpniOpts['masifpni_site']
    baseGroup = {"A": ["N9", "C8", "N7", "C5", "C6", "N6", "N1", "C2", "N3", "C4"],
                 "G": ["N9", "C8", "N7", "C5", "C6", "O6", "N1", "C2", "N2", "N3", "C4"],
                 "U": ["N1", "C2", "O2", "N3", "C4", "O4", "C5", "C6"],
                 "C": ["N1", "C2", "O2", "N3", "C4", "N4", "C5", "C6"],
                 "DA": ["N9", "C8", "N7", "C5", "C6", "N6", "N1", "C2", "N3", "C4"],
                 "DG": ["N9", "C8", "N7", "C5", "C6", "O6", "N1", "C2", "N2", "N3", "C4"],
                 "DT": ["N1", "C2", "O2", "N3", "C4", "O4", "C5", "C6", "C7"],
                 "DC": ["N1", "C2", "O2", "N3", "C4", "N4", "C5", "C6"],
                 }

    my_precomp_dir = os.path.join(params['masif_precomputation_dir'], pdb_id)
    resolveDir(my_precomp_dir, chdir=False)

    ppdb = PandasPdb()
    pdbStruc = ppdb.read_pdb(rawPDB)
    atomDf = pdbStruc.df["ATOM"]
    validAtomDf = []
    naChainList = list(naChain)
    atomDf = atomDf.loc[atomDf.chain_id.isin(naChainList)]
    target_bases = list(set(baseGroup.keys()) & set(atomDf["residue_name"].to_list()))
    if len(target_bases) == 0: return
    for base in target_bases:
        tmpDf = atomDf.loc[(atomDf["residue_name"] == base)]
        validAtomDf.append(tmpDf.loc[:, ['x_coord', 'y_coord', 'z_coord']].to_numpy())
    #     pass
    # for base, base_alist in baseGroup.items():
    #     if base not in atomDf["residue_name"].to_list(): continue
    #     tmpDf = atomDf.loc[(atomDf["residue_name"] == base) & atomDf["atom_name"].isin(base_alist)]
    #     validAtomDf.append(tmpDf)

    # validAtomDf = pd.concat(validAtomDf, ignore_index=True)
    # validAtomDf = validAtomDf.sort_values(["chain_id", "residue_number"])
    outName = "{}_{}_{}".format(pdb_id, pChain, naChain)
    np.save(os.path.join(my_precomp_dir, '{}_NA_types.npy'.format(outName)), target_bases)
    np.save(os.path.join(my_precomp_dir, '{}_NA_coords.npy'.format(outName)), validAtomDf)


# def precomputeProteinPlyInfo(masifpniOpts, pdb_id, pChain):
#     ply_file = masifpniOpts['ply_file_template'].format(pdb_id, pChain)
#     if not os.path.exists(ply_file): return
#
#     params = masifpniOpts['masifpni_site']
#
#     try:
#         input_feat, rho, theta, mask, neigh_indices, iface_labels, verts = read_data_from_surface(ply_file, params)
#     except:
#         return
#
#     my_precomp_dir = os.path.join(params['masif_precomputation_dir'], pdb_id)
#     resolveDir(my_precomp_dir, chdir=False)
#     np.save(os.path.join(my_precomp_dir, pChain + '_rho_wrt_center.npy'), rho)
#     np.save(os.path.join(my_precomp_dir, pChain + '_theta_wrt_center.npy'), theta)
#     np.save(os.path.join(my_precomp_dir, pChain + '_input_feat.npy'), input_feat)
#     np.save(os.path.join(my_precomp_dir, pChain + '_mask.npy'), mask)
#     np.save(os.path.join(my_precomp_dir, pChain + '_list_indices.npy'), neigh_indices)
#     np.save(os.path.join(my_precomp_dir, pChain + '_iface_labels.npy'), iface_labels)
#     # Save x, y, z
#     np.save(os.path.join(my_precomp_dir, pChain + '_X.npy'), verts[:, 0])
#     np.save(os.path.join(my_precomp_dir, pChain + '_Y.npy'), verts[:, 1])
#     np.save(os.path.join(my_precomp_dir, pChain + '_Z.npy'), verts[:, 2])


def precomputeProteinPlyInfo1(masifpniOpts, pPlyFile, pdb_id, pChain, naChain):
    if not os.path.exists(pPlyFile): return

    params = masifpniOpts['masifpni_site']

    try:
        input_feat, rho, theta, mask, neigh_indices, iface_labels, verts = read_data_from_surface(pPlyFile, params)
    except Exception as e:
        print(e)
        return

    my_precomp_dir = os.path.join(params['masif_precomputation_dir'], pdb_id)
    resolveDir(my_precomp_dir, chdir=False)
    outName = "{}_{}_{}".format(pdb_id, pChain, naChain)
    np.save(os.path.join(my_precomp_dir, '{}_rho_wrt_center.npy'.format(outName)), rho)
    np.save(os.path.join(my_precomp_dir, '{}_theta_wrt_center.npy'.format(outName)), theta)
    np.save(os.path.join(my_precomp_dir, '{}_input_feat.npy'.format(outName)), input_feat)
    np.save(os.path.join(my_precomp_dir, '{}_mask.npy'.format(outName)), mask)
    np.save(os.path.join(my_precomp_dir, '{}_list_indices.npy'.format(outName)), neigh_indices)
    np.save(os.path.join(my_precomp_dir, '{}_iface_labels.npy'.format(outName)), iface_labels)
    # Save x, y, z
    np.save(os.path.join(my_precomp_dir, '{}_X.npy'.format(outName)), verts[:, 0])
    np.save(os.path.join(my_precomp_dir, '{}_Y.npy'.format(outName)), verts[:, 1])
    np.save(os.path.join(my_precomp_dir, '{}_Z.npy'.format(outName)), verts[:, 2])


# def precomputeProteinPlyInfo2(masifpniOpts, pPlyFile, pdb_id, pChain, naChain):
#     # ply_file = masifpniOpts['ply_file_template'].format(pdb_id, pChain)
#     if not os.path.exists(pPlyFile): return
#
#     params = masifpniOpts['masifpni_site']
#
#     try:
#         rho, theta, mask, neigh_indices, verts = read_data_from_surface1(pPlyFile, params)
#     except Exception as e:
#         print(e)
#         return
#
#     my_precomp_dir = os.path.join(params['masif_precomputation_dir'], pdb_id)
#     resolveDir(my_precomp_dir, chdir=False)
#     np.save(os.path.join(my_precomp_dir, pChain + '_rho_wrt_center.npy'), rho)
#     np.save(os.path.join(my_precomp_dir, pChain + '_theta_wrt_center.npy'), theta)
#     # np.save(os.path.join(my_precomp_dir, pChain + '_input_feat.npy'), input_feat)
#     np.save(os.path.join(my_precomp_dir, pChain + '_mask.npy'), mask)
#     np.save(os.path.join(my_precomp_dir, pChain + '_list_indices.npy'), neigh_indices)
#     # np.save(os.path.join(my_precomp_dir, pChain + '_iface_labels.npy'), iface_labels)
#     # Save x, y, z
#     np.save(os.path.join(my_precomp_dir, pChain + '_X.npy'), verts[:, 0])
#     np.save(os.path.join(my_precomp_dir, pChain + '_Y.npy'), verts[:, 1])
#     np.save(os.path.join(my_precomp_dir, pChain + '_Z.npy'), verts[:, 2])


# def rebuildProtFeatWithNa(masifpniOpts, pPlyFile, naPlyFile, pdb_id, pChain, naChain, addNaFeat=False):
#     if not os.path.exists(pPlyFile): return
#
#     params = masifpniOpts['masifpni_site']
#
#     if addNaFeat:
#         try:
#             input_feat, iface_labels = rebuildInputFeat(pPlyFile, params, naPlyFile=naPlyFile)
#         except Exception as e:
#             print(e)
#             return
#
#         my_precomp_dir = os.path.join(params['masif_precomputation_dir'], pdb_id)
#         resolveDir(my_precomp_dir, chdir=False)
#
#         np.save(os.path.join(my_precomp_dir, '{}_{}_input_feat.npy'.format(pChain, naChain)), input_feat)
#         np.save(os.path.join(my_precomp_dir, '{}_iface_labels.npy'.format(pChain)), iface_labels)
#     else:
#         try:
#             input_feat, iface_labels = rebuildInputFeat(pPlyFile, params)
#         except Exception as e:
#             print(e)
#             return
#
#         my_precomp_dir = os.path.join(params['masif_precomputation_dir'], pdb_id)
#         resolveDir(my_precomp_dir, chdir=False)
#
#         np.save(os.path.join(my_precomp_dir, '{}_input_feat.npy'.format(pChain)), input_feat)
#         np.save(os.path.join(my_precomp_dir, '{}_{}_iface_labels.npy'.format(pChain, naChain)), iface_labels)


