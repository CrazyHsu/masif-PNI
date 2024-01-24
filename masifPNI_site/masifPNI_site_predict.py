#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
File name: masifPNI_site_predict.py
Author: CrazyHsu @ crazyhsu9627@gmail.com
Created on: 2022-09-12 16:42:58
Last modified: 2022-09-12 16:42:58
'''

import time, os, sys, importlib, glob
import numpy as np
import pandas as pd
import pymesh

from Bio.PDB import PDBList
from collections import namedtuple
from multiprocessing import Pool, JoinableQueue
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import precision_recall_curve
from Bio.PDB import StructureBuilder, PDBParser, Selection, PDBIO, MMCIFIO
from Bio.PDB.Polypeptide import three_to_one

from commonFuncs import *
from parseConfig1 import DefaultConfig, GlobalVars
from pdbDownload import targetPdbDownload
from precompute import precomputeProteinPlyInfo1
from dataPreparation import dataprepFromList3, extractSurfaceInfo
from inputOutputProcess import NotDisordered, find_modified_amino_acids
from readDataFromSurface import extractProteinTriangulate1
from masifPNI_site.masifPNI_site_train import pad_indices
from masifPNI_site.masifPNI_site_nn import MasifPNI_site_nn


def getProteinChains(pdbFile):
    from inputOutputProcess import PROTEIN_LETTERS
    ppdb = PandasPdb()
    pdbStruc = ppdb.read_pdb(pdbFile)
    atomDf = pdbStruc.df["ATOM"]

    pChains = atomDf[atomDf["residue_name"].isin(PROTEIN_LETTERS)]["chain_id"].tolist()
    pChains = list(set(pChains))
    return [pdbFile, pChains]


def charge_color(charges):
    # Assume a std deviation equal for all proteins....
    max_val = 1.0
    min_val = -1.0

    norm_charges = charges
    blue_charges = np.array(norm_charges)
    red_charges = np.array(norm_charges)
    blue_charges[blue_charges < 0] = 0
    red_charges[red_charges > 0] = 0
    red_charges = abs(red_charges)
    red_charges[red_charges > max_val] = max_val
    blue_charges[blue_charges < min_val] = min_val
    red_charges = red_charges / max_val
    blue_charges = blue_charges / max_val
    # red_charges[red_charges>1.0] = 1.0
    # blue_charges[blue_charges>1.0] = 1.0
    green_color = np.array([0.0] * len(charges))
    mycolor = [
        [
            0.9999 - blue_charges[i],
            0.9999 - (blue_charges[i] + red_charges[i]),
            0.9999 - red_charges[i],
        ]
        for i in range(len(charges))
    ]
    for i in range(len(mycolor)):
        for k in range(3):
            if mycolor[i][k] < 0:
                mycolor[i][k] = 0

    return [[round(item[0]*255), round(item[1]*255), round(item[2]*255)] for item in mycolor]


def iface_color(iface):
    # max value is 1, min values is 0
    hp = iface.copy()
    hp = (hp * 2 - 1) * -1
    mycolor = charge_color(hp)
    return mycolor


def change_bfactor_with_pred_prob(origin_pdb_file, aa_pred_prob_df, rescored_pdb_file):
    struc = PDBParser().get_structure('target', origin_pdb_file)
    model = struc[0]

    # c = model.child_list[0]  # First "chain"
    # score_dict = pd.Series(aa_pred_prob_df.iloc[:, 5].values, index=aa_pred_prob_df.iloc[:, 1]).to_dict()
    score_dict = aa_pred_prob_df.groupby('Chain')[['Position', 'Prediction probability']].apply(
        lambda x: x.set_index('Position').to_dict(orient='index')).to_dict()

    detach_chains = []
    for c in model.child_list:
        if c.id not in score_dict:
            detach_chains.append(c.id)
            continue
        for res in c.child_list:
            het, idx, icode = res.id
            new_bfac = 0

            if idx in score_dict[c.id].keys():
                new_bfac = round(score_dict[c.id][idx]["Prediction probability"]*100, 3)

            for atom in res.child_list:
                atom.set_bfactor(new_bfac)

    if detach_chains:
        for c in detach_chains:
            model.detach_child(c)

    out = open(rescored_pdb_file, 'w')
    io = PDBIO()
    io.set_structure(struc)
    io.save(out)
    out.close()


def convert_pdb_to_cif(pdb_file, aa_pred_prob_df, out_cif=None):
    import gemmi
    if not out_cif:
        filename, _ = os.path.splitext(pdb_file)
        out_cif = f"{filename}.cif"

    structure = gemmi.read_pdb(pdb_file)
    structure.setup_entities()
    structure.assign_label_seq_id()
    score_dict = aa_pred_prob_df.groupby('Chain')[['Position', 'Prediction probability']].apply(
        lambda x: x.set_index('Position').to_dict(orient='index')).to_dict()
    pChains = []
    for entity in structure.entities:
        if entity.polymer_type.__str__() == "PolymerType.PeptideL":
            pChains.extend(entity.subchains)

    ma_qa_metric_local_loop_list = []
    metric_id = 2
    for model in structure:
        detach_chains = []
        for subchain in model.subchains():
            subchain_id = subchain.subchain_id()
            if subchain_id in pChains:
                subchain_id_strip = subchain_id.strip("xp")
                if subchain_id_strip in score_dict:
                    for res in subchain:
                        pred_score = round(score_dict[subchain_id_strip][res.seqid.num]["Prediction probability"]*100, 3)
                        ma_qa_metric_local_loop_list.append(
                            [f"{res.subchain}", f"{res.name}", f"{res.seqid.num}", f"{metric_id}", f"{pred_score}",
                             f"{model.name}", f"{res.seqid.num}"])
                else:
                    detach_chains.append(subchain_id_strip)

        # print(detach_chains)
        if detach_chains:
            for c in detach_chains:
                model.remove_chain(c)

    block = structure.make_mmcif_block()
    block.find_mmcif_category('_chem_comp.').erase()

    ma_qa_metric_prefix = '_ma_qa_metric.'
    ma_qa_metric_local_prefix = '_ma_qa_metric_local.'
    ma_qa_metric_global_prefix = '_ma_qa_metric_global.'

    ma_qa_metric_attributes = ["id", "mode", "name", "software_group_id", "type"]
    ma_qa_metric_local_attributes = ["label_asym_id", "label_comp_id", "label_seq_id", "metric_id", "metric_value",
                                     "model_id", "ordinal_id"]
    ma_qa_metric_global_attributes = ["metric_id", "metric_value", "model_id", "ordinal_id"]

    ma_qa_metric_loop = block.init_loop(ma_qa_metric_prefix, ma_qa_metric_attributes)
    ma_qa_metric_loop.add_row(["1", "global", "pLDDT", "1", "pLDDT"])
    ma_qa_metric_loop.add_row(["2", "local", "pLDDT", "1", "pLDDT"])

    ma_qa_metric_local_loop = block.init_loop(ma_qa_metric_local_prefix, ma_qa_metric_local_attributes)
    for i in ma_qa_metric_local_loop_list:
        ma_qa_metric_local_loop.add_row(i)
    block.write_file(out_cif)


def extractProteinPDB(infilename, outfilename, chainIds=None):
    """
    extractPDB: Extract selected chains from a PDB and save the extracted chains to an output file.
    Pablo Gainza - LPDI STI EPFL 2019
    Released under an Apache License 2.0
    """
    # extract the chain_ids from infilename and save in outfilename.
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure(infilename, infilename)
    model = Selection.unfold_entities(struct, "M")[0]
    chains = Selection.unfold_entities(struct, "C")
    # Select residues to extract and build new structure
    structBuild = StructureBuilder.StructureBuilder()
    structBuild.init_structure("output")
    structBuild.init_seg(" ")
    structBuild.init_model(0)
    outputStruct = structBuild.get_structure()

    modified_amino_acids = find_modified_amino_acids(infilename)

    for chain in model:
        if chainIds == None or chain.get_id() in chainIds:
            structBuild.init_chain(chain.get_id())
            for residue in chain:
                het = residue.get_id()
                if het[0] == " ":
                    outputStruct[0][chain.get_id()].add(residue)
                elif het[0][-3:] in modified_amino_acids:
                    outputStruct[0][chain.get_id()].add(residue)

    # Output the selected residues
    pdbio = PDBIO()
    pdbio.set_structure(outputStruct)
    pdbio.save(outfilename, select=NotDisordered(), preserve_atom_numbering=True)


def processCustomPDBs(validPdbFiles, masifpniOpts):
    batchRunFlag = False if masifpniOpts["no_batch_run"] else True
    n_threads = masifpniOpts["n_threads"]
    desc = "Get all protein chains in custom PDB files"
    resultList = batchRun1(getProteinChains, validPdbFiles, n_threads=n_threads, desc=desc, batchRunFlag=batchRunFlag)

    pPdbDir = os.path.join(masifpniOpts["extract_pdb"], "protein")
    resolveDir(pPdbDir, chdir=False)
    pniChainPairs = []
    extractBatchRun = []
    precomputeProteinPlyInfoBatchRun = []
    extractProteinTriangulateBatchRun = []
    for i in resultList:
        pdb_id = os.path.basename(i[0]).split(".")[0]
        pChain = "".join(sorted("".join(i[1])))
        pPdbFile = os.path.join(pPdbDir, "{}_{}.pdb".format(pdb_id, pChain))
        pPlyFile = masifpniOpts['ply_file_template'].format(pdb_id, pChain, "")
        pVerticeFile = masifpniOpts['vertex_file_template'].format(pdb_id, pChain, "")
        resolveDir(os.path.dirname(pPdbFile), chdir=False)
        resolveDir(os.path.dirname(pPlyFile), chdir=False)
        resolveDir(os.path.dirname(pVerticeFile), chdir=False)

        extractBatchRun.append([i[0], pPdbFile, i[1]])
        extractProteinTriangulateBatchRun.append((pPdbFile, "", pPlyFile, pVerticeFile, False))
        precomputeProteinPlyInfoBatchRun.append((pPlyFile, pdb_id, pChain, ""))
        pniChainPairs.append([pdb_id, pChain, "", "", 0, 0])

    extractProteinTriangulateBatchRun = [(masifpniOpts,) + i for i in set(extractProteinTriangulateBatchRun)]
    precomputeProteinPlyInfoBatchRun = [(masifpniOpts,) + i for i in set(precomputeProteinPlyInfoBatchRun)]

    extractBatchRun = [i for i in extractBatchRun if not os.path.exists(i[1])]
    extractProteinTriangulateBatchRun = [i for i in extractProteinTriangulateBatchRun if not os.path.exists(i[3])]
    tmpList = []
    for i in precomputeProteinPlyInfoBatchRun:
        my_precomp_dir = os.path.join(masifpniOpts["masifpni_site"]['masif_precomputation_dir'], i[2])
        name = "{}_{}_{}".format(i[2], i[3], i[4])
        feat_file = os.path.join(my_precomp_dir, '{}_input_feat.npy'.format(name))
        if not os.path.exists(feat_file):
            tmpList.append(i)
    precomputeProteinPlyInfoBatchRun = tmpList

    desc = "Extract protein PDB files"
    batchRun1(extractProteinPDB, extractBatchRun, n_threads=n_threads, desc=desc, batchRunFlag=batchRunFlag)

    desc = "Extract protein triangulate"
    batchRun1(extractProteinTriangulate1, extractProteinTriangulateBatchRun, n_threads=n_threads, desc=desc,
              batchRunFlag=batchRunFlag)

    desc = "Precompute protein ply information"
    batchRun1(precomputeProteinPlyInfo1, precomputeProteinPlyInfoBatchRun, n_threads=n_threads, desc=desc,
              batchRunFlag=batchRunFlag)

    pniChainPairsDf = pd.DataFrame(pniChainPairs, columns=["PDB_id", "pChain", "naChain", "naType", "pChainLen", "naChainLen"])
    return pniChainPairsDf

# Run masif site on a protein, on a previously trained network.
def run_masif_site(params, learning_obj, rho_wrt_center, theta_wrt_center, input_feat, mask, indices):
    indices = pad_indices(indices, mask.shape[1])
    mask = np.expand_dims(mask, 2)
    feed_dict = {
        learning_obj.rho_coords: rho_wrt_center,
        learning_obj.theta_coords: theta_wrt_center,
        learning_obj.input_feat: input_feat,
        learning_obj.mask: mask,
        learning_obj.indices_tensor: indices,
    }

    score = learning_obj.session.run([learning_obj.full_score], feed_dict=feed_dict)
    return score


def eval_aa_level(vertice2nameFile, pred_npy_file, pred_residue_score_txt):
    vertice2name = np.load(vertice2nameFile, allow_pickle=True)
    pred_prob = np.load(pred_npy_file, allow_pickle=True)[0]
    names = vertice2name["names"]
    tmpList = []
    for i in names:
        tmpList.append(i.split("_"))
    ifaceLabelsDf = pd.DataFrame(tmpList, columns=["chain", "position", "x", "residue", "atom", "color"])
    ifaceLabelsDf = ifaceLabelsDf.astype({'position': 'int32'})
    ifaceLabelsDf["pred_prob"] = pred_prob
    ifaceLabelsDf = ifaceLabelsDf.sort_values('position')
    ifaceLabelsDf_agg = ifaceLabelsDf.groupby(["chain", "position"]).agg(
        {"residue": "first", "atom": "count", "pred_prob": "median"})
    ifaceLabelsDf_agg["aa_one_letter"] = ifaceLabelsDf_agg.residue.apply(three_to_one)
    ifaceLabelsDf_agg = ifaceLabelsDf_agg.reset_index()

    add_list = []
    for c in ifaceLabelsDf_agg.chain.unique():
        tmp_df = ifaceLabelsDf_agg[ifaceLabelsDf_agg.chain == c]
        add_list.extend([(c, i, "XXX", 0, 0, "X") for i in range(1, np.max(tmp_df.position) + 1) if
                         i not in tmp_df.position.values])
    add_df = pd.DataFrame(add_list, columns=ifaceLabelsDf_agg.columns)
    ifaceLabelsDf_final = pd.concat([ifaceLabelsDf_agg, add_df]).sort_values('position')
    ifaceLabelsDf_final.pred_prob = ifaceLabelsDf_final.pred_prob.round(3)
    ifaceLabelsDf_final = ifaceLabelsDf_final[["chain", "position", "residue", "aa_one_letter", "atom", "pred_prob"]]
    ifaceLabelsDf_final.columns = ["Chain", "Position", "Residue name", "Residue abbr.", "Atom count", "Prediction probability"]
    ifaceLabelsDf_final.to_csv(pred_residue_score_txt, index=False)
    return ifaceLabelsDf_final


def print_eval_result(name, ground_truth, scores, out_file):
    out_handle = open(out_file, "w")
    # print("All label number {}, true label {}, pred label {}".format(len(ground_truth), sum(ground_truth), sum(np.round(scores[0]))))
    out_handle.write("All label number {}, true label {}, pred label {}\n".format(len(ground_truth), sum(ground_truth), sum(np.round(1 * (scores[0] >= 0.7)))))

    # scores[0] = np.abs(np.ceil(scores[0] - 0.6)).astype('int')
    roc_auc = roc_auc_score(ground_truth, scores[0])
    precision, recall, thresholds = precision_recall_curve(ground_truth, scores[0])
    prauc = auc(recall, precision)

    out_handle.write("ROC AUC score for protein {} : {:.2f} \n".format(name, roc_auc))
    out_handle.write("PR AUC score for protein {} : {:.2f} \n".format(name, prauc))

    #        pred_acc = np.round(scores[0])
    pred_acc = np.round(1 * (scores[0] >= 0.7))
    tn, fp, fn, tp = confusion_matrix(ground_truth, pred_acc).ravel()
    sn = tp / (tp + fn)
    out_handle.write("SN score for protein {} : {:.2f} \n".format(name, sn))

    sp = tn / (tn + fp)
    out_handle.write("SP score for protein {} : {:.2f} \n".format(name, sp))

    acc = accuracy_score(ground_truth, pred_acc)
    out_handle.write("ACC score for protein {} : {:.2f} \n".format(name, acc))

    mcc = matthews_corrcoef(ground_truth, pred_acc)
    out_handle.write("MCC score for protein {} : {:.2f} \n".format(name, mcc))

    ppv = tp / (tp + fp)
    out_handle.write("Precision score for protein {} : {:.2f} \n".format(name, ppv))

    # F1 score - harmonic mean of precision and recall [2*tp/(2*tp + fp + fn)]
    f1 = 2 * ppv * sn / (ppv + sp)
    out_handle.write("F1 score for protein {} : {:.2f} \n".format(name, f1))
    out_handle.close()
    return roc_auc, prauc, sn, sp, acc, mcc, ppv, f1


"""
masif_site_predict: Evaluate one or multiple proteins on MaSIF-site. 
Pablo Gainza - LPDI STI EPFL 2019
This file is part of MaSIF.
Released under an Apache License 2.0
"""


def masifPNI_site_predict(argv):
    from parseConfig1 import check_params_changed_from_command_line
    masifpniOpts = mergeParams1(argv)
    argv_status = check_params_changed_from_command_line(argv)
    # print(masifpniOpts)
    # print(argv_status)
    params = masifpniOpts["masifpni_site"]
    GlobalVars().setEnviron()

    if masifpniOpts["use_gpu"]:
        idx_xpu = masifpniOpts["gpu_dev"] if masifpniOpts["gpu_dev"] else "/gpu:0"
    else:
        idx_xpu = masifpniOpts["cpu_dev"] if masifpniOpts["cpu_dev"] else "/cpu:0"

    # Set precomputation dir.
    # parent_in_dir = params["masif_precomputation_dir"]
    # precomputatedIds = os.listdir(parent_in_dir)
    eval_list = []
    validPdbFiles = []

    if masifpniOpts["pdb_id_list"]:
        eval_list = [i.upper() for i in masifpniOpts["pdb_id_list"].split(",")]
    if masifpniOpts["pdb_id_file"]:
        eval_list = getListFromFile(masifpniOpts["pdb_id_file"])
        eval_list = [i.upper() for i in eval_list]
    if masifpniOpts["custom_pdb"]:
        pdbFiles = masifpniOpts["custom_pdb"].split(",")
        for i in pdbFiles:
            if os.path.isfile(i) and (i.endswith("pdb") or i.endswith("pdb.gz")):
                validPdbFiles.append((i,))
        if not validPdbFiles:
            for pdbFile in pdbFiles:
                with open(pdbFile) as f:
                    for i in f.readlines():
                        if os.path.isfile(i.strip()) and (i.strip().endswith("pdb") or i.strip().endswith("pdb.gz")):
                            validPdbFiles.append((i.strip(),))

    if len(eval_list) == 0 and len(validPdbFiles) == 0:
        print("Please input the PDB ids or PDB files that you want to evaluate.")
        return

    batchRunFlag = False if masifpniOpts["no_batch_run"] else True
    naType = masifpniOpts["na_type"]
    mergeNaChain = masifpniOpts["merge_na_chain"]
    if eval_list:
        evalPniChainPairsDf = dataprepFromList3(eval_list, masifpniOpts, batchRunFlag=batchRunFlag, naType=naType)
        # evalPniChainPairsDf = dataprepFromList3(eval_list, masifpniOpts, batchRunFlag=argv.nobatchRun, naType=argv.naType)

        if mergeNaChain:
            keys = ["PDB_id", "pChain", "naType"]
            aggregations = {"naChain": "".join, "pChainLen": np.mean, "naChainLen": np.mean}
            evalPniChainPairsDf = evalPniChainPairsDf.groupby(keys).agg(aggregations).reset_index()
            evalPniChainPairsDf['pChain'] = evalPniChainPairsDf['pChain'].apply(lambda x: ''.join(sorted(x)))
            evalPniChainPairsDf['naChain'] = evalPniChainPairsDf['naChain'].apply(lambda x: ''.join(sorted(x)))
    elif validPdbFiles:
        evalPniChainPairsDf = processCustomPDBs(validPdbFiles, masifpniOpts)
        keys = ["PDB_id"]
        aggregations = {"pChain": "".join, "naChain": "first", "naType": "first", "pChainLen": np.mean, "naChainLen": np.mean}
        evalPniChainPairsDf = evalPniChainPairsDf.groupby(keys).agg(aggregations).reset_index()
        evalPniChainPairsDf['pChain'] = evalPniChainPairsDf['pChain'].apply(lambda x: ''.join(sorted(x)))
        # evalPniChainPairsDf['naChain'] = evalPniChainPairsDf['naChain'].apply(lambda x: ''.join(sorted(x)))
    else:
        return

    # print(evalPniChainPairsDf)
    # return
    # Build the neural network model
    learning_obj = MasifPNI_site_nn(
        params["max_distance"],
        n_thetas=params["n_theta"],
        n_rhos=params["n_rho"],
        n_rotations=params["n_rotations"],
        idx_xpu=idx_xpu,
        feat_mask=params["n_feat"] * [1.0],
        n_conv_layers=params["n_conv_layers"],
    )
    if params["model_dir"]:
        params["model_dir"] = os.path.realpath(params["model_dir"])
    print("Restoring model from: " + os.path.join(params["model_dir"], "model"))
    # params["model_dir"] = "/data2/xufeng_333/masifpni_nucleicnet_20221107/site/nn_models"
    learning_obj.saver.restore(learning_obj.session, os.path.join(params["model_dir"], "model"))

    resolveDirs([params["out_pred_dir"], params["out_surf_dir"], params["out_eval_dir"]])

    out_list = []
    for _, row in evalPniChainPairsDf.iterrows():
        pdb_id = row.PDB_id
        pChain = row.pChain
        naChain = row.naChain
        name = "{}_{}_{}".format(pdb_id, pChain, naChain)
        mydir = os.path.join(params["masif_precomputation_dir"], pdb_id)
        iface_file = os.path.join(mydir, "{}_iface_labels.npy".format(name))
        if not os.path.exists(iface_file):
            tmpDf = pd.DataFrame([row])
            extractSurfaceInfo(masifpniOpts, tmpDf, batchRunFlag=batchRunFlag, mergeNaChain=mergeNaChain)
            # extractSurfaceInfo(masifpniOpts, tmpDf, batchRunFlag=argv.nobatchRun, mergeNaChain=argv.mergeNaChain)
            if not os.path.exists(iface_file): continue

        print("\nEvaluating protein chain {} with nucleic acid chain {} in protein {}".format(pChain, naChain, pdb_id))

        rho_wrt_center = np.load(os.path.join(mydir, "{}_rho_wrt_center.npy".format(name)))
        theta_wrt_center = np.load(os.path.join(mydir, "{}_theta_wrt_center.npy".format(name)))
        input_feat = np.load(os.path.join(mydir, "{}_input_feat.npy".format(name)))
        input_feat = mask_input_feat(input_feat, params["n_feat"] * [1.0])
        mask = np.load(os.path.join(mydir, "{}_mask.npy".format(name)))
        indices = np.load(os.path.join(mydir, "{}_list_indices.npy".format(name)), encoding="latin1", allow_pickle=True)
        labels = np.zeros((len(mask)))

        print("Total number of patches:{}".format(len(mask)))

        tic = time.time()
        scores = run_masif_site(
            params,
            learning_obj,
            rho_wrt_center,
            theta_wrt_center,
            input_feat,
            mask,
            indices,
        )
        toc = time.time()
        print("Total number of patches for which scores were computed: {}".format(len(scores[0])))
        print("GPU time (real time, not actual GPU time): {:.3f}s\n".format(toc - tic))
        npy_file = os.path.join(params["out_pred_dir"], "pred_{}.npy".format(name))
        np.save(npy_file, scores)

        ply_file = masifpniOpts["ply_file_template"].format(pdb_id, pChain, naChain)
        vertices_name_file = masifpniOpts["vertex_file_template"].format(pdb_id, pChain, naChain)
        pred_residue_score_txt = os.path.join(params["out_pred_dir"], "pred_{}_residue_scores.csv".format(name))
        mymesh = pymesh.load_mesh(ply_file)
        try:
            ground_truth = mymesh.get_attribute('vertex_iface')
        except:
            pred_ply = os.path.join(params["out_surf_dir"], "pred_{}.ply".format(name))
            color_array_surf = np.array(iface_color(scores[0]))
            mymesh.add_attribute("iface")
            mymesh.add_attribute("red")
            mymesh.add_attribute("green")
            mymesh.add_attribute("blue")
            mymesh.set_attribute("iface", scores[0])
            mymesh.set_attribute("red", color_array_surf[:, 0])
            mymesh.set_attribute("green", color_array_surf[:, 1])
            mymesh.set_attribute("blue", color_array_surf[:, 2])
            mymesh.remove_attribute("vertex_x")
            mymesh.remove_attribute("vertex_y")
            mymesh.remove_attribute("vertex_z")
            mymesh.remove_attribute("face_vertex_indices")

            pymesh.save_mesh(pred_ply, mymesh, *mymesh.get_attribute_names(), use_float=True, ascii=True)
            aa_pred_prob_df = eval_aa_level(vertices_name_file, npy_file, pred_residue_score_txt)

            origin_pdb_file = os.path.join(masifpniOpts["extract_pdb"], "protein", "{}_{}.pdb".format(pdb_id, pChain))
            rescored_pdb_file = os.path.join(params["out_surf_dir"], "pred_{}.pdb".format(name))
            rescored_cif_file = os.path.join(params["out_surf_dir"], "pred_{}.cif".format(name))
            change_bfactor_with_pred_prob(origin_pdb_file, aa_pred_prob_df, rescored_pdb_file)
            convert_pdb_to_cif(origin_pdb_file, aa_pred_prob_df, rescored_cif_file)
            continue

        tmp_eval_file = os.path.join(params["out_eval_dir"], "eval_result.{}.txt".format(name))
        roc_auc, prauc, sn, sp, acc, mcc, ppv, f1 = print_eval_result(name, ground_truth, scores, tmp_eval_file)
        out_list.append([name, sum(ground_truth), sum(np.round(scores[0])), np.round(roc_auc, 2), np.round(prauc, 2),
                         np.round(sn, 2), np.round(sp, 2), np.round(acc, 2), np.round(mcc, 2), np.round(ppv, 2),
                         np.round(f1, 2)])

        mymesh.remove_attribute("vertex_iface")
        pred_ply = os.path.join(params["out_surf_dir"], "pred_{}.ply".format(name))
        color_array_surf = np.array(iface_color(scores[0]))
        mymesh.add_attribute("iface")
        mymesh.add_attribute("red")
        mymesh.add_attribute("green")
        mymesh.add_attribute("blue")
        mymesh.set_attribute("iface", scores[0])
        mymesh.set_attribute("red", color_array_surf[:, 0])
        mymesh.set_attribute("green", color_array_surf[:, 1])
        mymesh.set_attribute("blue", color_array_surf[:, 2])
        mymesh.remove_attribute("vertex_x")
        mymesh.remove_attribute("vertex_y")
        mymesh.remove_attribute("vertex_z")
        mymesh.remove_attribute("face_vertex_indices")

        pymesh.save_mesh(pred_ply, mymesh, *mymesh.get_attribute_names(), use_float=True, ascii=True)
        aa_pred_prob_df = eval_aa_level(vertices_name_file, npy_file, pred_residue_score_txt)

        origin_pdb_file = os.path.join(masifpniOpts["extract_pdb"], "protein", "{}_{}.pdb".format(pdb_id, pChain))
        rescored_pdb_file = os.path.join(params["out_surf_dir"], "pred_{}.pdb".format(name))
        rescored_cif_file = os.path.join(params["out_surf_dir"], "pred_{}.cif".format(name))
        change_bfactor_with_pred_prob(origin_pdb_file, aa_pred_prob_df, rescored_pdb_file)
        convert_pdb_to_cif(origin_pdb_file, aa_pred_prob_df, rescored_cif_file)

    colNames = ["name", "true_label", "pred_label", "roc_auc", "pr_auc", "sn", "sp", "acc", "mcc", "prc", "f1-score"]
    outFile = os.path.join(params["out_eval_dir"], "all_eval_result.txt")
    out_df = pd.DataFrame(out_list, columns=colNames)
    out_df.to_csv(outFile, sep="\t", index=False)


if __name__ == '__main__':
    masifPNI_site_predict(sys.argv)
