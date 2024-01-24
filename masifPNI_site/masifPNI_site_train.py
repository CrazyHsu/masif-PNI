#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
File name: masifPNI_site_train.py
Author: CrazyHsu @ crazyhsu9627@gmail.com
Created on: 2022-09-12 16:07:49
Last modified: 2022-09-12 16:07:49
'''

import os, sys, time, glob
import numpy as np
import pandas as pd

from commonFuncs import *
from Bio.PDB import *
from sklearn import metrics
# from inputOutputProcess import getPdbChainLength, findProteinChainBoundNA2
from dataPreparation import dataprepFromList3
from masifPNI_site.masifPNI_site_nn import MasifPNI_site_nn, MasifPNI_site_nn1

"""
masifPNI_site_train.py: Entry function to train MaSIF-site.
Pablo Gainza - LPDI STI EPFL 2019
This file is part of MaSIF.
Released under an Apache License 2.0
"""


def pad_indices(indices, max_verts):
    padded_ix = np.zeros((len(indices), max_verts), dtype=int)
    for patch_ix in range(len(indices)):
        padded_ix[patch_ix] = np.concatenate([indices[patch_ix], [patch_ix] * (max_verts - len(indices[patch_ix]))])
    return padded_ix


# def compute_roc_auc(pos, neg):
#     labels = np.concatenate([np.ones((len(pos))), np.zeros((len(neg)))])
#     dist_pairs = np.concatenate([pos, neg])
#     return metrics.roc_auc_score(labels, dist_pairs)


# def getPdbidChainPairs(masifpniOpts, fromFile=None, fromList=None, fromCustomPDB=None, batchRunFlag=False):
#     myList = []
#     pdbl = PDBList(server='http://ftp.wwpdb.org')
#     targetPdbDownloadBatchRun = []
#     if fromFile:
#         with open(fromFile) as f:
#             for line in f.readlines():
#                 if line.startswith("#"): continue
#                 fields = line.strip().split("_")
#                 pdb_id = fields[0].upper()
#                 pdbFile = os.path.join(masifpniOpts["raw_pdb_dir"], pdb_id + ".pdb")
#                 if not os.path.exists(pdbFile):
#                     targetPdbDownloadBatchRun.append((masifpniOpts, pdb_id, pdbl, True))
#                 if len(fields) == 1:
#                     myList.append([pdb_id, pdbFile])
#                 elif len(fields) == 2:
#                     myList.append([pdb_id, pdbFile, fields[1]])
#                 else:
#                     myList.append([pdb_id, pdbFile, fields[1], fields[2]])
#
#     if fromList:
#         for l in fromList:
#             fields = l.strip().split("_")
#             pdb_id = fields[0].upper()
#             pdbFile = os.path.join(masifpniOpts["raw_pdb_dir"], pdb_id + ".pdb")
#             if not os.path.exists(pdbFile):
#                 targetPdbDownloadBatchRun.append((masifpniOpts, pdb_id, pdbl, True))
#             if len(fields) == 1:
#                 myList.append([pdb_id, pdbFile])
#             elif len(fields) == 2:
#                 myList.append([pdb_id, pdbFile, fields[1]])
#             else:
#                 myList.append([pdb_id, pdbFile, fields[1], fields[2]])
#
#     if fromCustomPDB:
#         with open(fromCustomPDB) as f:
#             for pdbFile in f.readlines():
#                 if pdbFile.startswith("#"): continue
#                 fields = os.path.basename(pdbFile).split("_")
#                 pdb_id = fields[0].upper()
#                 pdbFile = pdbFile.strip()
#                 if not os.path.exists(pdbFile):
#                     targetPdbDownloadBatchRun.append((masifpniOpts, pdb_id, pdbl, True))
#                 if len(fields) == 1:
#                     myList.append([pdb_id, pdbFile])
#                 elif len(fields) == 2:
#                     myList.append([pdb_id, pdbFile, fields[1]])
#                 else:
#                     myList.append([pdb_id, pdbFile, fields[1], fields[2]])
#
#     desc = "Find protein-NA bounding chains"
#     resultList = batchRun1(findProteinChainBoundNA2, myList, n_threads=masifpniOpts["n_threads"], desc=desc,
#                            batchRunFlag=batchRunFlag)
#     pniChainPairs = list(itertools.chain.from_iterable(resultList))
#     pniChainPairsDf = pd.DataFrame(pniChainPairs, columns=["PDB_id", "pChain", "naChain", "naType"])
#     return pniChainPairsDf


def train_masifPNI_site2(argv):
    import random
    random.seed(25)
    masifpniOpts = mergeParams1(argv)
    # return
    params = masifpniOpts["masifpni_site"]
    if masifpniOpts["masifpni_site"]["training_list"]:
        params["training_list"] = masifpniOpts["masifpni_site"]["training_list"]
    if masifpniOpts["masifpni_site"]["testing_list"]:
        params["testing_list"] = masifpniOpts["masifpni_site"]["testing_list"]

    if masifpniOpts["use_gpu"]:
        idx_xpu = masifpniOpts["gpu_dev"] if masifpniOpts["gpu_dev"] else "/gpu:0"
    else:
        idx_xpu = masifpniOpts["cpu_dev"] if masifpniOpts["cpu_dev"] else "/cpu:0"

    # Open training list.

    list_training_loss = []
    list_training_auc = []
    list_validation_auc = []
    iter_time = []
    best_val_auc = 0

    out_dir = params["model_dir"]
    logfile = open(os.path.join(masifpniOpts["log_dir"], "nn_model.log.txt"), "w")
    for key in params:
        logfile.write("{}: {}\n".format(key, params[key]))

    batchRunFlag = False if masifpniOpts["no_batch_run"] else True
    naType = masifpniOpts["na_type"]
    training_list = getListFromFile(params["training_list"])
    trainingPniChainPairsDf = dataprepFromList3(training_list, masifpniOpts, batchRunFlag=batchRunFlag, naType=naType)

    testing_list = getListFromFile(params["testing_list"])
    testingPniChainPairsDf = dataprepFromList3(testing_list, masifpniOpts, batchRunFlag=batchRunFlag, naType=naType)

    mergeNaChain = masifpniOpts["merge_na_chain"]
    if mergeNaChain:
#        keys = ["PDB_id", "pChain", "naType"]
#        aggregations = {"naChain": "".join, "pChainLen": np.mean, "naChainLen": np.mean}
        keys = ["PDB_id", "naType"]
        aggregations = {"pChain": lambda s: max(s, key=len), "naChain": "".join, "pChainLen": np.mean, "naChainLen": np.mean}
        trainingPniChainPairsDf = trainingPniChainPairsDf.groupby(keys).agg(aggregations).reset_index()
        testingPniChainPairsDf = testingPniChainPairsDf.groupby(keys).agg(aggregations).reset_index()

        trainingPniChainPairsDf['pChain'] = trainingPniChainPairsDf['pChain'].apply(lambda x: ''.join(sorted(x)))
        trainingPniChainPairsDf['naChain'] = trainingPniChainPairsDf['naChain'].apply(lambda x: ''.join(sorted(x)))

        testingPniChainPairsDf['pChain'] = testingPniChainPairsDf['pChain'].apply(lambda x: ''.join(sorted(x)))
        testingPniChainPairsDf['naChain'] = testingPniChainPairsDf['naChain'].apply(lambda x: ''.join(sorted(x)))

    filterPdbByLen = False
    if filterPdbByLen:
        trainingPniChainPairsDf = trainingPniChainPairsDf[(trainingPniChainPairsDf.pChainLen >= 30) &
                                                          (trainingPniChainPairsDf.naChainLen >= 4)]
        testingPniChainPairsDf = testingPniChainPairsDf[(testingPniChainPairsDf.pChainLen >= 30) &
                                                        (testingPniChainPairsDf.naChainLen >= 4)]

    # data_dirs = list(set(training_list + testing_list))
    # print(data_dirs)
    # np.random.shuffle(data_dirs)
    # # data_dirs = data_dirs
    # n_val = len(data_dirs) // 10
    # val_dirs = set(data_dirs[(len(data_dirs) - n_val):])

    data_dirs = list(set(training_list + testing_list))
    np.random.shuffle(data_dirs)
    n_val = len(data_dirs) // 10
    n_val_training = int(n_val * (len(training_list) / len(data_dirs)))
    n_val_testing = int(n_val * (len(testing_list) / len(data_dirs)))
    val_dirs = np.concatenate([np.random.choice(training_list, n_val_training), np.random.choice(testing_list, n_val_testing)])

    if "n_theta" in params:
        learning_obj = MasifPNI_site_nn(
            params["max_distance"],
            n_thetas=params["n_theta"],
            n_rhos=params["n_rho"],
            n_rotations=params["n_rotations"],
            idx_xpu=idx_xpu,
            feat_mask=params["n_feat"] * [1.0],
            n_conv_layers=params["n_conv_layers"],
        )
    else:
        learning_obj = MasifPNI_site_nn(
            params["max_distance"],
            n_thetas=4,
            n_rhos=3,
            n_rotations=4,
            idx_xpu=idx_xpu,
            feat_mask=params["n_feat"] * [1.0],
            n_conv_layers=params["n_conv_layers"],
        )

    print(params["n_feat"] * [1.0])
    if masifpniOpts["masifpni_site"]["model_dir"]:
        params["model_dir"] = masifpniOpts["masifpni_site"]["model_dir"]
    if not os.path.exists(params["model_dir"]):
        os.makedirs(params["model_dir"])
    else:
        if len(os.listdir(params["model_dir"])) != 0:
            # Load existing network.
            print('Reading pre-trained network')
            learning_obj.saver.restore(learning_obj.session, os.path.join(params['model_dir'], 'model'))

    batch_size = 100
    num_iterations = 100

    stable_count = 0
    # for num_iter in [0]:
    for num_iter in range(num_iterations):
        # Start training epoch:
        list_training_loss = []
        list_training_auc = []
        list_val_auc = []
        list_val_pos_labels = []
        list_val_neg_labels = []
        list_val_names = []
        list_training_acc = []
        list_val_acc = []
        logfile.write("Starting epoch {}\n".format(num_iter))
        print("Starting epoch {}".format(num_iter))
        tic = time.time()
        all_training_labels = []
        all_training_scores = []
        all_val_labels = []
        all_val_scores = []
        all_test_labels = []
        all_test_scores = []
        count_proteins = {"training": {}, "validation": {}, "testing": {}}

        list_test_auc = []
        list_test_names = []
        list_test_acc = []
        all_test_labels = []
        all_test_scores = []

        for pdb_id in data_dirs:
            mydir = os.path.join(params["masif_precomputation_dir"], pdb_id)
            tmpPair = trainingPniChainPairsDf.loc[trainingPniChainPairsDf["PDB_id"] == pdb_id]
            if tmpPair.empty: continue
            for _, row in tmpPair.iterrows():
                pChain = row.pChain
                naChain = row.naChain
                name = "{}_{}_{}".format(pdb_id, pChain, naChain)
                try:
                    iface_labels = np.load(os.path.join(mydir, "{}_iface_labels.npy".format(name)))
                except:
                    continue

                if sum(iface_labels) > 8000: continue
                if np.sum(iface_labels) > 0.75 * len(iface_labels) or np.sum(iface_labels) < 30: continue

                rho_wrt_center = np.load(os.path.join(mydir, "{}_rho_wrt_center.npy".format(name)))
                theta_wrt_center = np.load(os.path.join(mydir, "{}_theta_wrt_center.npy".format(name)))
                input_feat = np.load(os.path.join(mydir, "{}_input_feat.npy".format(name)))

                if params["n_feat"] < 5:
                    input_feat = mask_input_feat(input_feat, params["n_feat"] * [1.0])
                mask = np.load(os.path.join(mydir, "{}_mask.npy".format(name)))
                mask = np.expand_dims(mask, 2)
                indices = np.load(os.path.join(mydir, "{}_list_indices.npy".format(name)), encoding="latin1", allow_pickle=True)
                indices = pad_indices(indices, mask.shape[1])
                tmp = np.zeros((len(iface_labels), 2))
                for i in range(len(iface_labels)):
                    if iface_labels[i] == 1:
                        tmp[i, 0] = 1
                    else:
                        tmp[i, 1] = 1
                iface_labels_dc = tmp
                logfile.flush()
                pos_labels = np.where(iface_labels == 1)[0]
                neg_labels = np.where(iface_labels == 0)[0]
                np.random.shuffle(neg_labels)
                np.random.shuffle(pos_labels)
                # Scramble neg idx, and only get as many as pos_labels to balance the training.
                if params["n_conv_layers"] == 1:
                    n = min(len(pos_labels), len(neg_labels))
                    n = min(n, batch_size // 2)
                    subset = np.concatenate([neg_labels[:n], pos_labels[:n]])

                    rho_wrt_center = rho_wrt_center[subset]
                    theta_wrt_center = theta_wrt_center[subset]
                    input_feat = input_feat[subset]
                    mask = mask[subset]
                    iface_labels_dc = iface_labels_dc[subset]
                    indices = indices[subset]
                    pos_labels = range(0, n)
                    neg_labels = range(n, n * 2)
                else:
                    n = min(len(pos_labels), len(neg_labels))
                    neg_labels = neg_labels[:n]
                    pos_labels = pos_labels[:n]

                feed_dict = {
                    learning_obj.rho_coords: rho_wrt_center,
                    learning_obj.theta_coords: theta_wrt_center,
                    learning_obj.input_feat: input_feat,
                    learning_obj.mask: mask,
                    learning_obj.labels: iface_labels_dc,
                    learning_obj.pos_idx: pos_labels,
                    learning_obj.neg_idx: neg_labels,
                    learning_obj.indices_tensor: indices,
                }

                if pdb_id in val_dirs:
                    logfile.write("Validating on PDB {} {} {}\n".format(pdb_id, pChain, naChain))
                    count_proteins["validation"].update({pdb_id: 1})
                    feed_dict[learning_obj.keep_prob] = 1.0
                    training_loss, score, eval_labels = learning_obj.session.run(
                        [
                            learning_obj.data_loss,
                            learning_obj.eval_score,
                            learning_obj.eval_labels,
                        ],
                        feed_dict=feed_dict,
                    )
                    auc = metrics.roc_auc_score(eval_labels[:, 0], score)
                    list_val_pos_labels.append(np.sum(iface_labels))
                    list_val_neg_labels.append(len(iface_labels) - np.sum(iface_labels))
                    list_val_auc.append(auc)
                    list_val_names.append(pdb_id)
                    all_val_labels = np.concatenate([all_val_labels, eval_labels[:, 0]])
                    all_val_scores = np.concatenate([all_val_scores, score])
                else:
                    logfile.write("Training on PDB {} {} {}\n".format(pdb_id, pChain, naChain))
                    count_proteins["training"].update({pdb_id: 1})
                    feed_dict[learning_obj.keep_prob] = 1.0
                    _, training_loss, norm_grad, score, eval_labels = learning_obj.session.run(
                        [
                            learning_obj.optimizer,
                            learning_obj.data_loss,
                            learning_obj.norm_grad,
                            learning_obj.eval_score,
                            learning_obj.eval_labels,
                        ],
                        feed_dict=feed_dict,
                    )
                    all_training_labels = np.concatenate([all_training_labels, eval_labels[:, 0]])
                    all_training_scores = np.concatenate([all_training_scores, score])
                    auc = metrics.roc_auc_score(eval_labels[:, 0], score)
                    list_training_auc.append(auc)
                    list_training_loss.append(np.mean(training_loss))
                logfile.flush()

        # Run testing cycle.
        for pdb_id in data_dirs:
            mydir = os.path.join(params["masif_precomputation_dir"], pdb_id)
            tmpPair = testingPniChainPairsDf.loc[testingPniChainPairsDf["PDB_id"] == pdb_id]
            if tmpPair.empty: continue
            for _, row in tmpPair.iterrows():
                pChain = row.pChain
                naChain = row.naChain
                name = "{}_{}_{}".format(pdb_id, pChain, naChain)
                try:
                    iface_labels = np.load(os.path.join(mydir, "{}_iface_labels.npy".format(name)))
                except:
                    continue

                if sum(iface_labels) > 8000: continue
                if np.sum(iface_labels) > 0.75 * len(iface_labels) or np.sum(iface_labels) < 30: continue
                count_proteins["testing"].update({pdb_id: 1})
                rho_wrt_center = np.load(os.path.join(mydir, "{}_rho_wrt_center.npy".format(name)))
                theta_wrt_center = np.load(os.path.join(mydir, "{}_theta_wrt_center.npy".format(name)))
                input_feat = np.load(os.path.join(mydir, "{}_input_feat.npy".format(name)))
                if params["n_feat"] < 5:
                    input_feat = mask_input_feat(input_feat, params["n_feat"] * [1.0])
                mask = np.load(os.path.join(mydir, "{}_mask.npy".format(name)))
                mask = np.expand_dims(mask, 2)
                indices = np.load(os.path.join(mydir, "{}_list_indices.npy".format(name)), encoding="latin1", allow_pickle=True)
                # indices is (n_verts x <30), it should be
                indices = pad_indices(indices, mask.shape[1])
                tmp = np.zeros((len(iface_labels), 2))
                for i in range(len(iface_labels)):
                    if iface_labels[i] == 1:
                        tmp[i, 0] = 1
                    else:
                        tmp[i, 1] = 1
                iface_labels_dc = tmp
                logfile.flush()
                pos_labels = np.where(iface_labels == 1)[0]
                neg_labels = np.where(iface_labels == 0)[0]

                feed_dict = {
                    learning_obj.rho_coords: rho_wrt_center,
                    learning_obj.theta_coords: theta_wrt_center,
                    learning_obj.input_feat: input_feat,
                    learning_obj.mask: mask,
                    learning_obj.labels: iface_labels_dc,
                    learning_obj.pos_idx: pos_labels,
                    learning_obj.neg_idx: neg_labels,
                    learning_obj.indices_tensor: indices,
                }

                feed_dict[learning_obj.keep_prob] = 1.0
                score = learning_obj.session.run([learning_obj.full_score], feed_dict=feed_dict)
                score = score[0]
                auc = metrics.roc_auc_score(iface_labels, score)
                list_test_auc.append(auc)
                list_test_names.append((pdb_id, pChain, naChain))
                all_test_labels.append(iface_labels)
                all_test_scores.append(score)

        count_proteins_num = len(list(itertools.chain.from_iterable([count_proteins[i].keys() for i in count_proteins])))
        if count_proteins_num == 0:
            print("Please check if you have prepare the label file!")
            break
        outstr = "Epoch ran on {} proteins, {} training, {} validation, {} testing\n".format(
            count_proteins_num, len(count_proteins["training"]),
            len(count_proteins["validation"]), len(count_proteins["testing"])
        )
        outstr += "Per protein AUC mean (training): {:.4f}; median: {:.4f} for iter {}\n".format(
            np.mean(list_training_auc), np.median(list_training_auc), num_iter
        )
        outstr += "Per protein AUC mean (validation): {:.4f}; median: {:.4f} for iter {}\n".format(
            np.mean(list_val_auc), np.median(list_val_auc), num_iter
        )
        outstr += "Per protein AUC mean (test): {:.4f}; median: {:.4f} for iter {}\n".format(
            np.mean(list_test_auc), np.median(list_test_auc), num_iter
        )
        flat_all_test_labels = np.concatenate(all_test_labels, axis=0)
        flat_all_test_scores = np.concatenate(all_test_scores, axis=0)
        outstr += "Testing auc (all points): {:.2f}\n".format(
            metrics.roc_auc_score(flat_all_test_labels, flat_all_test_scores)
        )
        outstr += "Epoch took {:2f}s\n".format(time.time() - tic)
        logfile.write(outstr + "\n")
        print(outstr)

        if np.mean(list_val_auc) > best_val_auc:
            logfile.write(">>> Saving model.\n")
            print(">>> Saving model.\n")
            best_val_auc = np.mean(list_val_auc)
            output_model = os.path.join(out_dir, "model")
            learning_obj.saver.save(learning_obj.session, output_model)
            # Save the scores for test.
            np.save(out_dir + ".test_labels.npy", all_test_labels)
            np.save(out_dir + ".test_scores.npy", all_test_scores)
            np.save(out_dir + ".test_names.npy", list_test_names)

            stable_count = 0

        stable_count += 1
        if stable_count > 10: break

    logfile.close()


# def train_masifPNI_site1(argv):
#     masifpniOpts = mergeParams(argv)
#     params = masifpniOpts["masifpni_site"]
#
#     used_na_type = argv.naType
#     if argv.training_list:
#         params["training_list"] = argv.training_list
#     if argv.testing_list:
#         params["testing_list"] = argv.testing_list
#
#     if masifpniOpts["use_gpu"]:
#         idx_xpu = masifpniOpts["gpu_dev"] if masifpniOpts["gpu_dev"] else "/gpu:0"
#     else:
#         idx_xpu = masifpniOpts["cpu_dev"] if masifpniOpts["cpu_dev"] else "/cpu:0"
#
#     if "n_theta" in params:
#         learning_obj = MasifPNI_site_nn(
#             params["max_distance"],
#             n_thetas=params["n_theta"],
#             n_rhos=params["n_rho"],
#             n_rotations=params["n_rotations"],
#             idx_xpu=idx_xpu,
#             feat_mask=params["n_feat"] * [1.0],
#             n_conv_layers=params["n_conv_layers"],
#         )
#     else:
#         learning_obj = MasifPNI_site_nn(
#             params["max_distance"],
#             n_thetas=4,
#             n_rhos=3,
#             n_rotations=4,
#             idx_xpu=idx_xpu,
#             feat_mask=params["n_feat"] * [1.0],
#             n_conv_layers=params["n_conv_layers"],
#         )
#
#     print(params["n_feat"] * [1.0])
#     if not os.path.exists(params["model_dir"]):
#         os.makedirs(params["model_dir"])
#     else:
#         if len(os.listdir(params["model_dir"])) != 0:
#             # Load existing network.
#             print('Reading pre-trained network')
#             learning_obj.saver.restore(learning_obj.session, os.path.join(params['model_dir'], 'model'))
#
#     batch_size = 100
#     num_iterations = 100
#
#     # Open training list.
#
#     list_training_loss = []
#     list_training_auc = []
#     list_validation_auc = []
#     iter_time = []
#     best_val_auc = 0
#
#     out_dir = params["model_dir"]
#     logfile = open(os.path.join(masifpniOpts["log_dir"], "nn_model.log.txt"), "w")
#     for key in params:
#         logfile.write("{}: {}\n".format(key, params[key]))
#
#     data_dirs = os.listdir(params["masif_precomputation_dir"])
#     training_pairs = getIdChainPairs(masifpniOpts, fromFile=params["training_list"])
#     training_pairs = [i for i in training_pairs if i.naType == used_na_type]
#     print(training_pairs)
#     training_df = pd.DataFrame(training_pairs)
#     filtered_training_list, data_dirs = filterPDBs(training_pairs, data_dirs, masifpniOpts,
#                                                    filterChainByLen=argv.filterChainByLen,
#                                                    batchRunFlag=argv.preprocessNobatchRun)
#
#     testing_pairs = getIdChainPairs(masifpniOpts, fromFile=params["testing_list"])
#     testing_pairs = [i for i in testing_pairs if i.naType == used_na_type]
#     testing_df = pd.DataFrame(testing_pairs)
#     filtered_testing_list, data_dirs = filterPDBs(testing_pairs, data_dirs, masifpniOpts,
#                                                   filterChainByLen=argv.filterChainByLen,
#                                                   batchRunFlag=argv.preprocessNobatchRun)
#     # np.random.shuffle(data_dirs)
#     # data_dirs = data_dirs
#     # n_val = len(data_dirs) // 10
#     # val_dirs = set(data_dirs[(len(data_dirs) - n_val):])
#     #
#     # for num_iter in range(num_iterations):
#     #     # Start training epoch:
#     #     list_training_loss = []
#     #     list_training_auc = []
#     #     list_val_auc = []
#     #     list_val_pos_labels = []
#     #     list_val_neg_labels = []
#     #     list_val_names = []
#     #     list_training_acc = []
#     #     list_val_acc = []
#     #     logfile.write("Starting epoch {}\n".format(num_iter))
#     #     print("Starting epoch {}".format(num_iter))
#     #     tic = time.time()
#     #     all_training_labels = []
#     #     all_training_scores = []
#     #     all_val_labels = []
#     #     all_val_scores = []
#     #     all_test_labels = []
#     #     all_test_scores = []
#     #     count_proteins = 0
#     #
#     #     list_test_auc = []
#     #     list_test_names = []
#     #     list_test_acc = []
#     #     all_test_labels = []
#     #     all_test_scores = []
#     #
#     #     for pdb_id in data_dirs:
#     #         mydir = os.path.join(params["masif_precomputation_dir"], pdb_id)
#     #
#     #         if pdb_id in filtered_training_list:
#     #             tmpList = glob.glob(os.path.join(mydir, "*_iface_labels.npy"))
#     #             if len(tmpList) == 0: continue
#     #             pChainNaChainPairs = [os.path.basename(i).split("_")[0:2] for i in tmpList]
#     #             for pChain, naChain in pChainNaChainPairs:
#     #                 iface_labels = np.load(os.path.join(mydir, "{}_{}_iface_labels.npy".format(pChain, naChain)))
#     #
#     #                 if len(iface_labels) > 8000:
#     #                     continue
#     #                 # if np.sum(iface_labels) > 0.75 * len(iface_labels) or np.sum(iface_labels) * 1.0 / len(iface_labels) < 0.05:
#     #                 if np.sum(iface_labels) > 0.75 * len(iface_labels) or np.sum(iface_labels) < 30:
#     #                     continue
#     #                 count_proteins += 1
#     #
#     #                 rho_wrt_center = np.load(os.path.join(mydir, pChain + "_rho_wrt_center.npy"))
#     #                 theta_wrt_center = np.load(os.path.join(mydir, pChain + "_theta_wrt_center.npy"))
#     #                 input_feat = np.load(os.path.join(mydir, pChain + "_input_feat.npy"))
#     #
#     #                 if params["n_feat"] < 5:
#     #                     input_feat = mask_input_feat(input_feat, params["n_feat"] * [1.0])
#     #                 mask = np.load(os.path.join(mydir, pChain + "_mask.npy"))
#     #                 mask = np.expand_dims(mask, 2)
#     #                 indices = np.load(os.path.join(mydir, pChain + "_list_indices.npy"), encoding="latin1", allow_pickle=True)
#     #                 indices = pad_indices(indices, mask.shape[1])
#     #                 tmp = np.zeros((len(iface_labels), 2))
#     #                 for i in range(len(iface_labels)):
#     #                     if iface_labels[i] == 1:
#     #                         tmp[i, 0] = 1
#     #                     else:
#     #                         tmp[i, 1] = 1
#     #                 iface_labels_dc = tmp
#     #                 logfile.flush()
#     #                 pos_labels = np.where(iface_labels == 1)[0]
#     #                 neg_labels = np.where(iface_labels == 0)[0]
#     #                 np.random.shuffle(neg_labels)
#     #                 np.random.shuffle(pos_labels)
#     #                 # Scramble neg idx, and only get as many as pos_labels to balance the training.
#     #                 if params["n_conv_layers"] == 1:
#     #                     n = min(len(pos_labels), len(neg_labels))
#     #                     n = min(n, batch_size // 2)
#     #                     subset = np.concatenate([neg_labels[:n], pos_labels[:n]])
#     #
#     #                     rho_wrt_center = rho_wrt_center[subset]
#     #                     theta_wrt_center = theta_wrt_center[subset]
#     #                     input_feat = input_feat[subset]
#     #                     mask = mask[subset]
#     #                     iface_labels_dc = iface_labels_dc[subset]
#     #                     indices = indices[subset]
#     #                     pos_labels = range(0, n)
#     #                     neg_labels = range(n, n * 2)
#     #                 else:
#     #                     n = min(len(pos_labels), len(neg_labels))
#     #                     neg_labels = neg_labels[:n]
#     #                     pos_labels = pos_labels[:n]
#     #
#     #                 feed_dict = {
#     #                     learning_obj.rho_coords: rho_wrt_center,
#     #                     learning_obj.theta_coords: theta_wrt_center,
#     #                     learning_obj.input_feat: input_feat,
#     #                     learning_obj.mask: mask,
#     #                     learning_obj.labels: iface_labels_dc,
#     #                     learning_obj.pos_idx: pos_labels,
#     #                     learning_obj.neg_idx: neg_labels,
#     #                     learning_obj.indices_tensor: indices,
#     #                 }
#     #
#     #                 if pdb_id in val_dirs:
#     #                     logfile.write("Validating on {} {}_{}\n".format(pdb_id, pChain, naChain))
#     #                     feed_dict[learning_obj.keep_prob] = 1.0
#     #                     training_loss, score, eval_labels = learning_obj.session.run(
#     #                         [
#     #                             learning_obj.data_loss,
#     #                             learning_obj.eval_score,
#     #                             learning_obj.eval_labels,
#     #                         ],
#     #                         feed_dict=feed_dict,
#     #                     )
#     #                     auc = metrics.roc_auc_score(eval_labels[:, 0], score)
#     #                     list_val_pos_labels.append(np.sum(iface_labels))
#     #                     list_val_neg_labels.append(len(iface_labels) - np.sum(iface_labels))
#     #                     list_val_auc.append(auc)
#     #                     list_val_names.append(pdb_id)
#     #                     all_val_labels = np.concatenate([all_val_labels, eval_labels[:, 0]])
#     #                     all_val_scores = np.concatenate([all_val_scores, score])
#     #                 else:
#     #                     logfile.write("Training on {} {}_{}\n".format(pdb_id, pChain, naChain))
#     #                     feed_dict[learning_obj.keep_prob] = 1.0
#     #                     _, training_loss, norm_grad, score, eval_labels = learning_obj.session.run(
#     #                         [
#     #                             learning_obj.optimizer,
#     #                             learning_obj.data_loss,
#     #                             learning_obj.norm_grad,
#     #                             learning_obj.eval_score,
#     #                             learning_obj.eval_labels,
#     #                         ],
#     #                         feed_dict=feed_dict,
#     #                     )
#     #                     all_training_labels = np.concatenate([all_training_labels, eval_labels[:, 0]])
#     #                     all_training_scores = np.concatenate([all_training_scores, score])
#     #                     auc = metrics.roc_auc_score(eval_labels[:, 0], score)
#     #                     list_training_auc.append(auc)
#     #                     list_training_loss.append(np.mean(training_loss))
#     #                 logfile.flush()
#     #
#     #     # Run testing cycle.
#     #     for pdb_id in data_dirs:
#     #         mydir = os.path.join(params["masif_precomputation_dir"], pdb_id)
#     #         if pdb_id in filtered_testing_list:
#     #             tmpList = glob.glob(os.path.join(mydir, "*_iface_labels.npy"))
#     #             pChainNaChainPairs = [os.path.basename(i).split("_")[0:2] for i in tmpList]
#     #             for pChain, naChain in pChainNaChainPairs:
#     #                 logfile.write("Testing on {} {}_{}\n".format(pdb_id, pChain, naChain))
#     #                 iface_labels = np.load(os.path.join(mydir, "{}_{}_iface_labels.npy".format(pChain, naChain)))
#     #
#     #                 if len(iface_labels) > 8000:
#     #                     continue
#     #                 # if np.sum(iface_labels) > 0.75 * len(iface_labels) or np.sum(iface_labels) * 1.0 / len(iface_labels) < 0.05:
#     #                 if np.sum(iface_labels) > 0.75 * len(iface_labels) or np.sum(iface_labels) < 30:
#     #                     continue
#     #                 count_proteins += 1
#     #
#     #                 rho_wrt_center = np.load(os.path.join(mydir, pChain + "_rho_wrt_center.npy"))
#     #                 theta_wrt_center = np.load(os.path.join(mydir, pChain + "_theta_wrt_center.npy"))
#     #                 input_feat = np.load(os.path.join(mydir, pChain + "_input_feat.npy"))
#     #                 if params["n_feat"] < 5:
#     #                     input_feat = mask_input_feat(input_feat, params["n_feat"] * [1.0])
#     #                 mask = np.load(os.path.join(mydir, pChain + "_mask.npy"))
#     #                 mask = np.expand_dims(mask, 2)
#     #                 indices = np.load(os.path.join(mydir, pChain + "_list_indices.npy"), encoding="latin1",
#     #                                   allow_pickle=True)
#     #                 # indices is (n_verts x <30), it should be
#     #                 indices = pad_indices(indices, mask.shape[1])
#     #                 tmp = np.zeros((len(iface_labels), 2))
#     #                 for i in range(len(iface_labels)):
#     #                     if iface_labels[i] == 1:
#     #                         tmp[i, 0] = 1
#     #                     else:
#     #                         tmp[i, 1] = 1
#     #                 iface_labels_dc = tmp
#     #                 logfile.flush()
#     #                 pos_labels = np.where(iface_labels == 1)[0]
#     #                 neg_labels = np.where(iface_labels == 0)[0]
#     #
#     #                 feed_dict = {
#     #                     learning_obj.rho_coords: rho_wrt_center,
#     #                     learning_obj.theta_coords: theta_wrt_center,
#     #                     learning_obj.input_feat: input_feat,
#     #                     learning_obj.mask: mask,
#     #                     learning_obj.labels: iface_labels_dc,
#     #                     learning_obj.pos_idx: pos_labels,
#     #                     learning_obj.neg_idx: neg_labels,
#     #                     learning_obj.indices_tensor: indices,
#     #                 }
#     #
#     #                 feed_dict[learning_obj.keep_prob] = 1.0
#     #                 score = learning_obj.session.run([learning_obj.full_score], feed_dict=feed_dict)
#     #                 score = score[0]
#     #                 auc = metrics.roc_auc_score(iface_labels, score)
#     #                 list_test_auc.append(auc)
#     #                 list_test_names.append((pdb_id, pChain, naChain))
#     #                 all_test_labels.append(iface_labels)
#     #                 all_test_scores.append(score)
#     #
#     #     outstr = "Epoch ran on {} proteins\n".format(count_proteins)
#     #     outstr += "Per protein AUC mean (training): {:.4f}; median: {:.4f} for iter {}\n".format(
#     #         np.mean(list_training_auc), np.median(list_training_auc), num_iter
#     #     )
#     #     outstr += "Per protein AUC mean (validation): {:.4f}; median: {:.4f} for iter {}\n".format(
#     #         np.mean(list_val_auc), np.median(list_val_auc), num_iter
#     #     )
#     #     outstr += "Per protein AUC mean (test): {:.4f}; median: {:.4f} for iter {}\n".format(
#     #         np.mean(list_test_auc), np.median(list_test_auc), num_iter
#     #     )
#     #     flat_all_test_labels = np.concatenate(all_test_labels, axis=0)
#     #     flat_all_test_scores = np.concatenate(all_test_scores, axis=0)
#     #     outstr += "Testing auc (all points): {:.2f}\n".format(
#     #         metrics.roc_auc_score(flat_all_test_labels, flat_all_test_scores)
#     #     )
#     #     outstr += "Epoch took {:2f}s\n".format(time.time() - tic)
#     #     logfile.write(outstr + "\n")
#     #     print(outstr)
#     #
#     #     if np.mean(list_val_auc) > best_val_auc:
#     #         logfile.write(">>> Saving model.\n")
#     #         print(">>> Saving model.\n")
#     #         best_val_auc = np.mean(list_val_auc)
#     #         output_model = os.path.join(out_dir, "model")
#     #         learning_obj.saver.save(learning_obj.session, output_model)
#     #         # Save the scores for test.
#     #         np.save(out_dir + "test_labels.npy", all_test_labels)
#     #         np.save(out_dir + "test_scores.npy", all_test_scores)
#     #         np.save(out_dir + "test_names.npy", list_test_names)
#     #
#     # logfile.close()


# def train_masifPNI_site_old(argv):
#     masifpniOpts = mergeParams(argv)
#     params = masifpniOpts["masifpni_site"]
#     if argv.training_list:
#         params["training_list"] = argv.training_list
#     if argv.testing_list:
#         params["testing_list"] = argv.testing_list
#
#     if masifpniOpts["use_gpu"]:
#         idx_xpu = masifpniOpts["gpu_dev"] if masifpniOpts["gpu_dev"] else "/gpu:0"
#     else:
#         idx_xpu = masifpniOpts["cpu_dev"] if masifpniOpts["cpu_dev"] else "/cpu:0"
#
#     if "n_theta" in params:
#         learning_obj = MasifPNI_site_nn(
#             params["max_distance"],
#             n_thetas=params["n_theta"],
#             n_rhos=params["n_rho"],
#             n_rotations=params["n_rotations"],
#             idx_xpu=idx_xpu,
#             feat_mask=params["n_feat"] * [1.0],
#             n_conv_layers=params["n_conv_layers"],
#         )
#     else:
#         learning_obj = MasifPNI_site_nn(
#             params["max_distance"],
#             n_thetas=4,
#             n_rhos=3,
#             n_rotations=4,
#             idx_xpu=idx_xpu,
#             feat_mask=params["n_feat"] * [1.0],
#             n_conv_layers=params["n_conv_layers"],
#         )
#
#     print(params["n_feat"] * [1.0])
#     if not os.path.exists(params["model_dir"]):
#         os.makedirs(params["model_dir"])
#     else:
#         if len(os.listdir(params["model_dir"])) != 0:
#             # Load existing network.
#             print('Reading pre-trained network')
#             learning_obj.saver.restore(learning_obj.session, params['model_dir'] + 'model')
#
#     batch_size = 100
#     num_iterations = 100
#
#     # Open training list.
#
#     list_training_loss = []
#     list_training_auc = []
#     list_validation_auc = []
#     iter_time = []
#     best_val_auc = 0
#
#     out_dir = params["model_dir"]
#     logfile = open(os.path.join(masifpniOpts["log_dir"], "nn_model.log.txt"), "w")
#     for key in params:
#         logfile.write("{}: {}\n".format(key, params[key]))
#
#     training_list = getIdChainPairs(masifpniOpts, fromFile=params["training_list"])
#     training_df = pd.DataFrame(training_list)
#     training_df_gp = training_df.groupby(["PDB_id", "pChain"])
#
#     testing_list = getIdChainPairs(masifpniOpts, fromFile=params["testing_list"])
#     testing_df = pd.DataFrame(testing_list)
#     testing_df_gp = testing_df.groupby(["PDB_id", "pChain"])
#
#     data_dirs = os.listdir(params["masif_precomputation_dir"])
#     idToDownload = [r.PDB_id for r in training_list if r.PDB_id not in data_dirs]
#     idToDownload.extend([r.PDB_id for r in testing_list if r.PDB_id not in data_dirs])
#     idToDownload = list(set(idToDownload))
#     if len(idToDownload) > 0:
#         dataprepFromList1(idToDownload, masifpniOpts, batchRunFlag=argv.nobatchRun)
#         data_dirs = list(set(data_dirs + idToDownload))
#
#     filterByLen = False
#     if filterByLen:
#         pdbIds = list(training_df.PDB_id.unique())
#         commonKeys = ["PDB_id", "pChain", "naChain"]
#         chainLenDf = getPdbChainLength(pdbIds, masifpniOpts['raw_pdb_dir'])
#         chainLenDf = chainLenDf[(chainLenDf.pChainLen >= 50) & (chainLenDf.naChainLen >= 15)]
#         i1 = training_df.set_index(commonKeys).index
#         i2 = chainLenDf.set_index(commonKeys).index
#         training_df = training_df[i1.isin(i2)]
#         training_df_gp = training_df.groupby(["PDB_id", "pChain"])
#
#     np.random.shuffle(data_dirs)
#     data_dirs = data_dirs
#     n_val = len(data_dirs) // 10
#     val_dirs = set(data_dirs[(len(data_dirs) - n_val):])
#
#     pni_pairs = np.load(masifpniOpts["pni_pairs_file"])
#     pni_pairs_df = pd.DataFrame(pni_pairs, columns=["PDB_id", "pChain", "naChain", "naType"])
#
#     for num_iter in range(num_iterations):
#         # Start training epoch:
#         list_training_loss = []
#         list_training_auc = []
#         list_val_auc = []
#         list_val_pos_labels = []
#         list_val_neg_labels = []
#         list_val_names = []
#         list_training_acc = []
#         list_val_acc = []
#         logfile.write("Starting epoch {}\n".format(num_iter))
#         print("Starting epoch {}".format(num_iter))
#         tic = time.time()
#         all_training_labels = []
#         all_training_scores = []
#         all_val_labels = []
#         all_val_scores = []
#         all_test_labels = []
#         all_test_scores = []
#         count_proteins = 0
#
#         list_test_auc = []
#         list_test_names = []
#         list_test_acc = []
#         all_test_labels = []
#         all_test_scores = []
#
#         for pdb_id in data_dirs:
#             mydir = os.path.join(params["masif_precomputation_dir"], pdb_id)
#             pChains = set(pni_pairs_df.loc[pni_pairs_df["PDB_id"] == pdb_id]["pChain"].to_list())
#
#             for chain in pChains:
#                 if (pdb_id, chain) in training_df_gp.groups:
#                     naChains = set(training_df_gp.get_group((pdb_id, chain)).naChain.to_list())
#
#                     try:
#                         iface_labels = np.load(os.path.join(mydir, chain + "_iface_labels.npy"))
#                     except:
#                         continue
#
#                     if len(iface_labels) > 8000:
#                         continue
#                     if np.sum(iface_labels) > 0.75 * len(iface_labels) or np.sum(iface_labels) < 30:
#                         continue
#                     count_proteins += 1
#
#                     rho_wrt_center = np.load(os.path.join(mydir, chain + "_rho_wrt_center.npy"))
#                     theta_wrt_center = np.load(os.path.join(mydir, chain + "_theta_wrt_center.npy"))
#                     input_feat = np.load(os.path.join(mydir, chain + "_input_feat.npy"))
#
#                     if params["n_feat"] < 5:
#                         input_feat = mask_input_feat(input_feat, params["n_feat"] * [1.0])
#                     mask = np.load(os.path.join(mydir, chain + "_mask.npy"))
#                     mask = np.expand_dims(mask, 2)
#                     indices = np.load(os.path.join(mydir, chain + "_list_indices.npy"), encoding="latin1", allow_pickle=True)
#                     indices = pad_indices(indices, mask.shape[1])
#                     tmp = np.zeros((len(iface_labels), 2))
#                     for i in range(len(iface_labels)):
#                         if iface_labels[i] == 1:
#                             tmp[i, 0] = 1
#                         else:
#                             tmp[i, 1] = 1
#                     iface_labels_dc = tmp
#                     logfile.flush()
#                     pos_labels = np.where(iface_labels == 1)[0]
#                     neg_labels = np.where(iface_labels == 0)[0]
#                     np.random.shuffle(neg_labels)
#                     np.random.shuffle(pos_labels)
#                     # Scramble neg idx, and only get as many as pos_labels to balance the training.
#                     if params["n_conv_layers"] == 1:
#                         n = min(len(pos_labels), len(neg_labels))
#                         n = min(n, batch_size // 2)
#                         subset = np.concatenate([neg_labels[:n], pos_labels[:n]])
#
#                         rho_wrt_center = rho_wrt_center[subset]
#                         theta_wrt_center = theta_wrt_center[subset]
#                         input_feat = input_feat[subset]
#                         mask = mask[subset]
#                         iface_labels_dc = iface_labels_dc[subset]
#                         indices = indices[subset]
#                         pos_labels = range(0, n)
#                         neg_labels = range(n, n * 2)
#                     else:
#                         n = min(len(pos_labels), len(neg_labels))
#                         neg_labels = neg_labels[:n]
#                         pos_labels = pos_labels[:n]
#
#                     feed_dict = {
#                         learning_obj.rho_coords: rho_wrt_center,
#                         learning_obj.theta_coords: theta_wrt_center,
#                         learning_obj.input_feat: input_feat,
#                         learning_obj.mask: mask,
#                         learning_obj.labels: iface_labels_dc,
#                         learning_obj.pos_idx: pos_labels,
#                         learning_obj.neg_idx: neg_labels,
#                         learning_obj.indices_tensor: indices,
#                     }
#
#                     if pdb_id in val_dirs:
#                         logfile.write("Validating on {} {}\n".format(pdb_id, chain))
#                         feed_dict[learning_obj.keep_prob] = 1.0
#                         training_loss, score, eval_labels = learning_obj.session.run(
#                             [
#                                 learning_obj.data_loss,
#                                 learning_obj.eval_score,
#                                 learning_obj.eval_labels,
#                             ],
#                             feed_dict=feed_dict,
#                         )
#                         auc = metrics.roc_auc_score(eval_labels[:, 0], score)
#                         list_val_pos_labels.append(np.sum(iface_labels))
#                         list_val_neg_labels.append(len(iface_labels) - np.sum(iface_labels))
#                         list_val_auc.append(auc)
#                         list_val_names.append(pdb_id)
#                         all_val_labels = np.concatenate([all_val_labels, eval_labels[:, 0]])
#                         all_val_scores = np.concatenate([all_val_scores, score])
#                     else:
#                         logfile.write("Training on {} {}\n".format(pdb_id, chain))
#                         feed_dict[learning_obj.keep_prob] = 1.0
#                         _, training_loss, norm_grad, score, eval_labels = learning_obj.session.run(
#                             [
#                                 learning_obj.optimizer,
#                                 learning_obj.data_loss,
#                                 learning_obj.norm_grad,
#                                 learning_obj.eval_score,
#                                 learning_obj.eval_labels,
#                             ],
#                             feed_dict=feed_dict,
#                         )
#                         all_training_labels = np.concatenate([all_training_labels, eval_labels[:, 0]])
#                         all_training_scores = np.concatenate([all_training_scores, score])
#                         auc = metrics.roc_auc_score(eval_labels[:, 0], score)
#                         list_training_auc.append(auc)
#                         list_training_loss.append(np.mean(training_loss))
#                     logfile.flush()
#
#         # Run testing cycle.
#         for pdb_id in data_dirs:
#             mydir = os.path.join(params["masif_precomputation_dir"], pdb_id)
#             pChains = set(pni_pairs_df.loc[pni_pairs_df["PDB_id"] == pdb_id]["pChain"].to_list())
#
#             for chain in pChains:
#                 if (pdb_id, chain) in testing_df_gp.groups:
#                     naChains = set(testing_df_gp.get_group((pdb_id, chain)).naChain.to_list())
#
#                     logfile.write("Testing on {} {}\n".format(pdb_id, chain))
#                     try:
#                         iface_labels = np.load(os.path.join(mydir, chain + "_iface_labels.npy"))
#                     except:
#                         continue
#
#                     if len(iface_labels) > 20000:
#                         continue
#                     if np.sum(iface_labels) > 0.75 * len(iface_labels) or np.sum(iface_labels) < 30:
#                         continue
#                     count_proteins += 1
#
#                     rho_wrt_center = np.load(os.path.join(mydir, chain + "_rho_wrt_center.npy"))
#                     theta_wrt_center = np.load(os.path.join(mydir, chain + "_theta_wrt_center.npy"))
#                     input_feat = np.load(os.path.join(mydir, chain + "_input_feat.npy"))
#                     if params["n_feat"] < 5:
#                         input_feat = mask_input_feat(input_feat, params["n_feat"] * [1.0])
#                     mask = np.load(os.path.join(mydir, chain + "_mask.npy"))
#                     mask = np.expand_dims(mask, 2)
#                     indices = np.load(os.path.join(mydir, chain + "_list_indices.npy"), encoding="latin1", allow_pickle=True)
#                     # indices is (n_verts x <30), it should be
#                     indices = pad_indices(indices, mask.shape[1])
#                     tmp = np.zeros((len(iface_labels), 2))
#                     for i in range(len(iface_labels)):
#                         if iface_labels[i] == 1:
#                             tmp[i, 0] = 1
#                         else:
#                             tmp[i, 1] = 1
#                     iface_labels_dc = tmp
#                     logfile.flush()
#                     pos_labels = np.where(iface_labels == 1)[0]
#                     neg_labels = np.where(iface_labels == 0)[0]
#
#                     feed_dict = {
#                         learning_obj.rho_coords: rho_wrt_center,
#                         learning_obj.theta_coords: theta_wrt_center,
#                         learning_obj.input_feat: input_feat,
#                         learning_obj.mask: mask,
#                         learning_obj.labels: iface_labels_dc,
#                         learning_obj.pos_idx: pos_labels,
#                         learning_obj.neg_idx: neg_labels,
#                         learning_obj.indices_tensor: indices,
#                     }
#
#                     feed_dict[learning_obj.keep_prob] = 1.0
#                     score = learning_obj.session.run([learning_obj.full_score], feed_dict=feed_dict)
#                     score = score[0]
#                     auc = metrics.roc_auc_score(iface_labels, score)
#                     list_test_auc.append(auc)
#                     list_test_names.append((pdb_id, chain))
#                     all_test_labels.append(iface_labels)
#                     all_test_scores.append(score)
#
#         outstr = "Epoch ran on {} proteins\n".format(count_proteins)
#         outstr += "Per protein AUC mean (training): {:.4f}; median: {:.4f} for iter {}\n".format(
#             np.mean(list_training_auc), np.median(list_training_auc), num_iter
#         )
#         outstr += "Per protein AUC mean (validation): {:.4f}; median: {:.4f} for iter {}\n".format(
#             np.mean(list_val_auc), np.median(list_val_auc), num_iter
#         )
#         outstr += "Per protein AUC mean (test): {:.4f}; median: {:.4f} for iter {}\n".format(
#             np.mean(list_test_auc), np.median(list_test_auc), num_iter
#         )
#         flat_all_test_labels = np.concatenate(all_test_labels, axis=0)
#         flat_all_test_scores = np.concatenate(all_test_scores, axis=0)
#         outstr += "Testing auc (all points): {:.2f}\n".format(
#             metrics.roc_auc_score(flat_all_test_labels, flat_all_test_scores)
#         )
#         outstr += "Epoch took {:2f}s\n".format(time.time() - tic)
#         logfile.write(outstr + "\n")
#         print(outstr)
#
#         if np.mean(list_val_auc) > best_val_auc:
#             logfile.write(">>> Saving model.\n")
#             print(">>> Saving model.\n")
#             best_val_auc = np.mean(list_val_auc)
#             output_model = os.path.join(out_dir, "model")
#             learning_obj.saver.save(learning_obj.session, output_model)
#             # Save the scores for test.
#             np.save(out_dir + "test_labels.npy", all_test_labels)
#             np.save(out_dir + "test_scores.npy", all_test_scores)
#             np.save(out_dir + "test_names.npy", list_test_names)
#
#     logfile.close()


# def filterPDBs(pdb_list, data_dirs, masifpniOpts, batchRunFlag=False, filterChainByLen=False):
#     raw_pdbs = [i.split(".")[0] for i in os.listdir(masifpniOpts['raw_pdb_dir'])]
#     idToDownload = [r.PDB_id for r in pdb_list if r.PDB_id not in set(data_dirs + raw_pdbs)]
#     idToDownload = list(set(idToDownload))
#     if len(idToDownload) > 0:
#         dataprepFromList1(idToDownload, masifpniOpts, batchRunFlag=batchRunFlag)
#         precompute_dir = masifpniOpts["masifpni_site"]["masif_precomputation_dir"]
#         processedIds = [i for i in idToDownload if os.path.exists(os.path.join(precompute_dir, i))]
#         data_dirs = list(set(data_dirs + processedIds))
#
#     my_df = pd.DataFrame(pdb_list)
#     if filterChainByLen:
#         pdbIds = list(my_df.PDB_id.unique())
#         commonKeys = ["PDB_id", "pChain", "naChain"]
#         chainLenDf = getPdbChainLength(pdbIds, masifpniOpts['raw_pdb_dir'])
#         chainLenDf = chainLenDf[(chainLenDf.pChainLen >= 50)]
#         i1 = my_df.set_index(commonKeys).index
#         i2 = chainLenDf.set_index(commonKeys).index
#         my_df = my_df[i1.isin(i2)]
#
#     filtered_training_list = list(my_df.PDB_id.unique())
#     return filtered_training_list, data_dirs


# def train_masifPNI_site(argv):
#     masifpniOpts = mergeParams(argv)
#     params = masifpniOpts["masifpni_site"]
#     if argv.training_list:
#         params["training_list"] = argv.training_list
#     if argv.testing_list:
#         params["testing_list"] = argv.testing_list
#
#     if masifpniOpts["use_gpu"]:
#         idx_xpu = masifpniOpts["gpu_dev"] if masifpniOpts["gpu_dev"] else "/gpu:0"
#     else:
#         idx_xpu = masifpniOpts["cpu_dev"] if masifpniOpts["cpu_dev"] else "/cpu:0"
#
#     if "n_theta" in params:
#         learning_obj = MasifPNI_site_nn(
#             params["max_distance"],
#             n_thetas=params["n_theta"],
#             n_rhos=params["n_rho"],
#             n_rotations=params["n_rotations"],
#             idx_xpu=idx_xpu,
#             feat_mask=params["n_feat"] * [1.0],
#             n_conv_layers=params["n_conv_layers"],
#         )
#     else:
#         learning_obj = MasifPNI_site_nn(
#             params["max_distance"],
#             n_thetas=4,
#             n_rhos=3,
#             n_rotations=4,
#             idx_xpu=idx_xpu,
#             feat_mask=params["n_feat"] * [1.0],
#             n_conv_layers=params["n_conv_layers"],
#         )
#
#     print(params["n_feat"] * [1.0])
#     if not os.path.exists(params["model_dir"]):
#         os.makedirs(params["model_dir"])
#     else:
#         if len(os.listdir(params["model_dir"])) != 0:
#             # Load existing network.
#             print('Reading pre-trained network')
#             learning_obj.saver.restore(learning_obj.session, os.path.join(params['model_dir'], 'model'))
#
#     batch_size = 100
#     num_iterations = 100
#
#     # Open training list.
#
#     list_training_loss = []
#     list_training_auc = []
#     list_validation_auc = []
#     iter_time = []
#     best_val_auc = 0
#
#     out_dir = params["model_dir"]
#     logfile = open(os.path.join(masifpniOpts["log_dir"], "nn_model.log.txt"), "w")
#     for key in params:
#         logfile.write("{}: {}\n".format(key, params[key]))
#
#     data_dirs = os.listdir(params["masif_precomputation_dir"])
#     training_pairs = getIdChainPairs(masifpniOpts, fromFile=params["training_list"])
#     training_df = pd.DataFrame(training_pairs)
#     filtered_training_list, data_dirs = filterPDBs(training_pairs, data_dirs, masifpniOpts,
#                                                    filterChainByLen=argv.filterChainByLen,
#                                                    batchRunFlag=argv.preprocessNobatchRun)
#
#     testing_pairs = getIdChainPairs(masifpniOpts, fromFile=params["testing_list"])
#     testing_df = pd.DataFrame(testing_pairs)
#     filtered_testing_list, data_dirs = filterPDBs(testing_pairs, data_dirs, masifpniOpts,
#                                                   filterChainByLen=argv.filterChainByLen,
#                                                   batchRunFlag=argv.preprocessNobatchRun)
#     np.random.shuffle(data_dirs)
#     data_dirs = data_dirs
#     n_val = len(data_dirs) // 10
#     val_dirs = set(data_dirs[(len(data_dirs) - n_val):])
#
#     for num_iter in range(num_iterations):
#         # Start training epoch:
#         list_training_loss = []
#         list_training_auc = []
#         list_val_auc = []
#         list_val_pos_labels = []
#         list_val_neg_labels = []
#         list_val_names = []
#         list_training_acc = []
#         list_val_acc = []
#         logfile.write("Starting epoch {}\n".format(num_iter))
#         print("Starting epoch {}".format(num_iter))
#         tic = time.time()
#         all_training_labels = []
#         all_training_scores = []
#         all_val_labels = []
#         all_val_scores = []
#         all_test_labels = []
#         all_test_scores = []
#         count_proteins = 0
#
#         list_test_auc = []
#         list_test_names = []
#         list_test_acc = []
#         all_test_labels = []
#         all_test_scores = []
#
#         for pdb_id in data_dirs:
#             mydir = os.path.join(params["masif_precomputation_dir"], pdb_id)
#
#             if pdb_id in filtered_training_list:
#                 tmpList = glob.glob(os.path.join(mydir, "*_iface_labels.npy"))
#                 if len(tmpList) == 0: continue
#                 pChains = [os.path.basename(i).split("_")[0] for i in tmpList]
#                 for chain in pChains:
#                     iface_labels = np.load(os.path.join(mydir, chain + "_iface_labels.npy"))
#
#                     if len(iface_labels) > 8000:
#                         continue
#                     # if np.sum(iface_labels) > 0.75 * len(iface_labels) or np.sum(iface_labels) * 1.0 / len(iface_labels) < 0.05:
#                     if np.sum(iface_labels) > 0.75 * len(iface_labels) or np.sum(iface_labels) < 30:
#                         continue
#                     count_proteins += 1
#
#                     rho_wrt_center = np.load(os.path.join(mydir, chain + "_rho_wrt_center.npy"))
#                     theta_wrt_center = np.load(os.path.join(mydir, chain + "_theta_wrt_center.npy"))
#                     input_feat = np.load(os.path.join(mydir, chain + "_input_feat.npy"))
#
#                     if params["n_feat"] < 5:
#                         input_feat = mask_input_feat(input_feat, params["n_feat"] * [1.0])
#                     mask = np.load(os.path.join(mydir, chain + "_mask.npy"))
#                     mask = np.expand_dims(mask, 2)
#                     indices = np.load(os.path.join(mydir, chain + "_list_indices.npy"), encoding="latin1", allow_pickle=True)
#                     indices = pad_indices(indices, mask.shape[1])
#                     tmp = np.zeros((len(iface_labels), 2))
#                     for i in range(len(iface_labels)):
#                         if iface_labels[i] == 1:
#                             tmp[i, 0] = 1
#                         else:
#                             tmp[i, 1] = 1
#                     iface_labels_dc = tmp
#                     logfile.flush()
#                     pos_labels = np.where(iface_labels == 1)[0]
#                     neg_labels = np.where(iface_labels == 0)[0]
#                     np.random.shuffle(neg_labels)
#                     np.random.shuffle(pos_labels)
#                     # Scramble neg idx, and only get as many as pos_labels to balance the training.
#                     if params["n_conv_layers"] == 1:
#                         n = min(len(pos_labels), len(neg_labels))
#                         n = min(n, batch_size // 2)
#                         subset = np.concatenate([neg_labels[:n], pos_labels[:n]])
#
#                         rho_wrt_center = rho_wrt_center[subset]
#                         theta_wrt_center = theta_wrt_center[subset]
#                         input_feat = input_feat[subset]
#                         mask = mask[subset]
#                         iface_labels_dc = iface_labels_dc[subset]
#                         indices = indices[subset]
#                         pos_labels = range(0, n)
#                         neg_labels = range(n, n * 2)
#                     else:
#                         n = min(len(pos_labels), len(neg_labels))
#                         neg_labels = neg_labels[:n]
#                         pos_labels = pos_labels[:n]
#
#                     feed_dict = {
#                         learning_obj.rho_coords: rho_wrt_center,
#                         learning_obj.theta_coords: theta_wrt_center,
#                         learning_obj.input_feat: input_feat,
#                         learning_obj.mask: mask,
#                         learning_obj.labels: iface_labels_dc,
#                         learning_obj.pos_idx: pos_labels,
#                         learning_obj.neg_idx: neg_labels,
#                         learning_obj.indices_tensor: indices,
#                     }
#
#                     if pdb_id in val_dirs:
#                         logfile.write("Validating on {} {}\n".format(pdb_id, chain))
#                         feed_dict[learning_obj.keep_prob] = 1.0
#                         training_loss, score, eval_labels = learning_obj.session.run(
#                             [
#                                 learning_obj.data_loss,
#                                 learning_obj.eval_score,
#                                 learning_obj.eval_labels,
#                             ],
#                             feed_dict=feed_dict,
#                         )
#                         auc = metrics.roc_auc_score(eval_labels[:, 0], score)
#                         list_val_pos_labels.append(np.sum(iface_labels))
#                         list_val_neg_labels.append(len(iface_labels) - np.sum(iface_labels))
#                         list_val_auc.append(auc)
#                         list_val_names.append(pdb_id)
#                         all_val_labels = np.concatenate([all_val_labels, eval_labels[:, 0]])
#                         all_val_scores = np.concatenate([all_val_scores, score])
#                     else:
#                         logfile.write("Training on {} {}\n".format(pdb_id, chain))
#                         feed_dict[learning_obj.keep_prob] = 1.0
#                         _, training_loss, norm_grad, score, eval_labels = learning_obj.session.run(
#                             [
#                                 learning_obj.optimizer,
#                                 learning_obj.data_loss,
#                                 learning_obj.norm_grad,
#                                 learning_obj.eval_score,
#                                 learning_obj.eval_labels,
#                             ],
#                             feed_dict=feed_dict,
#                         )
#                         all_training_labels = np.concatenate([all_training_labels, eval_labels[:, 0]])
#                         all_training_scores = np.concatenate([all_training_scores, score])
#                         auc = metrics.roc_auc_score(eval_labels[:, 0], score)
#                         list_training_auc.append(auc)
#                         list_training_loss.append(np.mean(training_loss))
#                     logfile.flush()
#
#         # Run testing cycle.
#         for pdb_id in data_dirs:
#             mydir = os.path.join(params["masif_precomputation_dir"], pdb_id)
#             if pdb_id in filtered_testing_list:
#                 tmpList = glob.glob(os.path.join(mydir, "*_iface_labels.npy"))
#                 pChains = [os.path.basename(i).split("_")[0] for i in tmpList]
#                 for chain in pChains:
#                     logfile.write("Testing on {} {}\n".format(pdb_id, chain))
#                     iface_labels = np.load(os.path.join(mydir, chain + "_iface_labels.npy"))
#
#                     if len(iface_labels) > 8000:
#                         continue
#                     # if np.sum(iface_labels) > 0.75 * len(iface_labels) or np.sum(iface_labels) * 1.0 / len(iface_labels) < 0.05:
#                     if np.sum(iface_labels) > 0.75 * len(iface_labels) or np.sum(iface_labels) < 30:
#                         continue
#                     count_proteins += 1
#
#                     rho_wrt_center = np.load(os.path.join(mydir, chain + "_rho_wrt_center.npy"))
#                     theta_wrt_center = np.load(os.path.join(mydir, chain + "_theta_wrt_center.npy"))
#                     input_feat = np.load(os.path.join(mydir, chain + "_input_feat.npy"))
#                     if params["n_feat"] < 5:
#                         input_feat = mask_input_feat(input_feat, params["n_feat"] * [1.0])
#                     mask = np.load(os.path.join(mydir, chain + "_mask.npy"))
#                     mask = np.expand_dims(mask, 2)
#                     indices = np.load(os.path.join(mydir, chain + "_list_indices.npy"), encoding="latin1",
#                                       allow_pickle=True)
#                     # indices is (n_verts x <30), it should be
#                     indices = pad_indices(indices, mask.shape[1])
#                     tmp = np.zeros((len(iface_labels), 2))
#                     for i in range(len(iface_labels)):
#                         if iface_labels[i] == 1:
#                             tmp[i, 0] = 1
#                         else:
#                             tmp[i, 1] = 1
#                     iface_labels_dc = tmp
#                     logfile.flush()
#                     pos_labels = np.where(iface_labels == 1)[0]
#                     neg_labels = np.where(iface_labels == 0)[0]
#
#                     feed_dict = {
#                         learning_obj.rho_coords: rho_wrt_center,
#                         learning_obj.theta_coords: theta_wrt_center,
#                         learning_obj.input_feat: input_feat,
#                         learning_obj.mask: mask,
#                         learning_obj.labels: iface_labels_dc,
#                         learning_obj.pos_idx: pos_labels,
#                         learning_obj.neg_idx: neg_labels,
#                         learning_obj.indices_tensor: indices,
#                     }
#
#                     feed_dict[learning_obj.keep_prob] = 1.0
#                     score = learning_obj.session.run([learning_obj.full_score], feed_dict=feed_dict)
#                     score = score[0]
#                     auc = metrics.roc_auc_score(iface_labels, score)
#                     list_test_auc.append(auc)
#                     list_test_names.append((pdb_id, chain))
#                     all_test_labels.append(iface_labels)
#                     all_test_scores.append(score)
#
#         outstr = "Epoch ran on {} proteins\n".format(count_proteins)
#         outstr += "Per protein AUC mean (training): {:.4f}; median: {:.4f} for iter {}\n".format(
#             np.mean(list_training_auc), np.median(list_training_auc), num_iter
#         )
#         outstr += "Per protein AUC mean (validation): {:.4f}; median: {:.4f} for iter {}\n".format(
#             np.mean(list_val_auc), np.median(list_val_auc), num_iter
#         )
#         outstr += "Per protein AUC mean (test): {:.4f}; median: {:.4f} for iter {}\n".format(
#             np.mean(list_test_auc), np.median(list_test_auc), num_iter
#         )
#         flat_all_test_labels = np.concatenate(all_test_labels, axis=0)
#         flat_all_test_scores = np.concatenate(all_test_scores, axis=0)
#         outstr += "Testing auc (all points): {:.2f}\n".format(
#             metrics.roc_auc_score(flat_all_test_labels, flat_all_test_scores)
#         )
#         outstr += "Epoch took {:2f}s\n".format(time.time() - tic)
#         logfile.write(outstr + "\n")
#         print(outstr)
#
#         if np.mean(list_val_auc) > best_val_auc:
#             logfile.write(">>> Saving model.\n")
#             print(">>> Saving model.\n")
#             best_val_auc = np.mean(list_val_auc)
#             output_model = os.path.join(out_dir, "model")
#             learning_obj.saver.save(learning_obj.session, output_model)
#             # Save the scores for test.
#             np.save(out_dir + "test_labels.npy", all_test_labels)
#             np.save(out_dir + "test_scores.npy", all_test_scores)
#             np.save(out_dir + "test_names.npy", list_test_names)
#
#     logfile.close()


if __name__ == '__main__':
    train_masifPNI_site2(sys.argv)

