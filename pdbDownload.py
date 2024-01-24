#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
File name: pdbDownload.py
Author: CrazyHsu @ crazyhsu9627@gmail.com
Created on: 2022-09-11 20:47:44
Last modified: 2022-09-11 20:47:44
'''

from Bio.PDB import *

from commonFuncs import *
from inputOutputProcess import protonate


def targetPdbDownload(masifpniOpts, pdb_id, pdbl, overwrite=True):
    protonated_file = os.path.join(masifpniOpts['raw_pdb_dir'], pdb_id + ".pdb")
    pdb_filename = ""
    if overwrite:
        pdb_filename = pdbl.retrieve_pdb_file(pdb_id, pdir=masifpniOpts['tmp_dir'], file_format='pdb', overwrite=overwrite)
    else:
        if not os.path.exists(protonated_file):
            pdb_filename = pdbl.retrieve_pdb_file(pdb_id, pdir=masifpniOpts['tmp_dir'], file_format='pdb', overwrite=overwrite)
        else:
            return

    ##### Protonate with reduce, if hydrogens included.
    # - Always protonate as this is useful for charges. If necessary ignore hydrogens later.
    unDownload = []
    if os.path.exists(pdb_filename):
        protonate(pdb_filename, protonated_file)
    else:
        unDownload.append(pdb_id)
    return unDownload


def pdbDownload(argv):
    masifpniOpts = mergeParams1(argv)

    pdbIds = []
    if masifpniOpts["pdb_id_list"]:
        pdbIds = [j.split("_")[0].upper() for j in [i for i in masifpniOpts["pdb_id_list"].split(",")]]
    if masifpniOpts["pdb_id_file"]:
        with open(masifpniOpts["pdb_id_file"]) as f:
            for i in f.readlines():
                if i.startswith("#"): continue
                pdbIds.append(i.strip().split("_")[0].upper())
    if not pdbIds and not masifpniOpts["all_pdbs"]:
        pdbIds = ["4UN3"]
    pdbIds = list(set(pdbIds))

    pdbl = PDBList(server='http://files.wwpdb.org')
    targetPdbDownloadBatchRun = []
    for pdb_id in pdbIds:
        pdbFile = os.path.join(masifpniOpts['raw_pdb_dir'], pdb_id + ".pdb")
        if os.path.exists(pdbFile): continue
        targetPdbDownloadBatchRun.append((masifpniOpts, pdb_id, pdbl, masifpniOpts["overwrite_pdb"]))
    downloadDesc = "Download PDBs"
    resolveDirs([masifpniOpts['raw_pdb_dir'], masifpniOpts['tmp_dir']])
    batchRunFlag = False if masifpniOpts["no_batch_run"] else True
    resultList = batchRun1(targetPdbDownload, targetPdbDownloadBatchRun, n_threads=masifpniOpts["n_threads"],
                           desc=downloadDesc, batchRunFlag=batchRunFlag)

    unDownload = list(itertools.chain.from_iterable(resultList))
    with open(os.path.join(masifpniOpts["log_dir"], "unable_download.txt"), "w") as f:
        for i in unDownload:
            print(i, file=f)

    removeDirs([masifpniOpts["tmp_dir"]], empty=True)
