#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
File name: dataPreparation.py
Author: CrazyHsu @ crazyhsu9627@gmail.com
Created on: 2022-09-13 00:16:34
Last modified: 2022-09-13 00:16:34
'''

import sys, itertools, tempfile
import numpy as np
import pandas as pd
from Bio.PDB import PDBList

from commonFuncs import *
from parseConfig1 import DefaultConfig, GlobalVars
from pdbDownload import targetPdbDownload
from precompute import precomputeProteinPlyInfo1, getNaCorrd
from inputOutputProcess import *
from readDataFromSurface import *

def generate_NA_coord(masifpniOpts, pdbChainFullInfo, batchRunFlag=False, mergeNaChain=True):
    generateNaInfoBatchRun = []

    pdbId2naChain = {}
    for _, row in pdbChainFullInfo.iterrows():
        pdb_id = row.PDB_id
        pChain = "".join(sorted(row.pChain))
        naChain = "".join(sorted(row.naChain))
        if pdb_id not in pdbId2naChain:
            pdbId2naChain[pdb_id] = {"pChain": pChain, "naChain": [naChain]}
        else:
            pdbId2naChain[pdb_id]["naChain"].append(naChain)

        rawPdbFile = os.path.join(masifpniOpts["raw_pdb_dir"], pdb_id + ".pdb")
        generateNaInfoBatchRun.append((rawPdbFile, pdb_id, pChain, naChain))

    if mergeNaChain:
        generateNaInfoBatchRun = []
        for pdb_id in pdbId2naChain:
            rawPdbFile = os.path.join(masifpniOpts["raw_pdb_dir"], pdb_id + ".pdb")
            pChain = pdbId2naChain[pdb_id]["pChain"]
            naChain = "".join(sorted("".join(pdbId2naChain[pdb_id]["naChain"])))
            generateNaInfoBatchRun.append((rawPdbFile, pdb_id, pChain, naChain))

    generateNaInfoBatchRun = [(masifpniOpts,) + i for i in set(generateNaInfoBatchRun)]

    getNaInfoDesc = "Getting NA information"
    batchRun1(getNaCorrd, generateNaInfoBatchRun, n_threads=masifpniOpts["n_threads"], desc=getNaInfoDesc, batchRunFlag=batchRunFlag)


def extractSurfaceInfo(masifpniOpts, pdbChainFullInfo, batchRunFlag=False, mergeNaChain=True):
    pPdbDir = os.path.join(masifpniOpts["extract_pdb"], "protein")
    rnaPdbDir = os.path.join(masifpniOpts["extract_pdb"], "RNA")
    dnaPdbDir = os.path.join(masifpniOpts["extract_pdb"], "DNA")
    pnaPdbDir = os.path.join(masifpniOpts["extract_pdb"], "protNaComplex")
    naPdbDir = os.path.join(masifpniOpts["extract_pdb"], "allNaPerComplex")
    resolveDirs([pPdbDir, rnaPdbDir, dnaPdbDir, pnaPdbDir, naPdbDir, masifpniOpts["ply_chain_dir"], masifpniOpts["vertex_chain_dir"]])
    extractPchainPDBbatchRun = []
    extractNAchainPDBbatchRun = []
    extractAllNAchainPDBbatchRun = []
    extractPnaPDBbatchRun = []
    extractProteinTriangulateBatchRun = []
    extractNaTriangulateBatchRun = []
    precomputeProteinPlyInfoBatchRun = []

    pdbId2naChain = {}
    for _, row in pdbChainFullInfo.iterrows():
        pdb_id = row.PDB_id
        pChain = row.pChain
        naChain = row.naChain
        naType = row.naType
        if pdb_id not in pdbId2naChain:
            pdbId2naChain[pdb_id] = {"pChain": pChain, "naChain": [naChain]}
        else:
            pdbId2naChain[pdb_id]["naChain"].append(naChain)

        rawPdbFile = os.path.join(masifpniOpts["raw_pdb_dir"], pdb_id + ".pdb")
        pPdbFile = os.path.join(pPdbDir, "{}_{}.pdb".format(pdb_id, pChain))
        naPdbFile = os.path.join(masifpniOpts["extract_pdb"], naType, "{}_{}.pdb".format(pdb_id, naChain))
        pnaPdbFile = os.path.join(pnaPdbDir, "{}_{}_{}.pdb".format(pdb_id, pChain, naChain))
        pPlyFile = masifpniOpts['ply_file_template'].format(pdb_id, pChain, naChain)
        naPlyFile = masifpniOpts['ply_file_template'].format(pdb_id, naChain, pChain)
        pVerticeFile = masifpniOpts['vertex_file_template'].format(pdb_id, pChain, naChain)
        naVerticeFile = masifpniOpts['vertex_file_template'].format(pdb_id, naChain, pChain)

        extractPchainPDBbatchRun.append((rawPdbFile, pPdbFile, pChain))
        extractNAchainPDBbatchRun.append((rawPdbFile, naPdbFile, naChain))
        extractPnaPDBbatchRun.append((rawPdbFile, pnaPdbFile, pChain + naChain))
        extractProteinTriangulateBatchRun.append((pPdbFile, naPdbFile, pPlyFile, pVerticeFile))
        extractNaTriangulateBatchRun.append((naPdbFile, pPdbFile, naPlyFile, naVerticeFile))
        precomputeProteinPlyInfoBatchRun.append((pPlyFile, pdb_id, pChain, naChain))
        # precomputeNaPlyInfoBatchRun.append((pid, naChain))

    extractPchainPDBbatchRun = set(extractPchainPDBbatchRun)
    extractNAchainPDBbatchRun = set(extractNAchainPDBbatchRun)
    extractPnaPDBbatchRun = set(extractPnaPDBbatchRun)

    if mergeNaChain:
        extractProteinTriangulateBatchRun = []
        precomputeProteinPlyInfoBatchRun = []
        for pdb_id in pdbId2naChain:
            rawPdbFile = os.path.join(masifpniOpts["raw_pdb_dir"], pdb_id + ".pdb")
            pChain = pdbId2naChain[pdb_id]["pChain"]
            pChain = "".join(sorted(pChain))
            naChain = "".join(pdbId2naChain[pdb_id]["naChain"])
            naChain = "".join(sorted(naChain))
            pPdbFile = os.path.join(pPdbDir, "{}_{}.pdb".format(pdb_id, pChain))
            naPdbFile = os.path.join(naPdbDir, "{}_{}.pdb".format(pdb_id, naChain))
            pPlyFile = masifpniOpts['ply_file_template'].format(pdb_id, pChain, naChain)
            pVerticeFile = masifpniOpts['vertex_file_template'].format(pdb_id, pChain, naChain)
            extractAllNAchainPDBbatchRun.append((rawPdbFile, naPdbFile, naChain))
            extractProteinTriangulateBatchRun.append((pPdbFile, naPdbFile, pPlyFile, pVerticeFile))
            precomputeProteinPlyInfoBatchRun.append((pPlyFile, pdb_id, pChain, naChain))

    extractProteinTriangulateBatchRun = [(masifpniOpts,) + i for i in set(extractProteinTriangulateBatchRun)]
    extractNaTriangulateBatchRun = [(masifpniOpts,) + i for i in set(extractNaTriangulateBatchRun)]
    precomputeProteinPlyInfoBatchRun = [(masifpniOpts,) + i for i in set(precomputeProteinPlyInfoBatchRun)]
    # precomputeNaPlyInfoBatchRun = [(masifpniOpts,) + i for i in set(precomputeNaPlyInfoBatchRun)]

    extractPchainPDBDesc = "Extract protein chains"
    batchRun1(extractPDB, extractPchainPDBbatchRun, n_threads=masifpniOpts["n_threads"], desc=extractPchainPDBDesc,
              batchRunFlag=batchRunFlag)
    extractNAchainPDBDesc = "Extract nucleic acid chains"
    batchRun1(extractPDB, extractNAchainPDBbatchRun, n_threads=masifpniOpts["n_threads"], desc=extractNAchainPDBDesc,
              batchRunFlag=batchRunFlag)
    extractPnaChainPDBDesc = "Extract protein and nucleic acid complex chains"
    batchRun1(extractPDB, extractPnaPDBbatchRun, n_threads=masifpniOpts["n_threads"], desc=extractPnaChainPDBDesc,
              batchRunFlag=batchRunFlag)
    extractAllNAChainPDBDesc = "Extract all nucleic acid chains per complex"
    batchRun1(extractPDB, extractAllNAchainPDBbatchRun, n_threads=masifpniOpts["n_threads"],
              desc=extractAllNAChainPDBDesc, batchRunFlag=batchRunFlag)

    extractProteinTriangulateDesc = "Extract protein triangulate"
    batchRun1(extractProteinTriangulate1, extractProteinTriangulateBatchRun, n_threads=masifpniOpts["n_threads"],
              desc=extractProteinTriangulateDesc, batchRunFlag=batchRunFlag)

    plySize = 5000000
    precomputeProteinPlyInfoBatchRun = [i for i in precomputeProteinPlyInfoBatchRun if os.path.exists(i[1])]
    precomputeProteinPlyInfoBatchRun1 = [i for i in precomputeProteinPlyInfoBatchRun if os.path.getsize(i[1]) <= plySize]
    precomputeProteinPlyInfoBatchRun2 = [i for i in precomputeProteinPlyInfoBatchRun if os.path.getsize(i[1]) > plySize]

    # print([i[1:] for i in precomputeProteinPlyInfoBatchRun])
    if len(precomputeProteinPlyInfoBatchRun1) > 0:
        precomputeProteinPlyInfoDesc1 = "Precompute protein ply information for size less than {}".format(plySize)
        batchRun1(precomputeProteinPlyInfo1, precomputeProteinPlyInfoBatchRun1, n_threads=masifpniOpts["n_threads"]+10,
                  desc=precomputeProteinPlyInfoDesc1, batchRunFlag=batchRunFlag)
    if len(precomputeProteinPlyInfoBatchRun2) > 0:
        precomputeProteinPlyInfoDesc2 = "Precompute protein ply information for size large than {}".format(plySize)
        batchRun1(precomputeProteinPlyInfo1, precomputeProteinPlyInfoBatchRun2, n_threads=masifpniOpts["n_threads"]//2,
                  desc=precomputeProteinPlyInfoDesc2, batchRunFlag=batchRunFlag)


def dataprepFromList3(pdbIdChains, masifpniOpts, runAll=False, resumeDownload=False, resumeFindBound=False,
                      resumeExtractPDB=False, resumeExtractTriangulate=False, resumePrecomputePly=False,
                      batchRunFlag=True, mergeNaChain=True, naType=None):
    GlobalVars().setEnviron()
    resolveDirs([masifpniOpts['raw_pdb_dir'], masifpniOpts['tmp_dir']])

    pdbIds = [i.split("_")[0].upper() for i in set(pdbIdChains)]
    pdbl = PDBList(server='http://files.wwpdb.org')
    targetPdbDownloadBatchRun = []
    for pdb_id in pdbIds:
        pdbFile = os.path.join(masifpniOpts['raw_pdb_dir'], pdb_id + ".pdb")
        if os.path.exists(pdbFile): continue
        targetPdbDownloadBatchRun.append((masifpniOpts, pdb_id, pdbl, True))

    downloadDesc = "Download PDBs"
    resultList = batchRun1(targetPdbDownload, targetPdbDownloadBatchRun, n_threads=masifpniOpts["n_threads"],
                           desc=downloadDesc, batchRunFlag=batchRunFlag)
    unDownload = list(itertools.chain.from_iterable(resultList))
    with open(os.path.join(masifpniOpts["log_dir"], "unable_download.txt"), "w") as f:
        for i in unDownload:
            print(i, file=f)

    resumeFromChainPairs = False
    if not resumeFromChainPairs:
        pdbChainFullInfoBatchRun = []
        pdbHeaderInfoBatchRun = []
        pdbId2field = {}
        for i in pdbIdChains:
            fields = i.split("_")
            fields[0] = fields[0].upper()
            pid = fields[0]
            pdbId2field[pid] = fields
            pdbFile = os.path.join(masifpniOpts["raw_pdb_dir"], pid + ".pdb")
            if not os.path.exists(pdbFile): continue
            pdbChainFullInfoBatchRun.append((pid, pdbFile,))
            pdbHeaderInfoBatchRun.append((pdbFile,))

        ############### Get header of pdb files ###############
        findProteinChainBoundNADesc = "Get header of pdb files"
        resultList = batchRun1(getPdbHeaderInfo, pdbHeaderInfoBatchRun, n_threads=masifpniOpts["n_threads"],
                               desc=findProteinChainBoundNADesc, batchRunFlag=batchRunFlag)
        pdbHeaderInfo = list(itertools.chain.from_iterable(resultList))
        pdbHeader = ["name", "head", "idcode", "deposition_date", "release_date", "structure_method", "resolution"]
        pdbHeaderInfo = pd.DataFrame(pdbHeaderInfo, columns=pdbHeader)

        ############### Get chain full information in pdb files ###############
        findProteinChainBoundNADesc = "Get chain full information in pdb files"
        resultList = batchRun1(getPdbChainFullInfo, pdbChainFullInfoBatchRun,
                               n_threads=masifpniOpts["n_threads"], desc=findProteinChainBoundNADesc,
                               batchRunFlag=batchRunFlag)
        pdbChainFullInfo = list(itertools.chain.from_iterable(resultList))
        fullInfoHeader = ["PDB_id", "pChain", "naChain", "naType", "pChainSeq", "naChainSeq", "pChainLen", "naChainLen"]
        pdbChainFullInfo = pd.DataFrame(pdbChainFullInfo, columns=fullInfoHeader)
        pdbChainFullInfo = filterByUserFields(pdbChainFullInfo, pdbId2field)
        if pdbChainFullInfo.empty: return pd.DataFrame()
        aggregations = {'pChainSeq': "first", 'pChainLen': np.mean, 'naChainLen': np.mean, "naChain": "".join}
        pdbChainFullInfo = pdbChainFullInfo.groupby(["PDB_id", "pChain", "naType", "naChainSeq"]).agg(aggregations).reset_index()

        ############### Find protein-NA bounding chains ###############
        findProteinChainBoundNABatchRun = []
        pdbChainFullInfo['pChain'] = pdbChainFullInfo['pChain'].apply(lambda x: ''.join(sorted(x)))
        pdbChainFullInfo['naChain'] = pdbChainFullInfo['naChain'].apply(lambda x: ''.join(sorted(x)))
        for _, item in pdbChainFullInfo.iterrows():
            pdb_id = item.PDB_id
            pdbFile = os.path.join(masifpniOpts["raw_pdb_dir"], pdb_id + ".pdb")
            pChain = item.pChain
            naChain = item.naChain
            findProteinChainBoundNABatchRun.append([pdb_id, pdbFile, pChain, naChain])

        findProteinChainBoundNADesc = "Find protein-NA bounding chains"
        resultList = batchRun1(findProteinChainBoundNA2, findProteinChainBoundNABatchRun,
                               n_threads=masifpniOpts["n_threads"], desc=findProteinChainBoundNADesc,
                               batchRunFlag=batchRunFlag)
        pniChainPairs = list(itertools.chain.from_iterable(resultList))
        if len(pniChainPairs) == 0: return pd.DataFrame()
        pniChainPairsDf = pd.DataFrame(pniChainPairs, columns=["PDB_id", "pChain", "naChain", "naType"])

        ############### Filter pdb chain information with bound or not ###############
        keys = ["PDB_id", "pChain", "naChain"]
        i1 = pdbChainFullInfo.set_index(keys).index
        i2 = pniChainPairsDf.set_index(keys).index
        pdbChainFullInfo = pdbChainFullInfo[i1.isin(i2)]

        tempfileBase = os.path.basename(tempfile.mkdtemp())
        pniPairFile = os.path.join(masifpniOpts['tmp_dir'], tempfileBase + ".pdb_info.npz")
        np.savez(pniPairFile, chainInfo=pdbChainFullInfo, header=pdbHeaderInfo)
    else:
        try:
            tmp = np.load(masifpniOpts["pni_pairs_file"], allow_pickle=True)
        except:
            print("Please specify the right pni_pairs_file!")
            return
        chainInfoHeader = ["PDB_id", "pChain", "naChain", "naType", "pChainSeq", "naChainSeq", "pChainLen", "naChainLen"]
        pdbHeader = ["name", "head", "idcode", "deposition_date", "release_date", "structure_method", "resolution"]
        pdbChainFullInfo = pd.DataFrame(tmp["chainInfo"], columns=chainInfoHeader)
        pdbHeaderInfo = pd.DataFrame(tmp["header"], columns=pdbHeader)

    ############### Filter pdb chain information with length or resolution ###############
    filterPdb = False
    if filterPdb:
        pdbChainFullInfo = filterPdbChains(pdbChainFullInfo, pdbHeaderInfo)
    else:
        mergedDf = pd.merge(pdbChainFullInfo, pdbHeaderInfo, how="inner", left_on="PDB_id", right_on="idcode")
        selectColumns = ["PDB_id", "pChain", "naChain", "naType", "pChainSeq", "naChainSeq", "pChainLen", "naChainLen",
                         "resolution"]
        pdbChainFullInfo = mergedDf.loc[:, selectColumns]

    aggregations = {"pChain": "".join, "pChainSeq": "first", "naChainSeq": "first", "pChainLen": np.mean,
                    "naChainLen": np.mean, "resolution": "first"}
    pdbChainFullInfo = pdbChainFullInfo.groupby(["PDB_id", "naChain", "naType"]).agg(aggregations).reset_index()

    ############### Extract pdb surface information ###############
    if naType == "both":
        pdbChainFullInfo = pdbChainFullInfo
    elif naType == "RNA":
        pdbChainFullInfo = pdbChainFullInfo[pdbChainFullInfo.naType == "RNA"]
    else:
        pdbChainFullInfo = pdbChainFullInfo[pdbChainFullInfo.naType == "DNA"]
    return pdbChainFullInfo


def dataprep(argv):
    masifpniOpts = mergeParams1(argv)

    pdbIdChains = []
    if masifpniOpts["pdb_id_file"]:
        with open(masifpniOpts["pdb_id_file"]) as f:
            for line in f.readlines():
                if line.startswith("#"): continue
                pdbIdChains.append(line.strip().upper())
    if masifpniOpts["pdb_id_list"]:
        pdbIdChains.extend([i.upper() for i in masifpniOpts["pdb_id_list"].split(",")])

    if not len(pdbIdChains):
        print("Please input the PDB ids or PDB files that you want to prepare.")
        return

    pdbIdChains = list(set(pdbIdChains))
    if len(pdbIdChains) > 0:
        batchRunFlag = False if masifpniOpts["no_batch_run"] else True
        mergeChainFlag = masifpniOpts["merge_na_chain"]
        naType = masifpniOpts["na_type"]
        pdbChainFullInfo = dataprepFromList3(pdbIdChains, masifpniOpts, batchRunFlag=batchRunFlag, naType=naType)
        if not pdbChainFullInfo.empty:
            extractSurfaceInfo(masifpniOpts, pdbChainFullInfo, batchRunFlag=batchRunFlag, mergeNaChain=mergeChainFlag)
            # generate_NA_coord(masifpniOpts, pdbChainFullInfo, batchRunFlag=batchRunFlag, mergeNaChain=mergeChainFlag)

    # removeDirs([masifpniOpts["tmp_dir"]], empty=True)


if __name__ == '__main__':

    dataprep(sys.argv)

