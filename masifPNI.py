#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
File name: masifPNI.py
Author: CrazyHsu @ crazyhsu9627@gmail.com
Created on: 2022-09-11 20:31:49
Last modified: 2022-09-11 20:31:49
'''

import sys, warnings
import multiprocessing
from argparse import ArgumentParser

warnings.filterwarnings("ignore")

__version__ = "1.0.0"

defaultArguments = ["masifPNI.py", "masifPNI-site", "dataprep", "train", "predict"]


def parseArgsDownload(parser, argv):
    parser.add_argument('-l', '--pdb_id_list', type=str, default=None, help="Lists of PDB ids, separated by comma")
    parser.add_argument('-f', '--pdb_id_file', type=str, default=None, help="File contains lists of PDB ids. One per separate line.")
    parser.add_argument('-a', "--all_pdbs", action="store_true", default=False, help="Download all PDB entries.")
    parser.add_argument('--overwrite_pdb', action="store_true", default=False, help="Overwrite existing PDB files.")
    parser.add_argument('-v', '--version', action='version', version='%(prog)s {version}'.format(version=__version__))
    from pdbDownload import pdbDownload
    parser.set_defaults(func=pdbDownload)


def parseArgsSite(parser, argv):
    parser.add_argument('-v', '--version', action='version', version='%(prog)s {version}'.format(version=__version__))
    subparsers = parser.add_subparsers(help="Modes to run masifPNI-site", metavar='[dataprep, train, predict]')
    parent_parser = ArgumentParser(add_help=False)
    parent_parser.add_argument('--config', dest='config', help='Config file contains the parameters to run masifPNI.', type=str)
    parent_parser.add_argument('-o', '--out_base_dir', type=str, default=None, help="Output directory")
    parent_parser.add_argument('-n', "--n_threads", type=int, default=1, help="The number of threads to download pdb files.")
    parent_parser.add_argument('--na_type', type=str, choices=["RNA", "DNA", "both"], default="RNA", help="Extract surface information for the specified nucleic acid type. RNA or DNA or both. Default: RNA")
    parent_parser.add_argument('--no_batch_run', action="store_true", default=False, help="Don't batch run the program.")
    parent_parser.add_argument('--merge_na_chain', action="store_true", default=False, help="Merge all satisfied nucleic acid chains in a PDB file.")
    parent_parser.add_argument('-r', '--radius', type=float, default=9, help="The radius used to count vertices in a patch.")
    parent_parser.add_argument('-max_n', '--max_vertices_n', type=int, default=100, help="The maximum number of vertices in a patch.")

    parser_dataprep = subparsers.add_parser("dataprep", parents=[parent_parser], help="Prepare the data used in next steps")
    optional_dataprep = parser_dataprep._action_groups.pop()
    optional_dataprep.add_argument('-l', '--pdb_id_list', type=str, default=None, help="Lists of PDB ids, separated by comma")
    optional_dataprep.add_argument('-f', '--pdb_id_file', type=str, default=None, help="File contains lists of PDB ids. One per separate line.")
    optional_dataprep.add_argument('--overwrite_pdb', action="store_true", default=False, help="Overwrite existing PDB files.")
    optional_dataprep.add_argument('-v', '--version', action='version', version='%(prog)s {version}'.format(version=__version__))
    parser_dataprep._action_groups.append(optional_dataprep)
    from dataPreparation import dataprep
    parser_dataprep.set_defaults(func=dataprep)

    parent_parser.add_argument('--model_dir', dest='model_dir', type=str, default=None, help='The output directory used to save model.')
    parent_parser.add_argument('--n_theta', dest='n_theta', type=int, default=4, help='The number of average divisions of the 2π central angle. Default: 4')
    parent_parser.add_argument('--n_rho', dest='n_rho', type=int, default=3, help='The number of segments to divide the radius evenly along the axis. Default: 3')
    parent_parser.add_argument('--n_rotations', dest='n_rotations', type=int, default=4, help='The number of times to rotate the circle. Default: 4')

    parser_train = subparsers.add_parser("train", parents=[parent_parser], help="Train the neural network model with protein-RNA interaction complex files")
    parser_train.add_argument('--training_list', dest='training_list', type=str, default=None, help='A file contains the list of PDB ids used to train nucleic network model.')
    parser_train.add_argument('--testing_list', dest='testing_list', type=str, default=None, help='A file contains the list of PDB ids used to test nucleic network model.')
    parser_train.add_argument('--filterChainByLen', action="store_true", default=False, help="Filter chains by length.")
    parser_train.add_argument('--draw_roc', dest='draw_roc', action="store_true", default=False, help='Whether to draw ROC plot.')
    parser_train.add_argument('-v', '--version', action='version', version='%(prog)s {version}'.format(version=__version__))
    from masifPNI_site.masifPNI_site_train import train_masifPNI_site2
    parser_train.set_defaults(func=train_masifPNI_site2)

    parser_predict = subparsers.add_parser("predict", parents=[parent_parser], help="Predict the protein-RNA complex")
    parser_predict.add_argument('-l', '--pdb_id_list', type=str, default=None, help="Lists of PDB ids, separated by comma")
    parser_predict.add_argument('-f', '--pdb_id_file', type=str, default=None, help="File contains lists of PDB ids. One per separate line.")
    parser_predict.add_argument('-custom_pdb', '--custom_pdb', type=str, default=None, help="File contain the path of custom PDB files or the directory contains the target PDB files")
    parser_predict.add_argument('-v', '--version', action='version', version='%(prog)s {version}'.format(version=__version__))
    from masifPNI_site.masifPNI_site_predict import masifPNI_site_predict
    parser_predict.set_defaults(func=masifPNI_site_predict)

    opt2parser = {"dataprep": parser_dataprep, "train": parser_train, "predict": parser_predict}
    if len(set(argv) - set(defaultArguments)) == 0:
        if "masifPNI-site" in argv:
            tmp = list(set(argv) & set(["dataprep", "train", "predict"]))
            if len(tmp) == 1:
                tmpParser = opt2parser[tmp[0]]
                tmpParser.print_help()
                tmpParser.exit()
            else:
                parser.print_help()
                parser.exit()


def parseOptions1(argv):
    parser = ArgumentParser(prog='masifPNI')

    parent_parser = ArgumentParser(add_help=False)
    parent_parser.add_argument('--config', dest='config', help='Config file contains the parameters to run masifPNI.', type=str)
    parent_parser.add_argument('-o', '--out_base_dir', type=str, default=None, help="Output directory")
    parent_parser.add_argument('-n', "--n_threads", type=int, default=1, help="The number of threads to download pdb files.")
    parent_parser.add_argument('--no_batch_run', action="store_true", default=False, help="Don't batch run the program.")

    subparsers = parser.add_subparsers(help='Running modes', metavar='[download, masifPNI-site]')
    parser.add_argument('-v', '--version', action='version', version='%(prog)s {version}'.format(version=__version__))

    parser_download = subparsers.add_parser('download', parents=[parent_parser], help='Download target PDB files')
    parseArgsDownload(parser_download, argv)

    ### RUN MODE "masifPNI-site"
    parser_masifPNI_site = subparsers.add_parser('masifPNI-site', help='Run masifPNI-site')
    parseArgsSite(parser_masifPNI_site, argv)

    if len(argv) == 1:
        parser.print_help()
        parser.exit()

    return parser.parse_args(argv[1:])


def main(argv=sys.argv):
    options = parseOptions1(argv)
    options.func(options)


if __name__ == '__main__':
    main(sys.argv)
