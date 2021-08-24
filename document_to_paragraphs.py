import os
import sys
import logging
from argparse import ArgumentParser
import pandas as pd


def parse_args(argv):
    parser = ArgumentParser(
        description='A utility to convert documents to paragraphs.')

    parser.add_argument('input',
                        help='A Parquet file, e.g. train.parq')

    parser.add_argument('-v', '--verbose',
                        action='store_true',
                        default=False,
                        help='Specifies the name of the output JSON lines file. Default: %(deafult)s')

    args = parser.parse_args(argv)

    return parser, args


def convert_document_to_paragraphs(path):
    data = pd.read_parquet(path)
    dfs = []
    for j in range(data.shape[0]):
        document = data.iloc[j]['input']
        output = data.iloc[j]['output']
        doc_id = data.iloc[j]['id']
        document_split = document.split()
        output_split = output.split()
        split_indexes = [-1] + [i for i, x in enumerate(document_split) if x == "."]
        split_indexes = split_indexes[::5]
        documents = []
        outputs = []
        ids = []
        for i, ind in enumerate(split_indexes):
            if i == len(split_indexes) - 1:
                new_document = ' '.join(document_split[ind + 1:])
                new_output_list = output_split[ind + 1:]
            else:
                new_document = ' '.join(document_split[ind + 1:split_indexes[i + 1] + 1])
                new_output_list = output_split[ind + 1:split_indexes[i + 1] + 1]
            if not new_document:
                continue
            original_cluster_index = sorted(set(new_output_list), key=new_output_list.index)
            original_cluster_index = [x for x in original_cluster_index if x != '-']
            num_unique_clusters = len(original_cluster_index)
            new_cluster_index = list(range(1, num_unique_clusters + 1))
            replacements = {}
            for old_id, new_id in zip(original_cluster_index, new_cluster_index):
                replacements[old_id] = str(new_id)
            renumbered_new_output_list = [replacements.get(x, x) for x in new_output_list]
            new_output = ' '.join(renumbered_new_output_list)
            documents.append(new_document)
            outputs.append(new_output)
            ids.append(doc_id)
        new_augmented_data = pd.DataFrame({'id': ids, 'input': documents, 'output': outputs})
        new_augmented_data['input'] = "coref: " + new_augmented_data['input']
        dfs.append(new_augmented_data)
    full_augmented_data = pd.concat(dfs)
    if 'test' not in path:
        full_augmented_data['seq_len'] = full_augmented_data['input'].apply(lambda x: len(x.split()))
        full_augmented_data = full_augmented_data[(full_augmented_data['seq_len'] <= 512) & (full_augmented_data['seq_len'] >= 4)]
        full_augmented_data = full_augmented_data.drop(columns=['seq_len'])
    full_augmented_data = full_augmented_data.reset_index(drop=True)

    output_path = os.path.splitext(os.path.basename(path))[0] + '_with_paragraphs.parq'
    full_augmented_data.to_parquet(path=output_path)
    logging.info(f"Generated: {output_path}")

    output_path = os.path.splitext(os.path.basename(path))[0] + '_with_paragraphs.pkl'
    full_augmented_data.to_pickle(output_path)
    logging.info(f"Generated: {output_path}")


def init_logging(verbose):
    level_ = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level_, format='%(message)s')


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    parser, args = parse_args(argv)

    init_logging(args.verbose)

    convert_document_to_paragraphs(args.input)


if __name__ == '__main__':
    exit(main())
