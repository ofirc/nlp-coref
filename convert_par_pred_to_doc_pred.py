import os
import sys
import logging
from argparse import ArgumentParser
import pandas as pd


def parse_args(argv):
    parser = ArgumentParser(
        description='A utility to convert paragraph prediction to document level prediction for iter-paragraphs f1.')

    parser.add_argument('input',
                        help='A CSV file, e.g. predictions.csv')

    parser.add_argument('-v', '--verbose',
                        action='store_true',
                        default=False,
                        help='Specifies the name of the output JSON lines file. Default: %(deafult)s')

    args = parser.parse_args(argv)

    return parser, args


def func(x):
    l = []
    for i in x.split():
        if i.isnumeric():
            l.append(int(i))
    l = sorted(set(l))
    return l

def concat_paragraphs_preds(par_preds_per_doc):
    par_preds_per_doc = par_preds_per_doc.reset_index(drop=True)
    #par_preds_per_doc['clusters_ids'] = par_preds_per_doc['Generated Text'].apply(lambda x: sorted(list(set([int(i) for i in x.split() if i != '-']))))
    par_preds_per_doc['clusters_ids'] = par_preds_per_doc['Generated Text'].apply(func)
    par_preds_per_doc['clusters_ids'] = par_preds_per_doc['clusters_ids'].apply(lambda x: [str(i) for i in x])
    par_preds_per_doc['num_of_clusters'] = par_preds_per_doc['clusters_ids'].apply(lambda x: len(x))
    all_clusters_doc = list(range(1, par_preds_per_doc['num_of_clusters'].sum() + 1))
    all_clusters_doc = [str(i) for i in all_clusters_doc]
    preds_list = []
    for j in range(par_preds_per_doc.shape[0]):
        par_pred = par_preds_per_doc.iloc[j]['Generated Text']
        clusters_ids = par_preds_per_doc.iloc[j]['clusters_ids']
        num_of_clusters = par_preds_per_doc.iloc[j]['num_of_clusters']
        new_clusters_ids = all_clusters_doc[:num_of_clusters]
        par_pred_list = par_pred.split()
        replacements = {}
        for old_cluster, new_clusters in zip(clusters_ids, new_clusters_ids):
            replacements[old_cluster] = str(new_clusters)
        new_par_pred_list = [replacements.get(x, x) for x in par_pred_list]
        new_par_pred = ' '.join(new_par_pred_list)
        all_clusters_doc = all_clusters_doc[num_of_clusters:]
        preds_list.append(new_par_pred)
    document_level_generated_text = ' '.join(preds_list)
    return document_level_generated_text


def process_predictions(path):
    par_preds = pd.read_csv(path, index_col=0)
    doc_preds = pd.DataFrame(par_preds.groupby('id').apply(lambda x: concat_paragraphs_preds(x)))
    doc_preds = doc_preds.reset_index().rename(columns={0: 'Generated Text'})
    output_path = os.path.splitext(os.path.basename(path))[0] + '_document_level.parq'
    doc_preds.to_parquet(path=output_path)
    logging.info(f"Generated: {output_path}")


def init_logging(verbose):
    level_ = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level_, format='%(message)s')


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    parser, args = parse_args(argv)

    init_logging(args.verbose)

    process_predictions(args.input)


if __name__ == '__main__':
    exit(main())
