#!/usr/bin/python
import os
import sys
import json
import logging

from argparse import ArgumentParser

import pandas as pd


class Cluster(object):
    def __init__(self, mentions=None):
        self.mentions = mentions or []


class Mention(object):
    def __init__(self, start, end):
        assert isinstance(start, int)
        assert isinstance(end, int)
        self.start = start
        self.end = end
        self.enabled = True
        self.cluster_id = None

    @property
    def enabled(self):
        return self._enabled

    @enabled.setter
    def enabled(self, value):
        self._enabled = value

    def __str__(self):
        return '[{}, {}]'.format(self.start, self.end)

    def overlaps_with(self, mention):
        is_not_overlapping = mention.start > self.end or mention.end < self.start
        return not is_not_overlapping


class Sentence(object):
    def __init__(self, text, span_text):
        self.text = text
        self.span_text = span_text


class Document(object):
    def __init__(self):
        self.sentences = []

    def add_sentence(self, sentence):
        pass


def parse_args(argv):
    parser = ArgumentParser(
        description='A utility to convert a span-based JSON lines input onto text based JSON lines output.')

    parser.add_argument('input',
                        help='A json lines files, e.g. train.jsonlines')

    parser.add_argument('-o', '--output',
                        default='output.jsonlines',
                        help='Specifies the name of the output JSON lines file. Default: %(deafult)s')

    parser.add_argument('-v', '--verbose',
                        action='store_true',
                        default=False,
                        help='Specifies the name of the output JSON lines file. Default: %(deafult)s')

    parser.add_argument('-n', '--num-lines',
                        metavar='int',
                        type=int,
                        default=0,
                        help='Specifies the number of lines to process. Default: %(deafult)s')

    parser.add_argument('-m', '--max-sentences',
                        metavar='int',
                        type=int,
                        default=0,
                        help='Specifies the number of sentences per document to process. Default: %(deafult)s')

    parser.add_argument('-a', '--augment',
                        action='store_true',
                        default=False,
                        help='Augments the dataset. Default: %(default)s')

    args = parser.parse_args(argv)

    return parser, args


def load_clusters_with_mentions(clusters):
    clusters_with_mentions = []
    for cluster in clusters:
        mentions = []
        for mention in cluster:
            start, end = mention
            mentions.append(Mention(start, end))
        cluster = Cluster(mentions)
        clusters_with_mentions.append(cluster)
    return clusters_with_mentions


def find_overlapping_clusters(mention, clusters_with_mentions):
    overlapping_mentions = [mention]
    for cluster in clusters_with_mentions:
        for cur_mention in cluster.mentions:
            if cur_mention == mention:
                continue
            if cur_mention.enabled and mention.overlaps_with(cur_mention):
                overlapping_mentions.append(cur_mention)
    return overlapping_mentions


def sample_one_mention(overlapping_mentions):
    if len(overlapping_mentions) == 1:
        return overlapping_mentions[0]
    assert len(overlapping_mentions) > 1
    return overlapping_mentions[0]


def find_and_filter_overlapping_clusters(clusters_with_mentions):
    has_overlaps = True

    iteration = 0

    while has_overlaps:
        assert iteration < 2
        has_overlaps = False
        for cluster_id, cluster in enumerate(clusters_with_mentions):
            for mention in cluster.mentions:
                if not mention.enabled:
                    continue
                overlapping_mentions = find_overlapping_clusters(mention, clusters_with_mentions)
                if len(overlapping_mentions) > 1:
                    has_overlaps = True
                    logging.debug("Found overlapping mentions: %s", ' '.join([str(m) for m in overlapping_mentions]))
                    selected_mention = sample_one_mention(overlapping_mentions)
                    logging.debug("Sampled %s", str(selected_mention))
                    overlapping_mentions.remove(selected_mention)
                    for other_mention in overlapping_mentions:
                        logging.debug("Disabling %s from cluster %d", str(other_mention), cluster_id)
                        other_mention.enabled = False
        iteration += 1


def add_numbers_to_mentions(clusters_with_mentions):
    for cluster_id, cluster in enumerate(clusters_with_mentions):
        for mention in cluster.mentions:
            mention.cluster_id = cluster_id


def map_word_ids_to_cluster_ids(clusters_with_mentions):
    d = {}
    for cluster in clusters_with_mentions:
        for mention in cluster.mentions:
            if not mention.enabled:
                continue
            for word_idx in range(mention.start, mention.end + 1):
                assert word_idx not in d
                d[word_idx] = mention.cluster_id
    return d


def parse_one_document(lineno, line, num_sentences_per_document):
    logging.debug(f'Processing line {lineno + 1}')
    data = json.loads(line.strip())

    doc_key = data['doc_key']
    clusters = data['clusters']
    clusters_with_mentions = load_clusters_with_mentions(clusters)

    find_and_filter_overlapping_clusters(clusters_with_mentions)
    add_numbers_to_mentions(clusters_with_mentions)

    word_id_2_cluster_index = map_word_ids_to_cluster_ids(clusters_with_mentions)

    df = pd.DataFrame(["input", "output"])

    global_word_index = 0

    input_document = ""
    output_document = ""

    for sentence_id, sentence in enumerate(data['sentences']):
        if num_sentences_per_document > 0 and sentence_id == num_sentences_per_document:
            break
        input_sentence = ""
        output_sentence = ""
        last_word_idx_in_sentence = len(sentence) - 1
        for word in sentence:
            cluster_id = word_id_2_cluster_index.get(global_word_index)
            if cluster_id:
                cluster_id = str(cluster_id)
            else:
                cluster_id = "-"
            input_document += word + " "
            output_document += cluster_id + " "
            global_word_index += 1
    input_document = input_document.strip()
    output_document = output_document.strip()

    return doc_key, input_document, output_document

def init_logging(verbose):
    level_ = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level_, format='%(message)s')


def parse_jsonlines(path, num_lines=0, max_sentences=None):
    docs = []
    with open(path, 'r') as f:
        for lineno, line in enumerate(f):
            if num_lines > 0 and lineno == num_lines:
                break
            doc_key, input_document, output_document = parse_one_document(lineno, line, max_sentences)
            docs.append([doc_key, input_document, output_document])

    df = pd.DataFrame(docs, columns=['id', 'input', 'output'])
    output_path = os.path.splitext(os.path.basename(path))[0] + '.parq'
    df.to_parquet(path=output_path)
    logging.info(f"Generated: {output_path}")


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    parser, args = parse_args(argv)

    init_logging(args.verbose)

    parse_jsonlines(args.input, args.num_lines, args.max_sentences)


if __name__ == '__main__':
    exit(main())
