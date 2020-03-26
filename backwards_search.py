# -*- coding: utf-8 -*-
from __future__ import print_function

import os
import re
import sys
import uuid
from difflib import get_close_matches
from multiprocessing import Pool

import pandas
from pandas import DataFrame
from pybtex.database import parse_file
from pybtex.errors import set_strict_mode
from pybtex.style.formatting.unsrt import Style
from refextract import extract_references_from_file
from refextract.references.errors import FullTextNotAvailableError
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


class RemoveUrlsStyle(Style):
    def format_web_refs(self, e):
        # based on urlbst output.web.refs
        return None


_style = RemoveUrlsStyle(abbreviate_names=True)


def closest_filename(path):
    filename = os.path.basename(path)
    folder = os.path.dirname(path)
    candidates = os.listdir(folder)
    found_name = next(iter(get_close_matches(filename, candidates, n=1, cutoff=.8)), '')
    return os.path.join(folder, found_name)


def get_references_for_entry(entry):
    if "file" not in entry.fields:
        return
    filename = '/' + entry.fields["file"].split(':')[1]
    try:
        references = extract_references_from_file(filename)
    except FullTextNotAvailableError as e:
        filename = closest_filename(filename)
        try:
            references = extract_references_from_file(filename)
        except FullTextNotAvailableError as e:
            print("file not found " + filename, file=sys.stderr)
            return
    return entry, references


def get_entry_name(entry):
    try:
        return _style.format_entry(entry.key, entry).text.render_as('text')
    except BaseException as e:
        print(e, file=sys.stderr)
        return re.sub(r'(^{|}$)', '', entry.fields.get('title', ''))


def clean_raw_ref(raw_ref):
    return re.sub(r"^\s*(\[[^\]]+\]|\d+[\.\)])\s*", "", raw_ref)


def backwards_search(file):
    set_strict_mode(False)
    bibfile = parse_file(file)
    entries = bibfile.entries.values()
    #entries = entries[0:7]

    pool = Pool(processes=8)
    rows = []
    for result in pool.imap(get_references_for_entry, entries):
        if result:
            entry, refs = result
            entry_name = get_entry_name(entry)
            for ref in refs:
                ref_name = clean_raw_ref(ref["raw_ref"][0])
                rows.append((entry.key, entry_name, ref_name))

    df = DataFrame(rows, columns=['paper_key', 'paper_name', 'ref_name'])
    return df


def group_similar(df):
    df.drop_duplicates(inplace=True)
    df.reset_index(inplace=True, drop=True)

    # extract features
    tf_idf_matrix = extract_features(df)

    # create groups
    groups = create_groups(df, tf_idf_matrix)

    # add group info to df
    df = apply_groups(df, groups)

    return df


def apply_groups(df, groups):
    ref_count = len(df['ref_name'])
    df.loc[:, 'ref_key'] = None
    df.loc[:, 'count'] = 1
    for grp in groups:
        # create a group id, either a paper key, or if a none can be found a random uuid
        grpid = "__" + str(uuid.uuid4())
        for idx in grp:
            if idx >= ref_count:
                grpid = df.at[idx - ref_count, 'paper_key']
                break
        print('----' + grpid)
        grp = [idx for idx in grp if idx < ref_count]
        for idx in grp:
            print(df.at[idx, 'ref_name'])
            df.at[idx, 'ref_key'] = grpid
            df.at[idx, 'count'] = len(grp)

    return df


def create_groups(df, tf_idf_matrix):
    groups = set()
    # process all paper names and all ref names
    for i, row in df.iterrows():
        cosine_similarities = linear_kernel(tf_idf_matrix[i:i + 1], tf_idf_matrix).flatten()
        related_docs_indices = cosine_similarities.argsort()
        idx_to_sim = zip(related_docs_indices, cosine_similarities[related_docs_indices])
        # find indexes of all string similar to the given one
        idxs = frozenset(p[0] for p in idx_to_sim if p[1] > .5)

        # if merge all overlapping existing groups into the current group
        for g in groups.copy():
            if g & idxs:
                idxs = g | idxs
                groups.remove(g)
        groups.add(idxs)
    return groups


def extract_features(df):
    ref_names = list(df['ref_name'].values.astype('U'))
    paper_names = list(df['paper_name'].values.astype('U'))
    stop_words = ['proceedings', 'conference', 'in', 'of', 'the', 'acm', 'sigsac', 'ieee', 'http', 'https']
    tfidf = TfidfVectorizer(analyzer='word', stop_words=stop_words)  # ngram_range=(3,3)
    # train on refnames and paper names, so we can also later find papers corresponding to a reference
    tf_idf_matrix = tfidf.fit_transform(ref_names + paper_names)
    return tf_idf_matrix


if __name__ == '__main__':
    # collect references from all papers in a bibtex file (exported from mendeley)
    search = True
    if search:
        df = backwards_search('./My Collection.bib')
        df.to_csv('refs.csv', encoding='utf-8-sig')
    else:
        df = pandas.read_csv('refs.csv', encoding='utf-8-sig')

    # group references with similar titles
    df = group_similar(df)
    df.to_csv('refs_grouped.csv', encoding='utf-8-sig')
    #df.to_excel("refs_grouped.xlsx", encoding='utf-8-sig')


