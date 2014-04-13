import pandas as pd
import re
import numpy as np

from collections import defaultdict
from itertools import product

import sys

# df_cnt = pd.read_csv('data/gene.counts', header=None, names=['cnt','type','tag','name'], sep=' ') # wrong (irregular dimensions)

NUMERIC='_NUMERIC_'
ALL_CAPS='_ALL_CAPS_'
RARE='_RARE_'
LAST_CAP='_LAST_CAP_'

## parse it myself
def load_gene_counts(file_path):
    emissions = []
    gram_counts = []
    with open(file_path, 'r') as pile:
        for line in pile:
            cnt, tipe, rest = line.strip().split(' ', 2)
            if tipe == 'WORDTAG':
                tag, word = rest.split(' ')
                emissions.append( (tag, word, int(cnt)) )
            elif tipe.endswith('-GRAM'):
                m = re.match('([1-3])-GRAM', tipe)
                if m is not None:
                    n = int(m.group(1))
                    if n <= 0 or n >= 4:
                        raise ValueError("{}-GRAM unrecognized".format(n))
                    gram_counts.append( (tuple(rest.split()), int(cnt)) )
                else:
                    raise ValueError("Unknown -GRAM {}".format(tipe))
            else:
                raise ValueError("Unknown type {}".format(tipe))
    df_em = pd.DataFrame(emissions, columns=['tag','word','cnt']).set_index(['tag','word'])
    df_gram = pd.DataFrame([(k,v) for k,v in gram_counts],columns=['gram','cnt']).set_index(['gram'])
    # ok: df_gram.loc[[('I-GENE',)]]
    # not ok: df_gram.loc[('I-GENE',)] (KeyError), df_gram.loc['I-GENE',] (KeyError), df_gram.loc['I-GENE']
    return df_em, df_gram


def calc_word_class(word):
    if not word:
        return

    ## check numeric first because isupper() yields true for '1E3'
    for l in word:
        if l.isdigit():
            return NUMERIC

    if word.isupper():
        return ALL_CAPS

    if word[-1].isupper():
        return LAST_CAP

    return RARE


def replace_rare(df_em, part='p2'):
    acc = defaultdict(int)
    for (tag, word), cnt in df_em.itertuples():
        if part in ('p1','p2'):
            if cnt < 5:
                word = RARE
        elif part == 'p3':
            if cnt < 5:
                word = calc_word_class(word)
        acc[(tag, word)] += cnt
    df_em_rare = pd.DataFrame([(t,w,c) for (t,w),c in acc.iteritems()], columns=['tag','word','cnt']).set_index(['tag','word'])
    return df_em_rare


def emission(word, tag, df_em, dict_gram, part='p2'):
    """e(x|y) = count(y \to x)/count(y). y \to x means tag y emits word x"""
    if tag in ('*', 'STOP'):
        return 0

    ## use a dict instead, otherwise too slow
    # cnt_tag = df_gram.loc[[(tag,)],'cnt'][0]
    # if np.isnan(cnt_tag):
    #     raise ValueError('Unrecognized tag {}'.format(tag))

    cnt_tag = dict_gram.get((tag,), None)
    if cnt_tag is None:
        raise ValueError('Unrecognized tag {}'.format(tag))

    try:
        cnt_tag_emit_word = df_em.loc[tag, word]['cnt']
    except KeyError: # (tag,word) not in training
        ## is it tagged with any other tag? hardcode for now
        try:
            if tag == 'I-GENE':
                df_em.loc['O', word]['cnt']
            elif tag == 'O':
                df_em.loc['I-GENE', word]['cnt']
        except KeyError:
            pass
        else: # is tagged with another other tag, so it isn't rare in the sense that the word appears in the training data (just with a different tag)
            return 0
        ## without this block, F1 falls to ~0

        try:
            ## should always match if using df_em_rare
            if part in ('p1','p2'):
                word = RARE
            elif part == 'p3':
                word = calc_word_class(word)
            else:
                raise ValueError("Unknown part {}".format(part))
            cnt_tag_emit_word = df_em.loc[tag, word]['cnt']
        except KeyError:
            cnt_tag_emit_word = 0
    return 1.*cnt_tag_emit_word/cnt_tag


def tag_star(word, df_em, dict_gram):
    """unigram tagging"""
    best_val, best_tag = -float('inf'), None
    for tag, in dict_gram.keys(): # unigram only
        val = emission(word, tag, df_em, dict_gram, part='p1')
        if val > best_val:
            best_val, best_tag = val, tag
    return best_val, best_tag


def qparam(ym2, ym1, ym0, df_gram):
    """q(y_i|y_{i-2},y_{i-1})"""
    return float(1.*df_gram.loc[[(ym2, ym1, ym0)]].values/df_gram.loc[[(ym2, ym1)]].values)


def viterbi(sentence, df_em, df_gram, part='p2', debug=False):
    """Tag the sentence given the counted emissions and 1,2, or 3-grams. sentence input is the words x_1,\dots,x_n"""
    n = len(sentence)
    dict_gram = df_gram[df_gram.index.map(lambda r: len(r) == 1)]['cnt'].to_dict()

    S = [tag for tag, in dict_gram.keys()]
    dtab, pbtab, valtab = {(0,'*','*'): np.log10(1)}, {}, {} # valtab for debugging
    def _tags(k):
        return ['*'] if k <= 0 else S

    if debug:
        import pdb
        pdb.set_trace()

    for k in xrange(1,n+1):
        word = sentence[k-1]
        for u, v in product(_tags(k-1),_tags(k)):
            ## \max_{w \in S_{k-2}}
            max_val, max_arg = -float('inf'), None
            for w in _tags(k-2):
                em_val = np.log10(emission(word, v, df_em, dict_gram, part)) # probability the tag v would emit word
                q_val = np.log10(qparam(w, u, v, df_gram))
                dtab_val = dtab[(k-1,w,u)]
                ## previous most likely sentence so far, then probability of this trigram
                val = dtab_val +q_val +em_val
                valtab[(k,w,u,v)] = (val, em_val, q_val, dtab_val) # for debugging
                if val >= max_val:
                    max_val, max_arg = val, w

            idx = (k,u,v)
            dtab[idx], pbtab[idx] = max_val, max_arg

    ## calculate for the end of sentence
    max_val, max_arg = -float('inf'), None
    for u, v in product(_tags(n-1),_tags(n)):
        val = dtab[(n,u,v)] +np.log10(qparam(u, v, 'STOP', df_gram))
        if val >= max_val:
            max_val, max_arg = val, (u,v)

    ## go back in the chain ending with y_{n-1} and y_n
    out_tags = [None]*n
    out_tags[n -1] = max_arg[1] # y_n
    out_tags[n-1 -1] = max_arg[0] # y_{n-1}
    for k in xrange(n-2,1-1,-1):
        out_tags[k -1] = pbtab[(k+2, out_tags[(k+1) -1], out_tags[(k+2) -1])]

    ## for debugging
    df_dtab = pd.DataFrame([(k, sm1, s, val) for (k, sm1, s), val in dtab.iteritems()], columns=['k','sm1','s','val']).sort(['k','sm1','s']).set_index(['k','sm1','s'])
    df_pbtab = pd.DataFrame([(k, sm1, s, tag) for (k, sm1, s), tag in pbtab.iteritems()], columns=['k','sm1','s','tag']).sort(['k','sm1','s']).set_index(['k','sm1','s'])
    df_valtab = pd.DataFrame([(k, sm2, sm1, s, val, em_val, q_val, dtab_val) for (k, sm2, sm1, s), (val, em_val, q_val, dtab_val) in valtab.iteritems()], columns=['k','sm2','sm1','s','pi','e','q','dtab']).sort(['k','sm1','s','sm2']).set_index(['k','sm2','sm1','s'])
    return out_tags, df_dtab, df_pbtab, df_valtab


def read_sentences(doc_fpath):
    """where the document separates each token (including newlines) into its own line"""
    acc = []
    with open(doc_fpath, 'r') as pile:
        sentence = []
        for line in pile:
            word = line.strip().split(' ')[0] # should handle "word" and "word tag" files
            if not word: # end of sentence
                acc.append(sentence) if sentence else None # ignore newline only sentences
                sentence = []
            else:
                sentence.append(word)
    return acc


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('part', help='part to run') #  nargs=1, produces a list, while nothing produes the value itself
    parser.add_argument('--gene-data', help='document of sentences (default: gene.dev)', default='gene.dev')
    parser.add_argument('--emission', help='emission table to use (default: df_em_rare)', default='df_em_rare')
    args = parser.parse_args()

    if args.part not in ('p1','p2','p3'):
        raise ValueError('unknown part specified')

    ## init
    gene_counts_fpath = 'data/gene.counts'

    # df_gram y_{i-2} y_{i-1} y
    df_em_raw, df_gram = load_gene_counts(gene_counts_fpath)
    df_em_rare = replace_rare(df_em_raw, part=args.part) # new counts with infrequent words replaced
    df_em_kept_rare = df_em_raw.copy()
    df_em_kept_rare = df_em_kept_rare.append(df_em_rare[df_em_rare.index.map(lambda r: r[1] in [RARE])])

    df_em_wo_rare = df_em_rare[df_em_rare.index.map(lambda r: r[1] != RARE)]

    dict_gram = df_gram[df_gram.index.map(lambda r: len(r) == 1)]['cnt'].to_dict()

    if args.emission == 'df_em_rare':
        df_em = df_em_rare
    elif args.emission == 'df_em_wo_rare':
        df_em = df_em_wo_rare
    elif args.emission == 'df_em_kept_rare':
        df_em = df_em_kept_rare

    if args.gene_data == 'gene.dev':
        gene_fpath = 'data/gene.dev'
    elif args.gene_data == 'gene.test':
        gene_fpath = 'data/gene.test'

    sentences = read_sentences(gene_fpath)

    ## parts
    if args.part == 'p1':
        acc_tags = []
        for sentence in sentences:
            acc = []
            for word in sentence:
                val, tag = tag_star(word, df_em, dict_gram)
                acc.append(tag)
            acc_tags.append(acc)
        for sentence, tags in zip(sentences,acc_tags):
            for word, tag in zip(sentence, tags):
                sys.stdout.write('{} {}\n'.format(word, tag))
            sys.stdout.write('\n')

    elif args.part in ('p2','p3'):
        acc = []
        for sentence in sentences:
            tags, df_dtab, df_pbtab, df_valtab = viterbi(sentence, df_em, df_gram, part=args.part, debug=False)
            acc.append((sentence, tags, df_dtab, df_pbtab, df_valtab))

        for sentence, tags, _, _, _ in acc:
            for word, tag in zip(sentence, tags):
                sys.stdout.write('{} {}\n'.format(word, tag))
            sys.stdout.write('\n')
