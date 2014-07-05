import pandas as pd
import numpy as np
from itertools import product
import sys
from collections import defaultdict

from hmm import load_gene_counts, read_sentences

def read_sentence_tags(doc_fpath):
    """where the document separates each token (including newlines) into its own line"""
    sentences = []
    tag_lines = []
    with open(doc_fpath, 'r') as pile:
        sentence = []
        tags = []
        for line in pile:
            tup = line.strip().split(' ')
            if len(tup) == 1: # end of sentence
                sentences.append(sentence) if sentence else None # ignore newline only sentences
                tag_lines.append(tags) if tags else None
                sentence = []
                tags = []
            else:
                word, tag = tup
                sentence.append(word)
                tags.append(tag)
    return sentences, tag_lines


def _read_tag_model(file_path):
    acc = {}
    with open(file_path, 'r') as pile:
        for line in pile:
            feature, weight = line.strip().split(' ')
            acc[feature] = float(weight)
        df = pd.DataFrame.from_dict(acc, orient='index')
        df.columns = ['weight']
        df.index.name = 'feature'
        return df


def _trigram_key(a,b,c):
    return 'TRIGRAM:{}:{}:{}'.format(a,b,c)

def _tag_key(word, tag):
    return 'TAG:{}:{}'.format(word,tag)

def _suff_keys(word, tag):
    suff_keys = []
    for idx in range(1,3+1):
        suff = word[-idx:]
        if len(suff) != idx: # ie. not enough letters remaining
            continue
        suff_keys.append( 'SUFF:{}:{}:{}'.format(suff,idx,tag))
    return suff_keys


def _vf(w,u,v,word, dict_tag_model):
    tri_key = _trigram_key(w,u,v)
    tag_key = _tag_key(word, v)
    suff_keys = _suff_keys(word, v)

    ## dot product on weights and if it exists
    suff_val = sum([dict_tag_model.get(sk, 0) for sk in suff_keys])
    vg_val = dict_tag_model.get(tri_key, 0) +dict_tag_model.get(tag_key, 0) +suff_val
    return vg_val



def viterbi(sentence, dict_gram, dict_tag_model):
    """calculate y^* = \arg\max_{t_1,\dots,t_n \in GEN(x)} f(x,y) \cdot v. x is the sentence history"""

    n = len(sentence)

    S = [tag for tag, in dict_gram.keys()]
    dtab, pbtab, valtab = {(0,'*','*'): 0}, {}, {} # valtab for debugging
    def _tags(k):
        return ['*'] if k <= 0 else S

    for k in xrange(1,n+1):
        word = sentence[k-1] # idx-0
        for u, v in product(_tags(k-1),_tags(k)):
            ## \max_{w \in S_{k-2}}
            max_val, max_arg = -float('inf'), None
            for w in _tags(k-2):
                vg_val = _vf(w,u,v,word, dict_tag_model)
                dtab_val = dtab[(k-1,w,u)]
                ## previous most likely sentence so far, then probability of this trigram
                val = dtab_val +vg_val
                if val > max_val:
                    max_val, max_arg = val, w

            idx = (k,u,v)
            dtab[idx], pbtab[idx] = max_val, max_arg

    ## calculate for the end of sentence
    max_val, max_arg = -float('inf'), None
    for u, v in product(_tags(n-1),_tags(n)):
        val = dtab[(n,u,v)] +_vf(u,v,'STOP',word, dict_tag_model)
        if val > max_val:
            max_val, max_arg = val, (u,v)

    ## go back in the chain ending with y_{n-1} and y_n
    out_tags = [None]*n
    out_tags[n -1] = max_arg[1] # y_n
    out_tags[n-1 -1] = max_arg[0] # y_{n-1}
    for k in xrange(n-2,1-1,-1):
        out_tags[k -1] = pbtab[(k+2, out_tags[(k+1) -1], out_tags[(k+2) -1])]

    return out_tags



def perceptron(sentences, tag_lines, n_iter, dict_gram):
    dict_tag_model = defaultdict(int) # v
    def _calc_features(tags, sentence):
        tmp_features = defaultdict(int)

        for tag,word in zip(tags,sentence):
            ## emissions
            tmp_features[_tag_key(word, tag)] += 1
            ## suffixes
            for sk in _suff_keys(word, tag):
                tmp_features[sk] += 1

        ## trigrams
        tmp_ts = ['*','*'] +tags +['STOP']
        for w,u,v in zip(tmp_ts,tmp_ts[1:],tmp_ts[2:]):
            tmp_features[_trigram_key(w,u,v)] += 1
        
        return dict(tmp_features)

    for _ in xrange(n_iter):
        for sentence, in_tags in zip(sentences,tag_lines):
            out_tags = viterbi(sentence, dict_gram, dict_tag_model) # best tag sequence (GEN) under the current model
            tmp_in_features = _calc_features(in_tags, sentence)
            tmp_out_features = _calc_features(out_tags, sentence)
            if tmp_in_features != tmp_out_features:
                ## update weight vector v
                for key,val in tmp_in_features.iteritems():
                    dict_tag_model[key] += val
                for key,val in tmp_out_features.iteritems():
                    dict_tag_model[key] -= val
    return dict_tag_model



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('part', help='part to run') #  nargs=1, produces a list, while nothing produes the value itself
    parser.add_argument('--gene-data', help='document of sentences (default: gene.dev)', default='data/gene.dev')
    parser.add_argument('--model', help='', default='data/tag.model')
    args = parser.parse_args()

    gene_fpath = args.gene_data

    ## init
    gene_counts_fpath = 'data/gene.counts'

    # df_gram y_{i-2} y_{i-1} y
    df_em_raw, df_gram = load_gene_counts(gene_counts_fpath)
    dict_gram = df_gram[df_gram.index.map(lambda r: len(r) == 1)]['cnt'].to_dict()

    if args.part == 'p1':
        sentences, tag_lines = read_sentence_tags(gene_fpath)
        df_tag_model = _read_tag_model(args.model)
        dict_tag_model = df_tag_model['weight'].to_dict()

        acc = []
        for sentence in sentences:
            tags = viterbi(sentence, dict_gram, dict_tag_model)
            acc.append((sentence, tags))
        for sentence, tags in acc:
            for word, tag in zip(sentence, tags):
                sys.stdout.write('{} {}\n'.format(word, tag))
            sys.stdout.write('\n')

    elif args.part == 'p2a':
        sentences, tag_lines = read_sentence_tags(gene_fpath)
        dict_tag_model = perceptron(sentences, tag_lines, 6, dict_gram)
        for key,val in dict_tag_model.iteritems():
            sys.stdout.write('{} {}\n'.format(key, val))
    elif args.part == 'p2b':
        sentences = read_sentences(gene_fpath)
        df_tag_model = _read_tag_model(args.model)
        dict_tag_model = df_tag_model['weight'].to_dict()

        acc = []
        for sentence in sentences:
            tags = viterbi(sentence, dict_gram, dict_tag_model)
            acc.append((sentence, tags))
        for sentence, tags in acc:
            for word, tag in zip(sentence, tags):
                sys.stdout.write('{} {}\n'.format(word, tag))
            sys.stdout.write('\n')
    else:
        raise ValueError('unknown part specified')
