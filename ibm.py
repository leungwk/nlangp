from itertools import izip
from collections import defaultdict
import pandas as pd
import sys
from itertools import product
import numpy as np

from util.func import persist_to_file

NULL = 'NULL'
NEG_INF = -float('inf')

def parallel_sentences(es_path, en_path):
    """create iterator for the parallel texts. es_path as f, en_path as e"""
    with open(es_path, 'r') as es_pile, open(en_path, 'r') as en_pile:
        for sen_es, sen_en in izip(es_pile, en_pile):
            yield sen_es.strip().split(), [NULL] +sen_en.strip().split()


def init_t(es_path, en_path):
    acc_t = set()
    for sen_es, sen_en in parallel_sentences(es_path, en_path):
        ## this should skip over empty translations (see l105)
        for w_es, w_en in product(sen_es, sen_en): # blank en lines aligned all es words to NULL
            acc_t.add( (w_es, w_en) ) # create all possible pairings
    df_t = pd.DataFrame([(es,en) for es,en in acc_t], columns=['w_es','w_en'])
    ## how many en words could possibly map to a es word?
    df_g = df_t.groupby('w_en')['w_es'].apply(lambda ser: len(ser.unique())) # don't use nunique(): http://stackoverflow.com/questions/17926273/how-to-count-distinct-values-in-a-column-of-a-pandas-group-by-object#comment26196484_17926411
    ## init t(f|e) = 1/n(e)
    df_t['val'] = 1./df_t['w_en'].map(df_g) # treat df_g as a dict
    df_t.set_index(['w_es','w_en'],inplace=True)
    return df_t


def init_q(es_path, en_path):
    acc_q = set()
    for sen_es, sen_en in parallel_sentences(es_path, en_path):
        len_m, len_l = len(sen_es), len(sen_en)
        for idx_es, _ in enumerate(sen_es): # idx-1
            idx_es += 1
            for idx_en, _ in enumerate(sen_en): # idx-0
                acc_q.add( (idx_es, idx_en, len_l, len_m, 1./(len_l +1)) ) # i,j,l,m,1/(l+1)
    df_q = pd.DataFrame(list(acc_q), columns=['i','j','l','m','val']).set_index(['i','j','l','m'])
    return df_q
    


def _calc_es(dict_es_en):
    acc = defaultdict(int)
    for (w_es, _), val in dict_es_en.iteritems():
        acc[w_es] += val
    return dict(acc)


@persist_to_file('data/cache/ibm.em.dat')
def ibm1(n_iter, es_path, en_path):
    def _delta(w_es, w_en, dict_es_en, dict_es):
        numer = dict_es_en.get( (w_es, w_en), 0)
        denom = dict_es.get( w_es, 0)
        return 1.*numer/denom if denom > 0 else 0

    df_t = init_t(es_path, en_path)
    ## init
    dict_es_en = df_t['val'].to_dict() # holds current t(f|e)

    for _ in xrange(n_iter):
        dict_es = _calc_es(dict_es_en) # \sum_j t(f|e_j)
        ## reset counts
        c_en_es = {k:0 for k in df_t.index}
        c_en = {k:0 for k in set(df_t.index.get_level_values('w_en'))}
        for sen_es, sen_en in parallel_sentences(es_path, en_path):
            for w_es, w_en in product(sen_es,sen_en):
                d = _delta(w_es, w_en, dict_es_en, dict_es)
                c_en_es[(w_es, w_en)] += d
                c_en[w_en] += d
        ## re-calc t(f|e)
        for w_es, w_en in dict_es_en.iterkeys():
            numer = c_en_es[(w_es, w_en)]
            denom = c_en[w_en]
            dict_es_en[(w_es,w_en)] = 1.*numer/denom if denom > 0 else 0
    return dict_es_en


@persist_to_file('data/cache/ibm2.em.dat')
def ibm2(n_iter, es_path, en_path):
    df_q = init_q(es_path, en_path) # q(j|i, l, m)
    dict_q_ijlm = df_q['val'].to_dict()
    dict_es_en = ibm1(5, es_path, en_path) # t(f|e)

    def _delta(i, j, f_i, e_j, sen_es, sen_en, len_m, len_l, dict_q_ijlm, dict_es_en):
        """.../\sum_{j=0}^{l_k} q(j|i,l_k,m_k)t(f_i^{(k)}|e_j^{(k)})"""
        acc = 0
        for idx_j, w_en in enumerate(sen_en):
            t_term = dict_es_en.get( (f_i,w_en), 0)
            q_term = dict_q_ijlm.get( (i,idx_j,len_l,len_m), 0)
            acc += (t_term * q_term)
        t_term = dict_es_en.get( (f_i,e_j), 0)
        q_term = dict_q_ijlm.get( (i,j,len_l,len_m), 0)
        return 1.*q_term*t_term/acc if acc > 0 else 0

    for _ in xrange(n_iter):
        c_en_es = {k:0 for k in dict_es_en.keys()}
        c_en = {k:0 for _,k in dict_es_en.keys()}
        c_ijlm = {k:0 for k in df_q.index}
        c_ilm = {(i,l,m):0 for i,_,l,m in df_q.index}
        for sen_es, sen_en in parallel_sentences(es_path, en_path):
            len_m, len_l = len(sen_es), len(sen_en)
            for idx_es, w_es in enumerate(sen_es): # idx-1
                idx_es += 1
                for idx_en, w_en in enumerate(sen_en): # idx-0
                    d = _delta(idx_es, idx_en, w_es, w_en, sen_es, sen_en, len_m, len_l, dict_q_ijlm, dict_es_en)
                    c_en_es[(w_es, w_en)] += d
                    c_en[w_en] += d
                    c_ijlm[(idx_es,idx_en,len_l,len_m)] += d
                    c_ilm[(idx_es,len_l,len_m)] += d
        ## re-calc t(f|e)
        for w_es, w_en in dict_es_en.iterkeys():
            numer = c_en_es[(w_es, w_en)]
            denom = c_en[w_en]
            dict_es_en[(w_es,w_en)] = 1.*numer/denom if denom > 0 else 0
        ## re-calc q(j|i,l,m)
        for i,j,l,m in dict_q_ijlm.iterkeys():
            numer = c_ijlm[(i,j,l,m)]
            denom = c_ilm[(i,l,m)]
            dict_q_ijlm[(i,j,l,m)] = 1.*numer/denom if denom > 0 else 0
    return dict_es_en, dict_q_ijlm





## for the alignment matrix
def neighbours(i,j,mtx):
    ## counterclock-wise
    for di, dj in [[i-1, j-1],[i, j-1],[i+1, j-1],[i+1, j],[i+1, j+1],[i, j+1],[i-1, j+1],[i-1, j]]:
        if di < 0 or dj < 0:
            continue
        if di >= mtx.shape[0] or dj >= mtx.shape[1]:
            continue
        yield di, dj


def cross_radiate(i,j,mtx):
    yield i,j
    for k in xrange(1,max(mtx.shape)):
        for di, dj in [[i, j-k],[i+k, j],[i, j+k],[i-k, j]]:
            if di < 0 or dj < 0:
                continue
            if di >= mtx.shape[0] or dj >= mtx.shape[1]:
                continue
            yield di, dj


def all_aligned(i,j,mtx):
    ## any row or column aligned in both directions
    return any(mtx[:,j] >= 2) and any(mtx[i,:] >= 2)

def any_aligned(i,j,mtx):
    return any(mtx[:,j] >= 2) or any(mtx[i,:] >= 2)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('part', help='part to run') # nargs=1, produces a list, while nothing produes the value itself
    parser.add_argument('--en-data', help='list of questions (default: data/dev.en)', default='data/dev.en')
    parser.add_argument('--es-data', help='list of questions (default: data/dev.es)', default='data/dev.es')
    args = parser.parse_args()

    if args.part not in ('p1','p2','p3'):
        raise ValueError('unknown part specified')

    ## init
    en_path = 'data/corpus.en'
    es_path = 'data/corpus.es'
    en_dev_path = args.en_data
    es_dev_path = args.es_data

    if args.part == 'p1':
        dict_es_en = ibm1(5, es_path, en_path) # values in [0,1]

        ## find a_i = \arg\max_{j \in \set{0...l}} t(f_i|e_j)
        for sen_idx, (sen_es, sen_en) in enumerate(parallel_sentences(es_dev_path, en_dev_path)):
            for w_es_idx, w_es in enumerate(sen_es):
                w_es_idx += 1 # idx-1
                max_idx, max_val = None, None
                for w_en_idx, w_en in enumerate(sen_en):
                    val = dict_es_en.get( (w_es, w_en), None)
                    if val is not None and (val > max_val):
                        max_idx, max_val = w_en_idx, val
                if max_val is not None and (max_val >= 0): # if no alignments, don't output anything
                    sys.stdout.write('{} {} {}\n'.format(sen_idx +1, max_idx, w_es_idx)) # idx-1
    elif args.part == 'p2':
        dict_es_en, dict_q_ijlm = ibm2(5, es_path, en_path)
        ## find a_i = \arg\max_{j \in \set{0...l}} q(j|i,l,m)t(f_i|e_j)
        for sen_idx, (sen_es, sen_en) in enumerate(parallel_sentences(es_dev_path, en_dev_path)):
            len_m, len_l = len(sen_es), len(sen_en)
            for w_es_idx, w_es in enumerate(sen_es): # i
                w_es_idx += 1 # idx-1
                max_idx, max_val = None, None
                for w_en_idx, w_en in enumerate(sen_en): # j
                    val = dict_es_en.get( (w_es, w_en), None)
                    val2 = dict_q_ijlm.get( (w_es_idx, w_en_idx, len_l, len_m), None )
                    prod = val*val2 if ((val is not None) and (val2 is not None)) else None
                    if prod is not None and (prod > max_val):
                        max_idx, max_val = w_en_idx, prod
                if (max_val is not None) and max_val >= 0: # if no alignments, don't output anything
                    sys.stdout.write('{} {} {}\n'.format(sen_idx +1, max_idx, w_es_idx)) # idx-1
    elif args.part == 'p3':
        dict_es_en, dict_q_ijlm_es_en = ibm2(5, es_path, en_path) # t(f|e), q(j|i,l,m) (as i,j,l,m, where i=es, j=en)
        dict_en_es, dict_q_ijlm_en_es = ibm2(5, en_path, es_path) # t(e|f), q(i|j,l,m)

        def alignments(es_path, en_path, dict_es_en, dict_q_ijlm_es_en):
            vec_as = []
            for sen_idx, (sen_es, sen_en) in enumerate(parallel_sentences(es_path, en_path)):
                ## for a given (f,e) as example k
                # a_i = \arg\max_{a \in \set{0...l}} q(a|i,l,m)t(f_i|e_a)
                len_m, len_l = len(sen_es), len(sen_en)
                vec_a = []
                for w_es_idx, w_es in enumerate(sen_es): # i
                    w_es_idx += 1
                    ## f_i fixed, a_j to vary
                    max_val, max_a = 0, None
                    for w_en_idx, w_en in enumerate(sen_en): # j
                        q_term = dict_q_ijlm_es_en.get( (w_es_idx, w_en_idx, len_l, len_m), 0)
                        t_term = dict_es_en.get( (w_es, w_en), 0 )
                        prod = q_term*t_term
                        if prod > max_val:
                            max_val, max_a = prod, w_en_idx
                    ## max_a is a_i
                    vec_a.append(max_a)
                ## vec_a now has best alignment for this (f,e)
                vec_as.append(vec_a)
            return vec_as

        as_es_en = alignments(es_dev_path, en_dev_path, dict_es_en, dict_q_ijlm_es_en)
        as_en_es = alignments(en_dev_path, es_dev_path, dict_en_es, dict_q_ijlm_en_es)

        for sen_idx, (a_es_en, a_en_es) in enumerate(zip(as_es_en, as_en_es)):
            ai_mtx = np.zeros( (len(a_en_es),len(a_es_en)) ) # en x es
            for i, a_i in enumerate(a_es_en):
                ai_mtx[a_i -1,i] = 1 # idx-0 for src to dest, idx-1 for idx of src
            aj_mtx = np.zeros( (len(a_es_en),len(a_en_es)) )
            for j, a_j in enumerate(a_en_es):
                aj_mtx[a_j -1,j] = 1
            a_tmp_mtx = ai_mtx +aj_mtx.T
            ## intersection
            a_tmp_mtx >= 2
            ## union
            a_tmp_mtx >= 1

            ### grow alignments

            ## consider other unaligned points
            for (i,j), val in np.ndenumerate(a_tmp_mtx):
                if a_tmp_mtx[i,j] == 1 and not any_aligned(i,j,a_tmp_mtx):
                    a_tmp_mtx[i,j] = 4

            ## now write out the alignments
            for (i,j), val in np.ndenumerate(a_tmp_mtx):
                if val < 2:
                    continue
                sys.stdout.write('{} {} {}\n'.format(sen_idx +1, i+1, j+1)) # idx-1
