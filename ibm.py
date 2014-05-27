from itertools import izip
from collections import defaultdict
import pandas as pd
import sys
from itertools import product

from util.func import persist_to_file

NULL = 'NULL'

def parallel_sentences(es_path, en_path):
    """create iterator for the parallel texts"""
    with open(es_path, 'r') as es_pile, open(en_path, 'r') as en_pile:
        for sen_es, sen_en in izip(es_pile, en_pile):
            yield sen_es, sen_en


def init_t(es_path, en_path):
    acc_t = set()
    for sen_es, sen_en in parallel_sentences(es_path, en_path):
        ## this should skip over empty translations (see l105)
        for w_es, w_en in product(sen_es.strip().split(),
                                  sen_en.strip().split() +[NULL]): # blank en lines aligned all es words to NULL
            acc_t.add( (w_es, w_en) ) # create all possible pairings
    df_t = pd.DataFrame([(es,en) for es,en in acc_t], columns=['w_es','w_en'])
    ## how many en words could possibly map to a es word?
    df_g = df_t.groupby('w_en')['w_es'].apply(lambda ser: len(ser.unique())) # don't use nunique(): http://stackoverflow.com/questions/17926273/how-to-count-distinct-values-in-a-column-of-a-pandas-group-by-object#comment26196484_17926411
    ## init t(f|e) = 1/n(e)
    df_t['val'] = 1./df_t['w_en'].map(df_g) # treat df_g as a dict
    df_t.set_index(['w_es','w_en'],inplace=True)
    return df_t


def delta(w_es, w_en, dict_en_es, dict_es):
    numer = dict_en_es.get( (w_es, w_en), 0)
    denom = dict_es.get( w_es, 0)
    return 1.*numer/denom if denom > 0 else 0


def _calc_es(dict_en_es):
    acc = defaultdict(int)
    for (w_es, _), val in dict_en_es.iteritems():
        acc[w_es] += val
    return dict(acc)


@persist_to_file('data/cache/ibm.em.dat')
def em(n_iter, es_path, en_path):
    df_t = init_t(es_path, en_path)
    ## init
    dict_en_es = df_t['val'].to_dict() # holds current t(f|e)

    for _ in xrange(n_iter):
        dict_es = _calc_es(dict_en_es) # \sum_j t(f|e_j)
        ## reset counts
        c_en_es = {k:0 for k in df_t.index}
        c_en = {k:0 for k in set(df_t.index.get_level_values('w_en'))}
        for sen_es, sen_en in parallel_sentences(es_path, en_path):
            for w_es, w_en in product(sen_es.strip().split(),
                                      sen_en.strip().split()):
                d = delta(w_es, w_en, dict_en_es, dict_es)
                c_en_es[(w_es, w_en)] += d
                c_en[w_en] += d
                # TODO: expand to incl ibm model 2
        ## re-calc t(f|e)
        for w_es, w_en in dict_en_es.iterkeys():
            numer = c_en_es[(w_es, w_en)]
            denom = c_en[w_en]
            dict_en_es[(w_es,w_en)] = 1.*numer/denom if denom > 0 else 0
    return dict_en_es



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
        dict_en_es = em(5, es_path, en_path) # values in [0,1]

        ## find a_i = \arg\max_{j \in \set{0...l}} t(f_i|e_j)
        for sen_idx, (sen_es, sen_en) in enumerate(parallel_sentences(es_dev_path, en_dev_path)):
            for w_es_idx, w_es in enumerate(sen_es.strip().split()):
                max_idx, max_val = None, -1
                for w_en_idx, w_en in enumerate(sen_en.strip().split() +[NULL]):
                    val = dict_en_es.get( (w_es, w_en), -1)
                    if val > max_val:
                        max_idx, max_val = w_en_idx, val
                if val >= 0: # if no alignments, don't output anything
                    sys.stdout.write('{} {} {}\n'.format(sen_idx +1, max_idx +1, w_es_idx +1)) # idx-1
