import pandas as pd
from collections import defaultdict

import sys
import numpy as np

import json

RARE='_RARE_'

def load_tree_counts(fpath):
    acc_nonterm = []
    acc_bin_rule = []
    acc_u_rule = []
    with open(fpath, 'r') as pile:
        for line in pile:
            cnt, tipe, rest = line.strip().split(' ', 2)
            if tipe == 'NONTERMINAL':
                acc_nonterm.append( (int(cnt), rest.split(' ')[0]) )
            elif tipe == 'BINARYRULE':
                x, y1, y2 = rest.split(' ')
                acc_bin_rule.append( (int(cnt), x, y1, y2) )
            elif tipe == 'UNARYRULE':
                x, y = rest.split(' ')
                acc_u_rule.append( (int(cnt), x, y) )
            else:
                raise ValueError("unrecognized type {}".format(tipe))
    df_nterm = pd.DataFrame(acc_nonterm, columns=['cnt','nterm']).set_index(['nterm'])
    df_brule = pd.DataFrame(acc_bin_rule, columns=['cnt','x','y1','y2']).set_index(['x','y1','y2'])
    df_urule = pd.DataFrame(acc_u_rule, columns=['cnt','x','w']).set_index(['x','w'])
    return df_nterm, df_brule, df_urule


def replace_rare(df_urule):
    acc = defaultdict(int)
    for (x, y), cnt in df_urule.itertuples():
        if cnt < 5:
            y = RARE
        acc[(x,y)] += cnt
    df_urule_rare = pd.DataFrame([(x,y,c) for (x,y),c in acc.iteritems()], columns=['x','w','cnt']).set_index(['x','w'])
    return df_urule_rare


def rule_params(rule, dict_rule, dict_nterm, tipe):
    """calc q(X \to Y1 Y2) or q(X \to w) depending on tipe"""
    if tipe == 'brule':
        x, y1, y2 = rule
    elif tipe == 'urule':
        x, w = rule
    else:
        raise ValueError('unrecognized rule type {}'.format(tipe))

    cnt_x = dict_nterm.get(x, None)
    if cnt_x is None:
        raise ValueError('unrecognized nonterminal {}'.format(x))

    if tipe == 'brule':
        try:
            cnt_x_to_y1_y2 = dict_rule[rule]
            # cnt_x_to_y1_y2 = dict_rule.loc[x, y1, y2]['cnt']
        except KeyError: # this should not be happening
            raise ValueError('unrecognized rule {}'.format(rule))
        return 1.*cnt_x_to_y1_y2/cnt_x
    elif tipe == 'urule':
        try:
            cnt_x_to_w = dict_rule.loc[x, w]['cnt']
        except KeyError:
            ## is the word emitted by some other nonterminal?
            tmp_df = dict_rule[dict_rule.index.map(lambda r: (r[0] != x) and (r[1] == w))]
            if len(tmp_df) >= 1: # yes, it is under some other nonterminal
                return 0
            w = RARE
            try:
                cnt_x_to_w = dict_rule.loc[x, w]['cnt']
            except KeyError:
                cnt_x_to_w = 0
        return 1.*cnt_x_to_w/cnt_x
        

def cky(sentence, dict_nterm, dict_brule, dict_brule_rhs, df_urule, debug=False):
    """return the most likely tree for all possible trees of this sentence"""
    ## init
    dtab, pbtab, valtab = {}, {}, {}
    nterms = df_nterm.index
    n = len(sentence)

    neg_inf = -float('inf')

    if debug:
        import pdb
        pdb.set_trace()

    ## urule
    for i, word in enumerate(sentence): # {1...n}
        i += 1 # idx-1
        for nterm in nterms:
            # try: # check if X \to x_i \in R
            #     q_x_to_w = df_urule.loc[nterm, word]
            # except KeyError:
            #     q_x_to_w = 0
            # else:
            #     q_x_to_w = rule_params( (nterm, word), df_urule, dict_nterm, tipe='urule')
            q_x_to_w = rule_params( (nterm, word), df_urule, dict_nterm, tipe='urule')
            dtab[(i,i,nterm)] = np.log10(q_x_to_w)
            pbtab[(i,i,nterm)] = (nterm, word, None, i, None, i)


    ## brule
    for l in xrange(1,n-1 +1): # incl. right limit; idx-1; the size of the substring to look at, starting with windows equal to 1, then 2, and so on
        for i in xrange(1,n-l +1): # left bound
            j = i+l # left bound plus window size equals right bound
            for nterm in nterms:
                ## \max_{X \to Y Z \in R, s \in \set{i,\dots,j-1}}
                ## X is fixed from above
                max_val, max_arg = neg_inf, (nterm, None, None, i, None, j)
                # try:
                #     df_brule_nterm = df_brule.xs(nterm,level='x')
                # except KeyError:
                #     pass # because X \to Y Z \not\in R
                # for (y1, y2), _ in df_brule_nterm.itertuples():

                if nterm not in dict_brule_rhs:
                    pass
                else:
                    for y1, y2 in dict_brule_rhs[nterm]:
                        for s in xrange(i,j-1 +1): # split point within the left and right bound inclusive
                            dtab_l = dtab.get( (i,s,y1), neg_inf )
                            dtab_r = dtab.get( (s+1,j,y2), neg_inf )

                            ## keep as many values as possible, but still filter early
                            if (dtab_l == neg_inf) or (dtab_r == neg_inf):
                                q_x_to_y1_y2 = None
                                val = None
                            else:
                                q_x_to_y1_y2 = np.log10(rule_params( (nterm, y1, y2), dict_brule, dict_nterm, tipe='brule'))
                                val = q_x_to_y1_y2 +dtab_l +dtab_r
                            valtab[(i,j,nterm,y1,y2,s)] = (val, q_x_to_y1_y2, dtab_l, dtab_r)
                            if val > max_val:
                                max_val, max_arg = val, (nterm, y1, y2, i, s, j)

                if max_val > neg_inf: # only store values that have a probability of occuring
                    idx = (i,j,nterm)
                    dtab[idx], pbtab[idx] = max_val, max_arg
    df_dtab = pd.DataFrame([(i,j,x,val) for (i,j,x), val in dtab.iteritems()], columns=['i','j','x','val']).sort(['i','j','x']).set_index(['i','j','x'])
    df_pbtab = pd.DataFrame([(i,j,x,nterm,y1,y2,s) for (i,j,x), (nterm, y1, y2, _, s, _) in pbtab.iteritems()], columns=['i','j','x','nterm','y1','y2','s']).sort(['i','j','x']).set_index(['i','j','x'])
    df_valtab = pd.DataFrame([(i,j,x,y1,y2,s,val,q,l,r) for (i,j,x,y1,y2,s), (val,q,l,r) in valtab.iteritems()], columns=['i','j','x','y1','y2','s','val','q','l','r']).sort(['i','j','x','s']).set_index(['i','j','x','y1','y2','s'])
    return df_dtab, df_pbtab, df_valtab


def backtrace(rule, df_pbtab):
    """unroll the backpointer table"""
    nterm, y1, y2, s = rule
    i, j, _ = rule.name
    if y2 is not None: # brule
        return [nterm,
                backtrace(df_pbtab.loc[i, s, y1], df_pbtab),
                backtrace(df_pbtab.loc[s+1, j, y2], df_pbtab) ]
    else: # urule (leaf)
        return [nterm, y1]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('part', help='part to run') #  nargs=1, produces a list, while nothing produes the value itself
    parser.add_argument('--q-data', help='list of questions (default: data/parse_dev.dat)', default='data/parse_dev.dat')
    args = parser.parse_args()

    if args.part not in ('p1','p2','p3'):
        raise ValueError('unknown part specified')

    ## init
    tree_counts_path = 'data/cfg.counts'
    df_nterm, df_brule, df_urule = load_tree_counts(tree_counts_path)
    df_urule_rare = replace_rare(df_urule)

    dict_nterm = df_nterm['cnt'].to_dict()
    dict_brule = df_brule['cnt'].to_dict()

    ## unroll rhs rules in df_brule
    df_brule_xs = set(df_brule.index.get_level_values('x'))
    dict_brule_rhs = {}
    for x in df_brule_xs:
        tmp_df = df_brule.xs(x,level='x')
        s = set()
        for (y1, y2), _ in tmp_df.itertuples():
            s.add( (y1, y2) )
        dict_brule_rhs[x] = s

    if args.part == 'p1':
        for nterm, cnt in df_nterm.itertuples():
            sys.stdout.write('{} NONTERMINAL {}\n'.format(cnt, nterm))
        for (x, w), cnt in df_urule_rare.itertuples():
            sys.stdout.write('{} UNARYRULE {} {}\n'.format(cnt, x, w))
        for (x, y1, y2), cnt in df_brule.itertuples():
            sys.stdout.write('{} BINARYRULE {} {} {}\n'.format(cnt, x, y1, y2))
    elif args.part == 'p2':
        with open(args.q_data, 'r') as pile:
            for line in pile:
                sentence = line.strip().split(' ') # strip required otherwise sentence length will be -1 on command line, but +0 in ipython
                df_dtab, df_pbtab, df_valtab = cky(sentence, dict_nterm, dict_brule, dict_brule_rhs, df_urule_rare)
                ## construct tree
                # tmp_df = df_dtab[
                #     (df_dtab.index.get_level_values('i') == 1) &
                #     (df_dtab.index.get_level_values('j') == len(sentence))]
                # ## setting SBARQ explicitly seems odd
                # try:
                #     key = tmp_df['val'].argmax()
                # except ValueError:
                #     key = (1, len(sentence), 'SBARQ')
                key = (1, len(sentence), 'SBARQ')
                ## needed otherwise "KeyError: 'the label [11] is not in the [columns]'"
                try:
                    tree = backtrace(df_pbtab.loc[key], df_pbtab)
                except KeyError as ke:
                    ## construct a random tree as a placeholder for now (so that the eval script works)
                    def _f(acc, sentence):
                        return 

                    tree = ['SBARQ']
                    for _ in sentence:
                        tree = ['SBARQ', '']
                    # raise Exception(ke.message +'\n {}'.format(sentence))
                sys.stdout.write(json.dumps(tree) +'\n')
