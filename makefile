gene_key_path=data/gene.key
gene_test_path=data/gene.test

rm=rm -f



.PHONY: all
all: all_a1
all_a1: a1_p1_all a1_p2_all a1_p3_all
# all_a2
all_a2: data/parse_train.counts.out a2_p2_eval data/parse_test.p2.out a2_p3_eval data/parse_test.p3.out

clean: clean_data_a1 clean_data_a2
	$(rm) *.pyc

# TODO: include cache clean

#### a1

data/gene.counts:
	python count_freqs.py data/gene.train > data/gene.counts



data/gene_dev.p1.out: data/gene.counts
	python hmm.py p1 > data/gene_dev.p1.out

data/gene_test.p1.out: data/gene.counts
	python hmm.py p1 --gene-data gene.test > data/gene_test.p1.out

a1_p1_eval: data/gene_dev.p1.out
	python eval_gene_tagger.py $(gene_key_path) data/gene_dev.p1.out

a1_p1_all: data/gene.counts
	python hmm.py p1 > data/gene_dev.p1.out
	python eval_gene_tagger.py $(gene_key_path) data/gene_dev.p1.out

	python hmm.py p1 --gene-data gene.test > data/gene_test.p1.out

	python hmm.py p1 --emission df_em_wo_rare > data/gene_dev.df_em_wo_rare.p1.out
	python eval_gene_tagger.py $(gene_key_path) data/gene_dev.df_em_wo_rare.p1.out

	python hmm.py p1 --emission df_em_kept_rare > data/gene_dev.df_em_kept_rare.p1.out
	python eval_gene_tagger.py $(gene_key_path) data/gene_dev.df_em_kept_rare.p1.out



data/gene_dev.p2.out: data/gene.counts
	python hmm.py p2 > data/gene_dev.p2.out

data/gene_test.p2.out: data/gene.counts
	python hmm.py p2 --gene-data gene.test > data/gene_test.p2.out

a1_p2_eval: data/gene_dev.p2.out
	python eval_gene_tagger.py $(gene_key_path) data/gene_dev.p2.out

a1_p2_all: data/gene.counts
	python hmm.py p2 > data/gene_dev.p2.out
	python eval_gene_tagger.py $(gene_key_path) data/gene_dev.p2.out

	python hmm.py p2 --gene-data gene.test > data/gene_test.p2.out

	python hmm.py p2 --emission df_em_wo_rare > data/gene_dev.df_em_wo_rare.p2.out
	python eval_gene_tagger.py $(gene_key_path) data/gene_dev.df_em_wo_rare.p2.out

	python hmm.py p2 --emission df_em_kept_rare > data/gene_dev.df_em_kept_rare.p2.out
	python eval_gene_tagger.py $(gene_key_path) data/gene_dev.df_em_kept_rare.p2.out



data/gene_dev.p3.out: data/gene.counts
	python hmm.py p3 > data/gene_dev.p3.out

data/gene_test.p3.out: data/gene.counts
	python hmm.py p3 --gene-data gene.test > data/gene_test.p3.out

a1_p3_eval: data/gene_dev.p3.out
	python eval_gene_tagger.py $(gene_key_path) data/gene_dev.p3.out

a1_p3_all: data/gene.counts
	python hmm.py p3 > data/gene_dev.p3.out
	python eval_gene_tagger.py $(gene_key_path) data/gene_dev.p3.out

	python hmm.py p3 --gene-data gene.test > data/gene_test.p3.out



clean_data_a1:
	$(rm) data/gene.counts

	$(rm) data/gene_dev.p1.out
	$(rm) data/gene_test.p1.out
	$(rm) data/gene_dev.df_em_wo_rare.p1.out
	$(rm) data/gene_dev.df_em_kept_rare.p1.out

	$(rm) data/gene_dev.p2.out
	$(rm) data/gene_test.p2.out
	$(rm) data/gene_dev.df_em_wo_rare.p2.out
	$(rm) data/gene_dev.df_em_kept_rare.p2.out

	$(rm) data/gene_dev.p3.out
	$(rm) data/gene_test.p3.out

#### a2

data/cfg.counts:
	python count_cfg_freq.py data/parse_train.dat > $@

data/cfg.vert.counts:
	python count_cfg_freq.py data/parse_train_vert.dat > $@

data/parse_train.counts.out: data/cfg.counts
	python pcfg.py p1 > $@

data/parse_dev.out: data/cfg.counts
	python pcfg.py p2 > $@.tmp
	mv $@.tmp $@

data/parse_test.p2.out: data/cfg.counts
	python pcfg.py p2 --q-data data/parse_test.dat > $@.tmp
	mv $@.tmp $@

a2_p2_eval: data/parse_dev.out
	python eval_parser.py data/parse_dev.key data/parse_dev.out

data/parse_dev_vert.out: data/cfg.vert.counts
	python pcfg.py p3 > $@.tmp
	mv $@.tmp $@

a2_p3_eval: data/parse_dev_vert.out
	python eval_parser.py data/parse_dev.key data/parse_dev_vert.out

data/parse_test.p3.out: data/cfg.vert.counts
	python pcfg.py p3 --q-data data/parse_test.dat > $@.tmp
	mv $@.tmp $@




clean_data_a2:
	$(rm) data/cfg.counts
	$(rm) data/parse_train.counts.out
	$(rm) data/parse_dev.out
	$(rm) data/parse_test.p2.out



#### a3

data/corpus_dev.p1.out:
	python ibm.py p1 --en-data data/dev.en --es-data data/dev.es > $@.tmp
	mv $@.tmp $@

a3_p1_eval: data/corpus_dev.p1.out
	python eval_alignment.py data/dev.key data/corpus_dev.p1.out

data/corpus_test.p1.out:
	python ibm.py p1 --en-data data/test.en --es-data data/test.es > $@.tmp
	mv $@.tmp $@

data/corpus_dev.p2.out:
	python ibm.py p2 --en-data data/dev.en --es-data data/dev.es > $@.tmp
	mv $@.tmp $@

a3_p2_eval: data/corpus_dev.p2.out
	python eval_alignment.py data/dev.key data/corpus_dev.p2.out

data/corpus_test.p2.out:
	python ibm.py p2 --en-data data/test.en --es-data data/test.es > $@.tmp
	mv $@.tmp $@
