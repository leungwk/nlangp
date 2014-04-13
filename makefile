gene_key_path=data/gene.key
gene_test_path=data/gene.test

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



.PHONY: clean
clean:
	rm *.pyc
