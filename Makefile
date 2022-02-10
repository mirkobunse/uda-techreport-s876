RES_DIR=/mnt/data/results
CP_DIR=/mnt/data/checkpoints
DEPS=$(wildcard util/*.py) /mnt/data/data/.cache/.CACHE

# one set of experiments for each RNG seed 0..9
define SEED_RULE
SEED$(seed)= \
  $(RES_DIR)/buschjaeger_s$(seed).csv \
  $(RES_DIR)/ganin_g10_w02_s$(seed).csv \
  $(RES_DIR)/ganin_g20_w01_s$(seed).csv \
  $(RES_DIR)/li_k1_s$(seed).csv \
  $(RES_DIR)/li_k0_s$(seed).csv
all: $$(SEED$(seed))
seed$(seed): $$(SEED$(seed))
.PHONY: seed$(seed)

$(RES_DIR)/buschjaeger_s$(seed).csv: buschjaeger.py model/buschjaeger.py $(DEPS)
	python $$< $$@ $$(CP_DIR)/buschjaeger_s$(seed) --seed $(seed)
$(RES_DIR)/ganin_g10_w02_s$(seed).csv: ganin.py model/ganin.py $(DEPS)
	python $$< $$@ $$(CP_DIR)/ganin_g10_w02_s$(seed) --gamma 10.0 --dc_weight 0.2 --seed $(seed)
$(RES_DIR)/ganin_g20_w01_s$(seed).csv: ganin.py model/ganin.py $(DEPS)
	python $$< $$@ $$(CP_DIR)/ganin_g20_w01_s$(seed) --gamma 20.0 --dc_weight 0.1 --seed $(seed)

$(RES_DIR)/li_k1_s$(seed).csv: li.py $(DEPS) $(RES_DIR)/buschjaeger_s$(seed).csv
	python $$< $$@ $$(CP_DIR)/li_k1_s$(seed) $$(CP_DIR)/buschjaeger_s$(seed)/bestValLoss.pth --max_k 1 --seed $(seed)
$(RES_DIR)/li_k0_s$(seed).csv: li.py $(DEPS) $(RES_DIR)/buschjaeger_s$(seed).csv
	python $$< $$@ $$(CP_DIR)/li_k0_s$(seed) $$(CP_DIR)/buschjaeger_s$(seed)/bestValLoss.pth --max_k 0 --seed $(seed)
endef
$(foreach seed,0 1 2 3 4 5 6 7 8 9,$(eval $(SEED_RULE)))

test: $(RES_DIR)/test_buschjaeger.csv $(RES_DIR)/test_ganin.csv $(RES_DIR)/test_li.csv
$(RES_DIR)/test_buschjaeger.csv: buschjaeger.py model/buschjaeger.py $(DEPS)
	python $< $@ $(CP_DIR)/test_buschjaeger --n_epochs 2 --is_test_run
$(RES_DIR)/test_ganin.csv: ganin.py model/ganin.py $(DEPS)
	python $< $@ $(CP_DIR)/test_ganin --n_epochs 2 --is_test_run
$(RES_DIR)/test_li.csv: li.py $(DEPS) $(RES_DIR)/test_buschjaeger.csv
	python $< $@ $(CP_DIR)/test_li $(CP_DIR)/test_buschjaeger/bestValLoss.pth --n_epochs 2 --max_k 1 --is_test_run

cache: /mnt/data/data/.cache/.CACHE
/mnt/data/data/.cache/.CACHE: generate_cache.py
	python $< --n_seeds 10 && touch $@

.PHONY: all test cache
