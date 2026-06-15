# Training: Vacancy в†’ CV Two-Stage Recommendation Model

Р­С‚Р° РёРЅСЃС‚СЂСѓРєС†РёСЏ РѕРїРёСЃС‹РІР°РµС‚, РєР°Рє РІРѕСЃРїСЂРѕРёР·РІРµСЃС‚Рё РѕР±СѓС‡РµРЅРёРµ С„РёРЅР°Р»СЊРЅРѕР№ РґРІСѓС…СЃС‚Р°РґРёР№РЅРѕР№ РјРѕРґРµР»Рё.

РЎРёСЃС‚РµРјР° СЂРµС€Р°РµС‚ Р·Р°РґР°С‡Сѓ:

```text
РЅР° РІС…РѕРґ: РІР°РєР°РЅСЃРёСЏ
РЅР° РІС‹С…РѕРґ: top-K РїРѕРґС…РѕРґСЏС‰РёС… СЂРµР·СЋРјРµ
```

Pipeline СЃРѕСЃС‚РѕРёС‚ РёР· РґРІСѓС… СЃС‚Р°РґРёР№:

```text
1. Retrieval / candidate generation:
   vacancy embedding -> top-500 CV РїРѕ cosine similarity

2. Ranking:
   CatBoostRanker РїРµСЂРµСѓРїРѕСЂСЏРґРѕС‡РёРІР°РµС‚ top-500 РєР°РЅРґРёРґР°С‚РѕРІ
   Рё С„РѕСЂРјРёСЂСѓРµС‚ С„РёРЅР°Р»СЊРЅС‹Р№ top-10
```

## 0. РџРѕРґРіРѕС‚РѕРІРєР° РґР°РЅРЅС‹С…

РЎС‹СЂС‹Рµ РґР°РЅРЅС‹Рµ РЅРµ РІС…РѕРґСЏС‚ РІ СЂРµРїРѕР·РёС‚РѕСЂРёР№. РћРЅРё РґРѕР»Р¶РЅС‹ Р»РµР¶Р°С‚СЊ РІ:

```text
data/aaa-out/
```

РћР¶РёРґР°РµРјС‹Рµ С„Р°Р№Р»С‹:

```text
cv.parquet
cv_embeddings.parquet
vacancies.parquet
vacancies_embeddings.parquet
applies.parquet
```

РџРµСЂРµРґ РѕР±СѓС‡РµРЅРёРµРј РЅСѓР¶РЅРѕ РІС‹РїРѕР»РЅРёС‚СЊ РЅРѕСЂРјР°Р»РёР·Р°С†РёСЋ:

```bash
python scripts/normalize_data.py --input-dir data/aaa-out --output-root data/processed --version v1
```

РџРѕСЃР»Рµ СЌС‚РѕРіРѕ РґРѕР»Р¶РЅР° РїРѕСЏРІРёС‚СЊСЃСЏ РїР°РїРєР°:

```text
data/processed/v1/
```

Р’ РЅРµР№ РґРѕР»Р¶РЅС‹ Р±С‹С‚СЊ РЅРѕСЂРјР°Р»РёР·РѕРІР°РЅРЅС‹Рµ С‚Р°Р±Р»РёС†С‹:

```text
cv_normalized.parquet
vacancies_normalized.parquet
manifest.json
normalization_report.json
```

Р’Р°Р¶РЅРѕ: РёСЃС…РѕРґРЅС‹Рµ embeddings РІСЃС‘ РµС‰С‘ РґРѕР»Р¶РЅС‹ РѕСЃС‚Р°РІР°С‚СЊСЃСЏ РІ `data/aaa-out/`, РїРѕС‚РѕРјСѓ С‡С‚Рѕ РїРµСЂРІР°СЏ СЃС‚Р°РґРёСЏ retrieval РёСЃРїРѕР»СЊР·СѓРµС‚ `cv_embeddings.parquet` Рё `vacancies_embeddings.parquet`.

---

## 1. Stage 1: build retrieval candidate pool

РџРµСЂРІР°СЏ СЃС‚Р°РґРёСЏ СЃС‚СЂРѕРёС‚ candidate pool. Р”Р»СЏ РєР°Р¶РґРѕР№ РІР°РєР°РЅСЃРёРё СЃС‡РёС‚Р°РµС‚СЃСЏ similarity РјРµР¶РґСѓ embedding РІР°РєР°РЅСЃРёРё Рё embeddings РІСЃРµС… CV.

РўР°Рє РєР°Рє embeddings L2-РЅРѕСЂРјРёСЂРѕРІР°РЅС‹, РёСЃРїРѕР»СЊР·СѓРµС‚СЃСЏ dot product, РєРѕС‚РѕСЂС‹Р№ СЌРєРІРёРІР°Р»РµРЅС‚РµРЅ cosine similarity.

Р”Р»СЏ РєР°Р¶РґРѕР№ РІР°РєР°РЅСЃРёРё РІС‹Р±РёСЂР°РµС‚СЃСЏ top-500 CV.

РљРѕРјР°РЅРґР°:

```bash
python scripts/build_ltr_dataset.py \
  --processed-version v1 \
  --top-k 500 \
  --negative-sampling-strategy hard \
  --negative-ratio 5 \
  --keep-debug-columns \
  --dataset-name ltr_v1_top500_neg5_ohe
```

РџРѕСЃР»Рµ РІС‹РїРѕР»РЅРµРЅРёСЏ РїРѕСЏРІРёС‚СЃСЏ РїР°РїРєР°:

```text
data/modeling/ltr_v1_top500_neg5_ohe/
```

РћСЃРЅРѕРІРЅС‹Рµ С„Р°Р№Р»С‹ РІРЅСѓС‚СЂРё:

```text
train_features.parquet
valid_features.parquet
train_positive_pairs.parquet
valid_positive_pairs.parquet
feature_columns.json
dataset_summary.json
candidate_generation_metrics.csv
```

Р§С‚Рѕ Р·РґРµСЃСЊ РїСЂРѕРёСЃС…РѕРґРёС‚:

```text
positive pairs = historical applies
negative pairs = hard negatives РёР· embedding top-500
negative ratio = 5 negatives РЅР° РєР°Р¶РґС‹Р№ positive
```

Р­С‚Р° РєРѕРјР°РЅРґР° РѕРґРЅРѕРІСЂРµРјРµРЅРЅРѕ РґРµР»Р°РµС‚:

```text
1. temporal train/validation split
2. embedding retrieval top-500
3. РїРѕСЃС‚СЂРѕРµРЅРёРµ pairwise features
4. negative sampling РґР»СЏ train
5. СЃРѕС…СЂР°РЅРµРЅРёРµ train/valid РґР°С‚Р°СЃРµС‚РѕРІ РґР»СЏ CatBoost
```

---

## 2. Stage 2: train CatBoostRanker

Р¤РёРЅР°Р»СЊРЅР°СЏ РјРѕРґРµР»СЊ РѕР±СѓС‡Р°РµС‚СЃСЏ РїРѕРІРµСЂС… candidate pool РёР· Stage 1.

Р›СѓС‡С€Р°СЏ РЅР°Р№РґРµРЅРЅР°СЏ РєРѕРЅС„РёРіСѓСЂР°С†РёСЏ:

```text
model = CatBoostRanker
loss_function = YetiRank
retrieval top-k = 500
negative sampling = hard-only, 5 negatives per positive
iterations = 500
depth = 6
learning_rate = 0.05
l2_leaf_reg = 3
one_hot_max_size = 10
label_weighting = none
selection metric = NDCG@10
```

РљРѕРјР°РЅРґР° РѕР±СѓС‡РµРЅРёСЏ:

```bash
python scripts/train_catboost_ranker.py \
  --dataset-dir data/modeling/ltr_v1_top500_neg5_ohe \
  --run-name catboost_final_yetirank_neg5 \
  --loss-function YetiRank \
  --iterations 500 \
  --depth 6 \
  --learning-rate 0.05 \
  --l2-leaf-reg 3 \
  --label-weighting none \
  --one-hot-max-size 10 \
  --early-stopping-rounds 80 \
  --ks 5 10 20 50 100 500 \
  --select-metric ndcg_at_k \
  --select-k 10 \
  --save-recommendations \
  --k-inf 10
```

РџРѕСЃР»Рµ РѕР±СѓС‡РµРЅРёСЏ РїРѕСЏРІСЏС‚СЃСЏ:

```text
data/models/catboost_final_yetirank_neg5/
data/experiments/catboost_final_yetirank_neg5/
```

Р’ `data/models/...` СЃРѕС…СЂР°РЅСЏРµС‚СЃСЏ `.cbm` РјРѕРґРµР»СЊ.

Р’ `data/experiments/...` СЃРѕС…СЂР°РЅСЏСЋС‚СЃСЏ:

```text
metrics.csv
model_results.csv
best_model.json
run_summary.json
recommendations_top10_*.parquet
recommendations_top10_*.csv
```

---

## 3. РЎРѕС…СЂР°РЅРµРЅРёРµ С„РёРЅР°Р»СЊРЅРѕР№ РјРѕРґРµР»Рё СЃРѕ РІС‚РѕСЂРѕР№ СЃС‚Р°РґРёРё

РџР°РїРєР° `models/` СЃРѕРґРµСЂР¶РёС‚ С‚РѕР»СЊРєРѕ РѕРґРЅСѓ С„РёРЅР°Р»СЊРЅСѓСЋ РјРѕРґРµР»СЊ РґР»СЏ РІРѕСЃРїСЂРѕРёР·РІРµРґРµРЅРёСЏ inference Р±РµР· РїРµСЂРµРѕР±СѓС‡РµРЅРёСЏ.
