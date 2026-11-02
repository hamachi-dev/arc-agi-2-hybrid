# Kaggle フォークへの適用手順 (failed-in-aimo → 本番用)

ローカル検証済み: selftest 9/9, queue 120/120・240/240, merge単調性,
LegC no-op, validator。以下だけ替えれば残りはそのまま動く。

## 0. Notebook設定 (変更なし・確認のみ)

- Accelerator: GPU (L4) / Internet: OFF
- Data: `arc-prize-2026-arc-agi-2` をアタッチ
- Models: `sorokin/qwen3_4b_grids15_sft139` (Apache 2.0, 必須),
  Qwen2.5-Coder-7B-Instruct (Leg C用。無ければ自動no-op)
- `sorokin/pip-install-unsloth-flash-patch` をアタッチ (上流通り)

## 1. starter セル (必須・バグ修正)

上流 `src/starter.py:58-61` は本番モード以外4タスクしか積まない
デバッグフィルタ。セル内容を `src/starter_fixed.py` で全置換。
効果: 全件キュー + コスト昇順 (any-timeで最多タスク完了) +
LegC済みスキップ + パスENV化。`--dry-run` でキュー確認可。

## 2. 実行セル (cell 9・ファイル名のみ)

```diff
-!... python starter.py --end-time {global_end_time}
+!... python starter_fixed.py --end-time {global_end_time}
```

## 3. arc_solver worker (推奨・堅牢化)

1パズルの例外でworkerごと死ぬ (per-key try/exceptなし)。
`worker()` 内、`if key is None: break` 以降のループ本体を
`try:` で囲み、`except Exception as e: print(...); continue` を追加。
部分出力は残るので submission は常に有効 (cell10はOUT_DIRの
存在分から [[0]] 埋めで生成)。

## 4. 初回投稿計画 (1日1件制限のため)

1. まず `LEGC_BUDGET_H=0` でベースのみ投稿 → 再現ベースライン取得
2. 次に `LEGC_BUDGET_H=1.0` でハイブリッド投稿 → mergeログの
   `promoted=` と `local score: base X -> merged Y` で上乗せ確認
3. `promoted=0` が続く場合のみ k1/k2 増加や per_task_cap 延長を検討

## 5. 変える必要がないもの

- `arc_loader.py`, `arc_decoder.py`, 集約セル, mergeセル (検証済み)
- Leg C (`arc_induction_v2.py`): モデル無し・予算不足時は自動no-op。
  パスはKaggle実機では一致する
- ベースモデルは公開・Apache 2.0 (入賞時のライセンス条件を満たす)

## 6. 適用記録 (2026-10-23, hamatz/failed-in-aimo v4/v5/v6)

- P1/P2/P3 適用済み・サーバー検証済み (9 cells, 他セル無改変, ast OK)
- GPU L4 / Internet OFF / TPU OFF は復旧済み
- ⚠️ 要手動復旧: QuickSave の副作用で Data 添付3件が外れた。
  Notebook画面 → 右ペイン Add data → 以下を再アタッチ:
  1. Competition: arc-prize-2026-arc-agi-2
  2. Model: sorokin/qwen3_4b_grids15_sft139 (Transformers/bfloat16/1)
  3. Script: sorokin/pip-install-unsloth-flash-patch
  (API経由の復旧は不可のため。CLI push は実行キューを伴うので未使用)
