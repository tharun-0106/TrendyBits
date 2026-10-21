#!/usr/bin/env python
"""Import / generate the three TrendyBits datasets (IEEE Dataport schemas), audit them and create
stratified train/valid/test splits BEFORE any model fitting or prompt tuning.

IEEE Dataport downloads require a (free) IEEE account, so files are downloaded manually from:
  * https://ieee-dataport.org/documents/trendybits-news-summarization-dataset
  * https://ieee-dataport.org/documents/trendybits-fake-news-dataset
and passed to this script. Without files, schema-faithful SYNTHETIC benchmark partitions are generated.

Examples
  python scripts/import_ieee_dataport.py --generate                      # synthetic partitions (default)
  python scripts/import_ieee_dataport.py --newssumm NewsSumm.xlsx
  python scripts/import_ieee_dataport.py --liar train.tsv valid.tsv test.tsv
  python scripts/import_ieee_dataport.py --isot-true True.csv --isot-fake Fake.csv --bharat bharat_fakenewskosh_en.csv
  python scripts/import_ieee_dataport.py --newssumm NewsSumm.csv --load-into-db 200
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):  # Windows consoles default to cp1252
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app import database as db  # noqa: E402
from app import datasets as ds  # noqa: E402
from app.text_utils import classify_topic  # noqa: E402


def finalize(name: str, df, origin: str) -> dict:
    clean, rep = ds.audit(df, name)
    man = ds.save_splits(name, ds.stratified_split(clean, ds.SCHEMAS[name]["label"]), rep, origin=origin)
    print(f"\n[{name}] origin={origin}")
    print(json.dumps({k: man[k] for k in ("sizes", "test_sha256")}, indent=1))
    print("audit:", json.dumps(rep))
    return man


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--newssumm", nargs="+", help="NewsSumm CSV/TSV/XLSX/JSON file(s) with Source, Content, Summary")
    ap.add_argument("--liar", nargs="+", help="LIAR TSV file(s) (train/valid/test are pooled and re-split stratified)")
    ap.add_argument("--isot-true", help="ISOT True.csv")
    ap.add_argument("--isot-fake", help="ISOT Fake.csv")
    ap.add_argument("--bharat", help="BharatFakeNewsKosh English subset (CSV/XLSX) with a label column")
    ap.add_argument("--integrated", help="Already-merged integrated CSV with date,title,subject,text,label")
    ap.add_argument("--generate", action="store_true", help="(Re)generate synthetic partitions for datasets not supplied")
    ap.add_argument("--load-into-db", type=int, default=0, metavar="N",
                    help="Also insert N NewsSumm test articles into trendybits.db so they appear in the feed")
    a = ap.parse_args()

    supplied = set()
    if a.newssumm:
        finalize("newssumm", ds.import_newssumm(a.newssumm), "IEEE Dataport: " + ", ".join(Path(p).name for p in a.newssumm))
        supplied.add("newssumm")
    if a.liar:
        finalize("liar", ds.import_liar(a.liar), "IEEE Dataport/LIAR: " + ", ".join(Path(p).name for p in a.liar))
        supplied.add("liar")
    if a.isot_true or a.isot_fake or a.bharat or a.integrated:
        df = ds.import_integrated(a.isot_true, a.isot_fake, a.bharat, a.integrated)
        finalize("integrated", df, "IEEE Dataport: ISOT + BharatFakeNewsKosh (English)")
        supplied.add("integrated")
    if a.generate or not supplied:
        gens = {"newssumm": (ds.generate_newssumm, 700), "liar": (ds.generate_liar, 4000), "integrated": (ds.generate_integrated, 4000)}
        for name, (gen, n) in gens.items():
            if name not in supplied:
                finalize(name, gen(n), "synthetic-benchmark (schema-faithful, template generated)")
        ds.build_retrieval_benchmark()
        print("\n[retrieval benchmark] data/datasets/retrieval_benchmark.json written")
    if a.load_into_db:
        db.init_db(seed=True)
        test = ds.load_split("newssumm", "test").head(a.load_into_db)
        n_new = 0
        for _, r in test.iterrows():
            first = r["Content"].split(". ")[0][:140]
            _, ins = db.upsert_article(dict(title=first, body=r["Content"], source=r["Source"], domain=None,
                                            url=f"https://dataset.local/newssumm/{r['row_id']}", published_at=None,
                                            topic=classify_topic(first, r["Content"]), kind="news", extra={"dataset": "newssumm"}))
            n_new += ins
        print(f"\n[db] inserted {n_new} NewsSumm articles into {db.config.DB_PATH}")


if __name__ == "__main__":
    main()
