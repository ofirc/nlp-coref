#!/usr/bin/env python
import pandas as pd
import subprocess
import os

DOC_LEVEL_PATH = 'predictions_document_level.parq'
OUT_PATH = 'test_inter_paragraph.parq'

if os.path.exists(DOC_LEVEL_PATH):
    os.remove(DOC_LEVEL_PATH)
cmd = ['python', './convert_par_pred_to_doc_pred.py', 'outputs/predictions.csv']
subprocess.check_call(cmd)
assert os.path.exists(DOC_LEVEL_PATH)

doc_level_df = pd.read_parquet(DOC_LEVEL_PATH)
test_df = pd.read_parquet('test.parq')

out_df = pd.merge(doc_level_df, test_df, on='id')
out_df.to_parquet(OUT_PATH)
print(f'Generated {OUT_PATH}')
