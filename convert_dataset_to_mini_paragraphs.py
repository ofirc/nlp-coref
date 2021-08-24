#!/usr/bin/env python
import os
import subprocess

SCRIPT = 'convert_span_to_generative.py'
JSONLINES = ['train.jsonlines', 'dev.jsonlines', 'test.jsonlines']
PARQ = ['train.parq', 'dev.parq', 'test.parq']
PARQ_with_paragraphs = ['train_with_paragraphs.parq', 'dev_with_paragraphs.parq', 'test_with_paragraphs.parq']

print("Converting JSONLines to parquet...")
for js, p in zip(JSONLINES, PARQ):
    if not os.path.exists(js):
        print(f"ERROR: {js} is missing.")
        exit(-1)
    if os.path.exists(p):
        os.remove(p)
    cmd = ['python', SCRIPT, js]
    print(f"Running {' '.join(cmd)}")
    subprocess.check_call(cmd)

SCRIPT = 'document_to_paragraphs.py'
print("Converting documents to mini-documents...")
for p, pa in zip(PARQ, PARQ_with_paragraphs):
    assert os.path.exists(p)
    cmd = ['python', SCRIPT, p]
    print(f"Running {' '.join(cmd)}")
    subprocess.check_call(cmd)
    assert os.path.exists(pa)
