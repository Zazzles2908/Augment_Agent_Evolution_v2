#!/usr/bin/env python3
import os, sys, numpy as np
import tritonclient.http as http
from tritonclient.utils import np_to_triton_dtype

TRITON_URL=os.getenv("TRITON_URL","http://localhost:8000").replace("http://","")
MODEL=os.getenv("MODEL","qwen3_4b_embedding")

texts=["hello world","quick brown fox"]
max_len=max(len(t) for t in texts)
ids=np.zeros((len(texts),max_len),dtype=np.int64)
mask=np.zeros_like(ids)
for i,t in enumerate(texts):
    arr=np.array([min(ord(c),255) for c in t],dtype=np.int64)
    ids[i,:arr.shape[0]]=arr
    mask[i,:arr.shape[0]]=1

client=http.InferenceServerClient(url=TRITON_URL)
in_ids=http.InferInput("input_ids",ids.shape,np_to_triton_dtype(ids.dtype))
in_ids.set_data_from_numpy(ids)
in_mask=http.InferInput("attention_mask",mask.shape,np_to_triton_dtype(mask.dtype))
in_mask.set_data_from_numpy(mask)
req_out=[http.InferRequestedOutput("embedding",binary_data=True)]
res=client.infer(MODEL,[in_ids,in_mask],outputs=req_out,timeout=60)
emb=res.as_numpy("embedding")
print(emb.shape)

