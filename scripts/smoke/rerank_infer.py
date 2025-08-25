#!/usr/bin/env python3
import os, numpy as np
import tritonclient.http as http
from tritonclient.utils import np_to_triton_dtype

TRITON_URL=os.getenv("TRITON_URL","http://localhost:8000").replace("http://","")
MODEL=os.getenv("MODEL","qwen3_0_6b_reranking")

q="what is latency"
cands=["latency is delay","throughput is rps"]
max_q=max(len(q),1)
max_c=max(len(c) for c in cands)
q_ids=np.zeros((1,max_q),dtype=np.int64)
q_mask=np.zeros_like(q_ids)
q_arr=np.array([min(ord(c),255) for c in q],dtype=np.int64)
q_ids[0,:q_arr.shape[0]]=q_arr
q_mask[0,:q_arr.shape[0]]=1

c_ids=np.zeros((len(cands),max_c),dtype=np.int64)
c_mask=np.zeros_like(c_ids)
for i,t in enumerate(cands):
    arr=np.array([min(ord(c),255) for c in t],dtype=np.int64)
    c_ids[i,:arr.shape[0]]=arr
    c_mask[i,:arr.shape[0]]=1

client=http.InferenceServerClient(url=TRITON_URL)
inputs=[
  http.InferInput("query_ids",q_ids.shape,np_to_triton_dtype(q_ids.dtype)),
  http.InferInput("query_mask",q_mask.shape,np_to_triton_dtype(q_mask.dtype)),
  http.InferInput("cand_ids",c_ids.shape,np_to_triton_dtype(c_ids.dtype)),
  http.InferInput("cand_mask",c_mask.shape,np_to_triton_dtype(c_mask.dtype)),
]
inputs[0].set_data_from_numpy(q_ids)
inputs[1].set_data_from_numpy(q_mask)
inputs[2].set_data_from_numpy(c_ids)
inputs[3].set_data_from_numpy(c_mask)
req_out=[http.InferRequestedOutput("scores",binary_data=True)]
res=client.infer(MODEL,inputs,outputs=req_out,timeout=60)
scores=res.as_numpy("scores")
print(scores)

