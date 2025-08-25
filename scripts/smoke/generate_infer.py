#!/usr/bin/env python3
import os, numpy as np
import tritonclient.http as http
from tritonclient.utils import np_to_triton_dtype

TRITON_URL=os.getenv("TRITON_URL","http://localhost:8000").replace("http://","")
MODEL=os.getenv("MODEL","glm45_air")

prompt="Explain Redis TTL in one line."
max_len=len(prompt)
ids=np.zeros((1,max_len),dtype=np.int64)
mask=np.zeros_like(ids)
arr=np.array([min(ord(c),255) for c in prompt],dtype=np.int64)
ids[0,:arr.shape[0]]=arr
mask[0,:arr.shape[0]]=1

client=http.InferenceServerClient(url=TRITON_URL)
inputs=[
  http.InferInput("input_ids",ids.shape,np_to_triton_dtype(ids.dtype)),
  http.InferInput("attention_mask",mask.shape,np_to_triton_dtype(mask.dtype)),
]
inputs[0].set_data_from_numpy(ids)
inputs[1].set_data_from_numpy(mask)
req_out=[http.InferRequestedOutput("tokens",binary_data=True)]
res=client.infer(MODEL,inputs,outputs=req_out,timeout=60)
toks=res.as_numpy("tokens")
print(toks.shape)

