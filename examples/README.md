# End-to-End Demo
Requirements
- pip install transformers tritonclient redis supabase numpy pyyaml
- Set SUPABASE_URL and SUPABASE_ANON_KEY in env (or edit examples/config/config.yaml)



Components
- end_to_end_demo.py: full pipeline demo
- config/config.yaml: endpoints and credentials
- utils/: small clients for Triton, Supabase, Redis

Run (after Triton is up and Supabase has RPC):
```
python examples/end_to_end_demo.py --config examples/config/config.yaml \
  --document examples/fixtures/sample.pdf \
  --question "Summarize the key points"
```

