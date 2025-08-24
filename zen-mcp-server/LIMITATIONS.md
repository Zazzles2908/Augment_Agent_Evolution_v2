# LIMITATIONS

2025-08-21
- Kimi model naming: The Moonshot API expects canonical model IDs (e.g., kimi-k2-0711-preview, kimi-k2-turbo-preview). If aliases or internal keys are passed directly, the API returns 404 esource_not_found_error. Resolved by mapping to ModelCapabilities.model_name in providers/kimi.py.
- Networking: This server disables environment proxy variables for OpenAI-compatible SDK initialization to avoid proxy interference. If your environment requires proxies, configure at the OS/network level or adjust provider initialization accordingly.
