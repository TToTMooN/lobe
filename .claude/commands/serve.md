Start a policy server for inference.

Ask the user for:
1. **Checkpoint path**: path to trained checkpoint
2. **Server type**: openpi (for pi0/pi0.5) or websocket (for xvla/walloss)
3. **Host**: (default: 0.0.0.0)
4. **Port**: (default: 8111 for openpi, 8000 for websocket)

Then run the appropriate command:

- **OpenPI**: `openpi serve --checkpoint <path> --port <port>`
- **WebSocket**: `uv run python scripts/serve_policy.py --checkpoint <path> --host <host> --port <port>`

Verify the server starts successfully and report the connection URL.
