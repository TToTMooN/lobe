Research a VLA/policy method or technique for potential integration.

Arguments: $ARGUMENTS (method name or topic, e.g. "octo", "action chunking", "multi-gpu VLA training")

1. **Web search**: Find the original paper, blog posts, and benchmark results.
   - Focus on: LIBERO, PushT, robomimic results
   - Note: model size, training cost (steps, GPUs, time), dataset requirements

2. **Code search**: Search GitHub for implementations.
   - Check: lerobot ecosystem, HuggingFace Hub, official repos
   - Look for: pretrained weights, training configs, known issues

3. **Compare with our baselines**: How does this method compare to what we already have?
   - Reference the published baselines table in CLAUDE.md
   - Is it worth integrating? What's the effort vs. improvement tradeoff?

4. **Summarize findings** in a concise report:
   - Method overview (1-2 sentences)
   - Published results on our target benchmarks
   - Available implementations and pretrained weights
   - Integration effort estimate (trivial/moderate/significant)
   - Recommendation: integrate now / investigate later / skip

Do NOT implement anything — this is research only. Use /add-method to actually integrate.
