"""Run real BF16 generation on either shared inference engine."""

import argparse
import json
import math


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["vllm", "sglang"], required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    prompt = "The capital of France is"
    if args.backend == "vllm":
        from vllm import LLM, SamplingParams

        engine = LLM(model=args.model, dtype="bfloat16", max_model_len=512,
                     max_num_seqs=2, gpu_memory_utilization=.15, enforce_eager=True)
        output = engine.generate([prompt], SamplingParams(temperature=0, max_tokens=8, logprobs=1))[0].outputs[0]
        tokens = list(output.token_ids)
        logprobs = [row[token].logprob for token, row in zip(tokens, output.logprobs, strict=True)]
        result = {"backend": args.backend, "text": output.text, "tokens": tokens}
    else:
        from sglang import Engine

        engine = Engine(model_path=args.model, dtype="bfloat16", context_length=512,
                        max_running_requests=2, mem_fraction_static=.15,
                        disable_cuda_graph=True, grammar_backend="xgrammar")
        try:
            output = engine.generate(prompt=prompt, sampling_params={"temperature": 0, "max_new_tokens": 8},
                                     return_logprob=True)
            rows = output["meta_info"]["output_token_logprobs"]
            tokens = [row[1] for row in rows]
            logprobs = [row[0] for row in rows]
            result = {"backend": args.backend, "text": output["text"], "tokens": tokens}
        finally:
            engine.shutdown()
    assert tokens and len(tokens) == len(logprobs) and all(math.isfinite(p) for p in logprobs)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
