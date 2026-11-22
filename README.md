# Norn

Quick decisions about a text, with probabilities, on the CPU: yes/no questions, a choice among options, a level on a scale.
Norn runs jevos, a small decision model (about 0.9 billion parameters), and answers in jev's wire format. It is meant as one
decision service for Heimdall, Sarina and Icarus.

## What it does

A request has a text (`state`) and named questions:

```json
{"state": "The engine idles rough when warm and the check engine light shows P0301.",
 "questions": {
   "misfire": {"type": "noul", "instructions": "Does this describe a misfire?"},
   "system":  {"type": "choice", "instructions": "Which system should be checked first?",
               "criteria": {"ignition": "spark plugs, coils", "fuel": "injectors, pump", "brakes": "pads, rotors"}},
   "urgency": {"type": "score", "instructions": "How urgent is this?", "criteria": ["low", "medium", "high", "critical"]}}}
```

- `noul`: P(yes).
- `choice`: each option asked against the others as a yes/no question; the answers normalized to one distribution, the most
  probable option and a confidence.
- `score`: "this level or higher?" for each level above the lowest, made monotonic, a distribution over the levels, the
  expected level and a confidence.

The prompt layout, the answer tokens ("0" and "1") and how yes/no answers become a choice or a score follow jev
(see NOTICE). All the questions about one text read it once: the prompts' common start runs through the model once.

## The engine and the model

`src/lib.rs` runs the model with Candle on the CPU (`src/llama.rs`: Candle's quantized llama, with the head size read from
the GGUF, which jevos needs). The model is `jevos-v4-q8_0.gguf` and its `tokenizer.json` from
[feder-cr/jev](https://huggingface.co/feder-cr/jev), downloaded, never part of this repository.

```sh
cargo build --release
target/release/norn-bench MODEL.gguf TOKENIZER.json [REQUESTS.jsonl]   # without a file: the latency case below
cargo test --release --lib
```

## Measurements (2026-11-21, ThinkPad X1 Yoga Gen 8, i7-1365U, 12 threads)

One request: a text of 177 tokens with three questions (a yes/no, a choice of three, a score of four: eight yes/no prompts),
median of 20 after 3 warm-up runs.

| Engine | Median |
| --- | --- |
| jev's own binary (OpenVINO INT8) | 704 ms |
| llama.cpp's `llama-server` (the same GGUF, the eight prompts in one request, answers limited to "0" or "1") | 325 ms |
| Norn's Candle engine (8 threads, built for the CPU) | about 4,300 ms |

Answers against jev's binary on 50 fixed requests (yes/no, choices, scores; the largest difference in each request's
probabilities): median 0.003, 90th percentile 0.019, 45 of 50 within 0.02, the largest 0.078 (a choice of four). The two
8-bit formats (OpenVINO INT8 and GGUF q8_0) round differently. Candle and llama.cpp give the same answers on the same GGUF.

Simple checks: "The sky is clear and blue today." gives P(yes) 0.99 for "Is the sky blue?" and 0.03 for "Is it raining?";
a rough idle with P0301 chooses ignition at 0.94.

## Known limitations

The Candle engine is correct but about six times slower than jev's binary on this laptop; llama.cpp's server answers the
same requests about twice as fast as jev's binary. There is no service yet: `norn-bench` answers requests from a file.

## License

Mozilla Public License 2.0 (LICENSE). The parts ported from jev and Candle keep their MIT notices (NOTICE).
