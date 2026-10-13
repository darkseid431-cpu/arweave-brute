# arweave-brute

Self-contained brute-force engine for the **Arweave community puzzles**
(client-side CryptoJS-compatible decryption): a single-file CUDA solver plus a
CPU self-test gate that runs with nothing but `g++`.

## Algorithm (verified byte-for-byte against the puzzle page's CryptoJS)

1. **Key stretching** — `SHA-512(pw)` re-hashed 11,512 more times (11,513 total)
2. **KDF** — `EvpKDF(passphrase_hex, salt)` = MD5, 10,000 iterations → 144 B
   (128 B key + 16 B IV); the 8-byte salt comes from the ciphertext's
   OpenSSL `Salted__` prefix
3. **Cipher** — Rijndael-**1024** CBC: 128-byte key (Nk=32 words), **38 rounds**,
   128-bit block (Nb=4) — *not* standard AES-256 (14 rounds)
4. **Hit test** — plaintext contains the solution tag. AR keyfiles start with
   `"kty":"RSA"`, ETH wallets with `{"version":3,"id`, so only the first two CBC
   blocks need decrypting for the check (fast path)

There is no early exit inside a candidate — MD5×10000 dominates the cost.

## Why trust it

`make test` compiles **the exact solver source** on the CPU
(`SOLVER_HOST_ONLY` + a `cuda_runtime.h` stub — no CUDA toolchain needed) and
runs it against three known-answer vectors, i.e. real passwords of already-solved
puzzles:

| Puzzle | Password | Tag |
|---|---|---|
| pzl8 | `RasputinWilhelmAlekhine` | `"kty":"RSA"` |
| pzl5 | `*48GCEErisUmberCastlePicasso` | `"kty":"RSA"` |
| pzl7 | `Vivaldi227PermanentArnheimBulgakov1157Ulysses143176176209` | `{"version":3,"id` |

The decrypt path was cross-checked byte-for-byte against the puzzle page's own
CryptoJS implementation (round-trip + known-answer both bit-exact).

**Gate discipline:** never trust a `found=False` from a build that hasn't passed
its gate. The CUDA binary has its own gates: `make solver-gate` runs the CPU
`--selftest-pw` on every fixture *and* a real-kernel `--selftest-gpu` run.
A solver that misses a known-good password is worthless — that failure mode
(silent `found=False` from a stale/broken binary) is exactly what these gates
exist to prevent.

## Build & run

```sh
make test          # CPU gate — only needs g++
make solver        # CUDA build (nvcc); default ARCH=sm_75 (T4/Turing)
make solver-gate   # gate the nvcc-built binary (needs a GPU)
```

Brute force:

```sh
./our_solver --msg_file=fixtures/pzl8_msg.b64 \
             --keys_file=keys.txt \
             --solution_tag='"kty":"RSA"' \
             --block=128 --grid=120 --begin=0 --end=1048576
```

`keys.txt`: one line per position, comma-separated word list; candidates are
concatenations picking one word per position, in order. Exit 0 + `FOUND …`
prints the winning passphrase and writes `SOLVED_wallet.json`; exit 1 = no match.

Defaults: `--block=128 --grid=120` (128×120 = 3 blocks/SM on a 40-SM T4).
`--begin=N --end=M` grinds only the mixed-radix index range `[N, M)` of the
cartesian space (default: the whole space) — that's the shard granularity a
range ledger uses to resume an interrupted run without re-testing covered
combinations. An empty range (`begin >= end`) exits 2.

## Layout

```
our_solver.cu        solver: SHA-512 / EvpKDF / Rijndael-1024 + CUDA kernel
host_selftest.cpp    CPU gate: #includes our_solver.cu with CUDA stubbed out
cuda_stub/           minimal cuda_runtime.h so g++ can compile the host path
fixtures/            known-answer ciphertexts (pzl5/7/8)
Makefile             make test | solver | solver-gate
```

## License

MIT — see [LICENSE](LICENSE).
