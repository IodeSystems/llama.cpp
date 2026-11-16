# IodeSystems llama.cpp patch queue

Local patches that corrallm's llama.cpp build carries, kept as git commits
instead of `.patch` files applied at build time. Purpose: per-patch history,
`git rerere` for rebases, `git bisect`, and later a CI that builds and signs
binaries from a git ref.

**Nothing is pushed.** This repo is local only. Both remotes have the push URL
`DISABLED-no-push`:

- `upstream` = https://github.com/ggml-org/llama.cpp.git (fetch only)
- `local-src` = `~/.corrallm/tools/llama.cpp/src` (the tree the clone came from)

## Branch scheme

`iode/<upstream release>`, one commit per patch on top of the release tag,
plus this README as the last commit.

- `iode/v1.0.0` = `v1.0.0` (`b1b770d90`) + p1..p10 + README.

The p1..p10 tree is byte-identical to `b1b770d90` + `~/.corrallm/tools/llama.cpp/patches/p*.patch`
applied in name order (checked 2026-11-15). Name order puts p10 second (`LC_ALL=C`: p1, p10, p2, ...);
p10 touches no file another patch touches, so the order does not matter.

Each commit message has a fixed header: **Upstream** (PR + state), **Gate**
(env var / flag and default), **Effect** (measured, with units and where),
**Status** (live / canary / building / retire-when).

## Patches

| Patch | Gate | Default | Upstream | Status | Effect |
|---|---|---|---|---|---|
| p1 reuse checkpoint | none | **always on** | port of #30683 (open) | live 11-13 | dup checkpoint save on 45 % of restores removed; ~0.13 s/request avg |
| p2 end checkpoint | none | **always on** | local | live 11-13 | drops a checkpoint used 1 % of the time, 35 % of checkpoint copies |
| p3 pinned cache pool | `LLAMA_HOST_POOL`, `_RESERVE`, `_IDLE_S`, `--cache-overflow` | **pool on** (RESERVE off) | local | live 11-10 | not isolated; pageable copies were ~1.1-1.4 GB/s |
| p4 cache LRU + metrics | `--cache-ram-policy` | fifo (= upstream); metrics + `id_slot` **always on** | local | live 11-10 | /metrics hit 92.5 % (11-13); policy effect not isolated |
| p5 cache compaction | `LLAMA_HOST_POOL_COMPACT_MAX_MIB` | **4096 MiB (on)**, acts with RESERVE=1 | local | live 11-12 | frag evictions 13.6/h vs ~40/h (-66 %); tokens cached 91.1 vs 87.2 % |
| p6 cross-slot share | `LLAMA_XSLOT`, `_MIN` | off, 256 tok | local (cf. #30970) | canary 11-14, kept | 892k tok reused / 16 h; decode p50 79.1 vs 77.3 tok/s |
| p7 prefix checkpoint | `LLAMA_PREFIX_CKPT`, `_MIN`, `_BOUNDARIES` | off, 256 tok, 0 | local | canary 11-15 | no turn-1 gain seen in 47 min (copy path never fired) |
| p8 parallel sampling | `LLAMA_PAR_SAMPLING`, `_THREADS` | off, 4; race guards **always on** | local | building (S2) | CPU: sampling wall -30..40 %; expect ~4 % step time on Swift |
| p9 fused FA q8 MMA | `GGML_CUDA_FATTN_MMA_FUSED` | **1 (on)** | rebase of #31802 (open, unreviewed); trimmed build: fused only q8_0/q8_0 at head dim 256, no 512 vec (`-DGGML_CUDA_FATTN_MMA_FUSED_ALL` = full set) | S2 canary since 2026-11-15 19:49 PST | target ~9.4 ms/step dequant; author: decode +18..63 % at 32k..131k |
| p10 recurrent rollback displace | none | **always on** (only with `n_rs_seq` > 0) | local, upstream bug not reported yet | S2 canary 2026-11-15 | fixes the S2 `seq_rm ... p1=-1` aborts and silent recurrent-state corruption after a rollback; CPU fuzz 12 bad -> 0 |

Design goal (plan/icebox.md): every patch env-gated and off by default, so
the fork with no env set behaves like upstream. **Not met yet** by p1, p2
(no gate), p3 (pool on by default), p4 (metrics and `id_slot` always on),
p5 (compaction on by default), p9 (fused path on by default), p10 (a bug fix,
no gate by design). p8's race
guards are ungated but do not change behavior.

Retire p1 when #30683 merges; retire p9 when #31802 merges (or rebase onto
the merged form).

## Rebase onto a new upstream tag

rerere is enabled in this repo (`git config rerere.enabled true`), so a
conflict resolved once is reused on the next rebase.

```sh
git fetch upstream --tags
git switch -c iode/vX.Y.Z iode/v1.0.0
git rebase --onto vX.Y.Z v1.0.0 iode/vX.Y.Z
# on conflict: fix, git add, git rebase --continue (rerere records it)
git range-diff v1.0.0..iode/v1.0.0 vX.Y.Z..iode/vX.Y.Z   # review what changed per patch
```

A patch that upstream merged shows up as an empty commit during the rebase:
drop it (`git rebase --skip`) and note it in this README.

## Verify a branch

```sh
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="86;120"
cmake --build build -j --target llama-server test-backend-ops
# FA kernels (p9): compare CUDA vs CPU
./build/bin/test-backend-ops -o FLASH_ATTN_EXT
GGML_CUDA_FATTN_MMA_FUSED=2 ./build/bin/test-backend-ops -o FLASH_ATTN_EXT   # fused path must be taken
# server (p1-p8): pytest subsets
cd tools/server/tests && pip install -r requirements.txt
./tests.sh unit/test_completion.py unit/test_chat_completion.py unit/test_metrics.py \
           unit/test_slot_save.py unit/test_speculative.py unit/test_ctx_shift.py
```

Run the server subsets once with no env set and once with the gates on
(`LLAMA_XSLOT=1 LLAMA_PREFIX_CKPT=1 LLAMA_PAR_SAMPLING=1`); failures must be
the same as on the upstream tag.

## Review

```sh
git log --stat iode/v1.0.0 ^v1.0.0
git show <commit>            # one patch with its header
git diff v1.0.0 iode/v1.0.0 -- tools/server/   # all server changes at once
```
