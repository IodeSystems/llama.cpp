#include "server-slot-sched.h"

#include "common.h"
#include "llama.h"
#include "log.h"

#include <algorithm>
#include <unordered_map>
#include <unordered_set>

//
// server_prompt
//

server_prompt_reuse server_prompt::get_reuse(int32_t n_prefix, int32_t n_unshifted, bool has_new_tokens, int32_t n_swa, llama_pos pos_min) const {
    server_prompt_reuse res;
    res.n_past   = n_prefix;
    res.pos_next = tokens.pos_next(n_prefix);

    // the first evaluated token attends to the positions > pos_eval - n_swa, so a SWA state
    // starting at pos_eval - n_swa + 1 is enough, while a state without SWA must end before pos_eval
    const llama_pos pos_eval = res.pos_next - (has_new_tokens ? 0 : 1);

    // a state or a checkpoint is useful when its pos_min is below this threshold
    const auto pos_min_thold = std::max(0, n_swa > 0 ? pos_eval - n_swa + 2 : pos_eval);

    if (pos_min < pos_min_thold) {
        return res;
    }

    // search for a context checkpoint
    const auto it = std::find_if(
        checkpoints.rbegin(),
        checkpoints.rend(),
        [&](const auto & ckpt) {
            const auto & cur = *ckpt;
            // guarantee that a checkpoint will result in at least one token being processed [TAG_PROMPT_LOGITS]
            SRV_TRC("checking checkpoint with [%d, %d] against %d...\n", cur.pos_min, cur.pos_max, pos_min_thold);
            // workaround for [TAG_CHECKPOINTS_FIX_POS_MIN]
            if (cur.pos_max > res.pos_next) {
                return false;
            }
            // cache reuse shifted chunks over the tokens this checkpoint covers
            if (n_prefix > n_unshifted && cur.n_tokens > n_unshifted) {
                return false;
            }
            return cur.pos_min < pos_min_thold || cur.pos_min == 0;
        }
    );

    if (it == checkpoints.rend()) {
        res.n_past   = 0;
        res.pos_next = 0;
        return res;
    }

    const auto & cur = **it;

    res.pos_next = std::min(res.pos_next, std::max(cur.pos_min + 1, cur.pos_max));
    res.n_past   = std::min(tokens.size_up_to_pos(res.pos_next), (size_t) cur.n_tokens);
    res.ckpt     = *it;

    return res;
}

//
// server_prompt_cache
//

size_t server_prompt_cache::size() const {
    size_t res = 0;

    // entries share checkpoints (with slots and with each other): count each once
    std::unordered_set<const common_prompt_checkpoint *> seen;
    for (const auto & state : states) {
        res += state.data.size();
        for (const auto & ckpt : state.prompt.checkpoints) {
            if (seen.insert(ckpt.get()).second) {
                res += ckpt->real_size();
            }
        }
    }

    return res;
}

size_t server_prompt_cache::n_tokens() const {
    size_t res = 0;

    for (const auto & state : states) {
        res += state.prompt.n_tokens();
    }

    return res;
}

size_t server_prompt_cache::size_overflow() const {
    size_t res = 0;

    for (const auto & state : states) {
        if (state.data.overflow) {
            res += state.data.size();
        }
    }

    return res;
}

std::list<server_prompt_cache_state>::iterator server_prompt_cache::erase_state(std::list<server_prompt_cache_state>::iterator it) {
    for (const auto & ckpt : it->prompt.checkpoints) {
        if (ckpt.use_count() > 1) {
            common_prompt_checkpoint_set_owner(*ckpt, COMMON_HOST_POOL_SLOT);
        }
    }

    return states.erase(it);
}

void server_prompt_cache::log_overflow(const char * what, size_t bytes) {
    const int64_t t_now = ggml_time_us();
    if (t_now - t_ovf_log < 10*1000*1000) {
        return;
    }

    const auto st = common_host_pool_get_stats();

    SRV_INF("prompt cache overflow (%s): %s %.1f MiB; since last log: %" PRIu64 " saves over budget, evicted %" PRIu64 " entries (%.1f MiB), "
            "released %" PRIu64 " idle regions (%.1f MiB), %" PRIu64 " entries to heap (%.1f MiB); "
            "arena fragmented %" PRIu64 " (last: request %.1f MiB, largest free %.1f MiB); "
            "cache %.1f / %.1f MiB real (idle %.1f MiB), heap %.1f / %.1f MiB, %zu entries\n",
            overflow_heap ? "heap" : "evict", what, bytes / (1024.0 * 1024.0),
            n_ovf_saves, n_ovf_evicted, bytes_ovf_evicted / (1024.0 * 1024.0),
            st.n_release_sync - n_ovf_rel_sync0, (st.bytes_release_sync - bytes_ovf_rel_sync0) / (1024.0 * 1024.0),
            n_ovf_heap, bytes_ovf_heap / (1024.0 * 1024.0),
            st.n_frag - n_ovf_frag0, st.frag_request / (1024.0 * 1024.0), st.frag_largest / (1024.0 * 1024.0),
            (size() - size_overflow()) / (1024.0 * 1024.0), limit_size / (1024.0 * 1024.0),
            st.idle[COMMON_HOST_POOL_CACHE] / (1024.0 * 1024.0),
            size_overflow() / (1024.0 * 1024.0), limit_overflow / (1024.0 * 1024.0), states.size());

    t_ovf_log           = t_now;
    n_ovf_saves         = 0;
    n_ovf_evicted       = 0;
    bytes_ovf_evicted   = 0;
    n_ovf_heap          = 0;
    bytes_ovf_heap      = 0;
    n_ovf_rel_sync0     = st.n_release_sync;
    bytes_ovf_rel_sync0 = st.bytes_release_sync;
    n_ovf_frag0         = st.n_frag;
}

// eviction preference under a policy: higher goes first
static double server_prompt_cache_score(common_cache_ram_policy policy, const server_prompt_cache_state & state, int64_t t_now, size_t pos) {
    if (policy == COMMON_CACHE_RAM_POLICY_FIFO) {
        return -(double) pos;
    }
    const double age = (double) std::max<int64_t>(0, t_now - state.t_last_used);
    return policy == COMMON_CACHE_RAM_POLICY_LRU ? age : age / (1.0 + state.prompt.n_hits);
}

std::list<server_prompt_cache_state>::iterator server_prompt_cache::victim() {
    GGML_ASSERT(!states.empty());

    if (policy == COMMON_CACHE_RAM_POLICY_FIFO) {
        return states.begin();
    }

    const int64_t t_now = ggml_time_us();

    // ties go to the entry nearest the front (oldest saved)
    auto   res       = states.begin();
    double score_res = -1.0;
    size_t pos       = 0;
    for (auto it = states.begin(); it != states.end(); ++it, ++pos) {
        const double score = server_prompt_cache_score(policy, *it, t_now, pos);
        if (score > score_res) {
            score_res = score;
            res       = it;
        }
    }

    return res;
}

std::list<server_prompt_cache_state>::iterator server_prompt_cache::victim_hole(size_t size) {
    GGML_ASSERT(!states.empty());

    const int64_t t_now = ggml_time_us();

    // policy order, ties to the front (stable sort)
    std::vector<std::pair<double, std::list<server_prompt_cache_state>::iterator>> order;
    size_t pos = 0;
    for (auto it = states.begin(); it != states.end(); ++it, ++pos) {
        order.emplace_back(server_prompt_cache_score(policy, *it, t_now, pos), it);
    }
    std::stable_sort(order.begin(), order.end(), [](const auto & a, const auto & b) { return a.first > b.first; });

    for (const auto & [score, it] : order) {
        // what evicting the entry frees: its state buffers and the checkpoints nothing else holds
        std::vector<const void *> freed;
        for (const auto * buf : { &it->data.main, &it->data.drft }) {
            if (buf->capacity() > 0) {
                freed.push_back(buf->data());
            }
        }
        for (const auto & ckpt : it->prompt.checkpoints) {
            if (ckpt.use_count() == 1) {
                for (const auto * buf : { &ckpt->data_tgt, &ckpt->data_dft }) {
                    if (buf->capacity() > 0) {
                        freed.push_back(buf->data());
                    }
                }
            }
        }
        if (common_host_pool_largest_if_freed(freed) >= size) {
            return it;
        }
    }

    return victim();
}

std::vector<const void *> server_prompt_cache::movable_blocks() const {
    std::vector<const void *> res;

    // a checkpoint is the cache's alone if every reference to it is held by cache entries
    std::unordered_map<const common_prompt_checkpoint *, long> refs;
    for (const auto & state : states) {
        for (const auto & ckpt : state.prompt.checkpoints) {
            refs[ckpt.get()]++;
        }
    }

    std::unordered_set<const common_prompt_checkpoint *> seen;
    for (const auto & state : states) {
        if (!state.data.overflow) {
            for (const auto * buf : { &state.data.main, &state.data.drft }) {
                if (buf->capacity() > 0) {
                    res.push_back(buf->data());
                }
            }
        }
        for (const auto & ckpt : state.prompt.checkpoints) {
            if (ckpt.use_count() != refs[ckpt.get()] || !seen.insert(ckpt.get()).second) {
                continue;
            }
            for (const auto * buf : { &ckpt->data_tgt, &ckpt->data_dft }) {
                if (buf->capacity() > 0) {
                    res.push_back(buf->data());
                }
            }
        }
    }

    return res;
}

std::list<server_prompt_cache_state>::iterator server_prompt_cache::evict(std::list<server_prompt_cache_state>::iterator it, server_prompt_cache_evict_reason reason) {
    metrics.add_evicted(reason, std::max<int64_t>(0, ggml_time_us() - it->t_last_used) / 1e6);

    return erase_state(it);
}

server_prompt_cache_metrics server_prompt_cache::get_metrics() const {
    server_prompt_cache_metrics res = metrics;

    res.enabled   = true;
    res.n_entries = states.size();
    res.n_bytes   = size();

    if (common_host_pool_enabled()) {
        const auto st = common_host_pool_get_stats();
        res.n_compactions         = st.n_compact;
        res.n_compactions_skipped = st.n_compact_skipped;
        res.bytes_compacted       = st.bytes_compact;
        res.compaction_s          = st.t_compact_us / 1e6;
        res.n_arenas              = st.n_arenas;
        for (int i = 0; i < 2; i++) {
            res.arena_free[i]    = st.arena_free[i];
            res.arena_largest[i] = st.arena_largest[i];
        }
    }

    return res;
}

server_prompt_cache_state * server_prompt_cache::alloc(const server_prompt & prompt, size_t state_size_tgt, size_t state_size_dft) {
    const int64_t t_now = ggml_time_us();

    // first check if the current state is contained fully in the cache
    for (auto it = states.begin(); it != states.end(); ++it) {
        const int cur_lcp_len = it->prompt.tokens.get_common_prefix(prompt.tokens);

        if (cur_lcp_len == (int) prompt.tokens.size()) {
            SRV_TRC("%s", " - prompt is already in the cache, skipping\n");
            it->t_last_used = t_now; // the slot was still using it
            return nullptr;
        }
    }

    // calculate checkpoints size to see if it will fit with the prompt (real allocated bytes)
    size_t checkpoints_size = 0;
    for (const auto & ckpt : prompt.checkpoints) {
        checkpoints_size += ckpt->real_size();
    }

    const bool pool = common_host_pool_enabled();

    // real bytes of the new entry (pooled state buffers are rounded up to a size class)
    const size_t state_size_new = common_host_pool_size_class(state_size_tgt) + common_host_pool_size_class(state_size_dft) + checkpoints_size;

    // skip over-limit entries to avoid disturbing the cache
    const size_t limit_entry = overflow_heap ? std::max(limit_size, limit_overflow) : limit_size;
    if (limit_size > 0 && state_size_new > limit_entry) {
        SRV_WRN(" - prompt state size %.3f MiB exceeds cache size limit %.3f MiB, skipping\n",
                state_size_new / (1024.0 * 1024.0), limit_entry / (1024.0 * 1024.0));
        return nullptr;
    }

    // the new entry continues the history of the prompts it supersedes
    int64_t  t_created = prompt.t_created > 0 ? prompt.t_created : t_now;
    uint32_t n_hits    = prompt.n_hits;

    // remove any cached prompts that are fully contained in the current prompt
    for (auto it = states.begin(); it != states.end();) {
        const int len = it->prompt.tokens.get_common_prefix(prompt.tokens);

        if (len == (int) it->prompt.tokens.size()) {
            SRV_TRC(" - removing obsolete cached prompt with length %d\n", len);

            t_created = std::min(t_created, it->prompt.t_created);
            n_hits    = std::max(n_hits,    it->prompt.n_hits);

            it = evict(it, SERVER_PROMPT_CACHE_EVICT_SUPERSEDED);
        } else {
            ++it;
        }
    }

    if (limit_size > 0 && !pool) {
        // no host pool: make room before allocating the new vectors to avoid breaching the limit
        while (!states.empty() && size() + state_size_new > limit_size) {
            const auto it = victim();
            SRV_WRN(" - making room for prompt cache entry, removing oldest entry (size = %.3f MiB)\n",
                    it->size() / (1024.0 * 1024.0));

            evict(it, SERVER_PROMPT_CACHE_EVICT_BUDGET);
        }
    }

    // with the host pool, the cache budget (limit_size) is enforced on real bytes, in use + idle,
    // by the pool: an allocation over budget first releases idle cache regions (oldest first),
    // then throws common_host_pool_over_budget and we evict the oldest entry and retry
    // (oldest = the policy's victim; one call = one counted eviction)
    const auto evict_oldest = [&](server_prompt_cache_evict_reason reason, size_t hole) {
        // fragmentation: prefer the entry whose blocks (and the free space around them) make the hole
        const auto it = reason == SERVER_PROMPT_CACHE_EVICT_FRAGMENTATION ? victim_hole(hole) : victim();
        n_ovf_evicted++;
        bytes_ovf_evicted += it->size();
        SRV_TRC(" - making room for prompt cache entry, removing oldest entry (size = %.3f MiB)\n",
                it->size() / (1024.0 * 1024.0));
        evict(it, reason);
    };

    // the entry's checkpoints are counted by the cache from now on
    for (const auto & ckpt : prompt.checkpoints) {
        common_prompt_checkpoint_set_owner(*ckpt, COMMON_HOST_POOL_CACHE);
    }
    const auto ckpts_to_slot = [&]() {
        for (const auto & ckpt : prompt.checkpoints) {
            common_prompt_checkpoint_set_owner(*ckpt, COMMON_HOST_POOL_SLOT);
        }
    };

    common_state_data state_data_tgt;
    common_state_data state_data_dft;

    bool over     = false; // did not fit without releasing idle regions or evicting
    bool overflow = false;

    const uint64_t n_release_sync0 = pool ? common_host_pool_get_stats().n_release_sync : 0;

    // check if we can allocate enough memory for the new state
    try {
        // the checkpoints alone may push the cache over budget
        while (!common_host_pool_trim(COMMON_HOST_POOL_CACHE)) {
            over = true;
            if (states.empty()) {
                throw common_host_pool_over_budget();
            }
            evict_oldest(SERVER_PROMPT_CACHE_EVICT_BUDGET, 0);
        }

        // one free extent for both state buffers
        const size_t hole = common_host_pool_size_class(state_size_tgt) + common_host_pool_size_class(state_size_dft);
        bool compacted = false; // this attempt follows a compaction

        for (;;) {
            // the pool counts arena requests that failed although enough bytes were free
            const uint64_t n_frag0 = pool ? common_host_pool_get_stats().n_frag : 0;
            bool frag = false;
            try {
                common_host_pool_scope scope(COMMON_HOST_POOL_CACHE);
                state_data_tgt.resize(state_size_tgt);
                state_data_dft.resize(state_size_dft);
                break;
            } catch (const common_host_pool_over_budget &) {
                state_data_tgt = common_state_data();
                state_data_dft = common_state_data();
                over = true;
                frag = pool && common_host_pool_get_stats().n_frag != n_frag0;
            }

            // fragmented: move blocks only the cache holds before evicting anything
            if (frag && !compacted) {
                compacted = common_host_pool_compact(hole, movable_blocks());
                if (compacted) {
                    continue;
                }
            }
            compacted = false;

            if (overflow_heap) {
                try {
                    common_host_pool_scope scope(COMMON_HOST_POOL_OVERFLOW);
                    state_data_tgt.resize(state_size_tgt);
                    state_data_dft.resize(state_size_dft);
                    overflow = true;
                    break;
                } catch (const common_host_pool_over_budget &) {
                    state_data_tgt = common_state_data();
                    state_data_dft = common_state_data();
                }
            }

            if (states.empty()) {
                throw common_host_pool_over_budget();
            }
            evict_oldest(frag ? SERVER_PROMPT_CACHE_EVICT_FRAGMENTATION : SERVER_PROMPT_CACHE_EVICT_BUDGET, hole);
        }
    } catch (const common_host_pool_over_budget &) {
        ckpts_to_slot();

        n_ovf_saves++;
        SRV_WRN(" - prompt state size %.3f MiB does not fit the cache budget even when empty, skipping\n",
                state_size_new / (1024.0 * 1024.0));
        log_overflow("skipped", state_size_new);

        return nullptr;
    } catch (const std::bad_alloc & e) {
        ckpts_to_slot();

        SRV_ERR("failed to allocate memory for prompt cache state: %s\n", e.what());

        limit_size = std::max<size_t>(1, 0.4*size());
        if (pool) {
            common_host_pool_set_budget(COMMON_HOST_POOL_CACHE, limit_size);
        }

        SRV_WRN(" - cache size limit reduced to %.3f MiB\n", limit_size / (1024.0 * 1024.0));

        update();

        return nullptr;
    }

    states.push_back({
        /*.prompt =*/ {
            /*.tokens      =*/ prompt.tokens.clone(),
            /*.checkpoints =*/ prompt.checkpoints,
            /*.t_created   =*/ t_created,
            /*.n_hits      =*/ n_hits,
        },
        /*.data   =*/ {
            /*.main     =*/ std::move(state_data_tgt),
            /*.drft     =*/ std::move(state_data_dft),
            /*.overflow =*/ overflow,
        },
        /*.t_last_used =*/ t_now,
    });

    if (pool && common_host_pool_get_stats().n_release_sync != n_release_sync0) {
        over = true; // idle regions were released to fit
    }

    if (over) {
        n_ovf_saves++;
        if (overflow) {
            n_ovf_heap++;
            bytes_ovf_heap += states.back().data.size();
        }
        log_overflow(overflow ? "saved to heap" : "saved", states.back().size());
    }

    return &states.back();
}

int64_t server_prompt_cache::usable(const server_prompt & p, int lcp) const {
    if (lcp <= 0 || !ckpt_limited || lcp >= p.n_tokens()) {
        return lcp;
    }
    // the newest checkpoint update_slots() would restore for this prefix
    const llama_pos pos_next = p.tokens.pos_next(lcp);
    int64_t best = 0;
    for (const auto & ck : p.checkpoints) {
        if (ck->n_tokens <= lcp && ck->pos_max <= pos_next && ck->n_tokens > best) {
            best = ck->n_tokens;
        }
    }
    return best;
}

std::list<server_prompt_cache_state>::const_iterator server_prompt_cache::find_prefix_copy(const server_tokens & tokens_new, int64_t n_base, int64_t & n_usable) const {
    auto res = states.cend();
    n_usable = 0;
    if (!prefix_ckpt || !ckpt_limited) {
        return res;
    }
    for (auto it = states.cbegin(); it != states.cend(); ++it) {
        const int lcp_cur = it->prompt.tokens.get_common_prefix(tokens_new);
        if (lcp_cur >= it->prompt.n_tokens()) {
            continue; // the whole entry is a prefix: the f_keep/f_sim selection handles it
        }
        const llama_pos pos_next = it->prompt.tokens.pos_next(lcp_cur);
        for (const auto & ck : it->prompt.checkpoints) {
            if (ck->pin != common_prompt_checkpoint::PIN_PREFIX || ck->n_tokens > lcp_cur || ck->pos_max > pos_next) {
                continue;
            }
            if (ck->n_tokens >= prefix_ckpt_min && ck->n_tokens > n_base && ck->n_tokens > n_usable) {
                n_usable = ck->n_tokens;
                res = it;
            }
        }
    }
    return res;
}

const server_prompt * server_prompt_cache::peek(const server_prompt & prompt, const server_tokens & tokens_new) const {
    // same selection as load()
    const int lcp_base = prompt.tokens.get_common_prefix(tokens_new);

    float f_keep_best = prompt.tokens.size() > 0 ? float(lcp_base) / prompt.tokens.size() : -1.0f;
    float f_sim_best  = float(lcp_base) / tokens_new.size();

    // p7: an entry that resumes fewer tokens than the slot's own prompt is not better
    const int64_t n_base = prefix_ckpt ? usable(prompt, lcp_base) : 0;

    const server_prompt * res = nullptr;
    for (const auto & state : states) {
        const int lcp_cur = state.prompt.tokens.get_common_prefix(tokens_new);

        const float f_keep_cur = float(lcp_cur) / state.prompt.tokens.size();
        const float f_sim_cur  = float(lcp_cur) / tokens_new.size();

        if (f_keep_cur < 0.25f) {
            continue;
        }
        if (prefix_ckpt && usable(state.prompt, lcp_cur) < n_base) {
            continue;
        }
        if (f_keep_best < f_keep_cur && f_sim_best < f_sim_cur) {
            f_keep_best = f_keep_cur;
            f_sim_best  = f_sim_cur;
            res = &state.prompt;
        }
    }
    if (res == nullptr) {
        int64_t n_copy = 0;
        const auto it = find_prefix_copy(tokens_new, n_base, n_copy);
        if (it != states.cend()) {
            res = &it->prompt;
        }
    }
    return res;
}

bool server_prompt_cache::load(server_prompt & prompt, const server_tokens & tokens_new, llama_context * ctx_tgt, llama_context * ctx_dft, int32_t id_slot) {
    int lcp_best = prompt.tokens.get_common_prefix(tokens_new);

    float f_keep_best = prompt.tokens.size() > 0 ? float(lcp_best) / prompt.tokens.size() : -1.0f; // empty slot: any cache entry wins
    float f_sim_best  = float(lcp_best) / tokens_new.size();

    SRV_TRC(" - looking for better prompt, base f_keep = %.3f, f_sim = %.3f\n", f_keep_best, f_sim_best);

    // p7: an entry that resumes fewer tokens than the slot's own prompt is not better
    const int64_t n_base = prefix_ckpt ? usable(prompt, lcp_best) : 0;

    auto it_best = states.end();

    // the most similar cached prompt regardless of the slot's own prompt (for the recency refresh)
    auto  it_match    = states.end();
    float f_sim_match = 0.0f;

    // find the most similar cached prompt, that would also preserve the most context
    for (auto it = states.begin(); it != states.end(); ++it) {
        const int lcp_cur = it->prompt.tokens.get_common_prefix(tokens_new);

        const float f_keep_cur = float(lcp_cur) / it->prompt.tokens.size();
        const float f_sim_cur  = float(lcp_cur) / tokens_new.size();

        SRV_TRC("   - prompt with length %7zu, lcp = %7d, f_keep = %.3f, f_sim = %.3f\n", it->prompt.tokens.size(), lcp_cur, f_keep_cur, f_sim_cur);

        // don't trash large prompts
        if (f_keep_cur < 0.25f) {
            continue;
        }

        if (lcp_cur > 0 && f_sim_match < f_sim_cur) {
            f_sim_match = f_sim_cur;
            it_match    = it;
        }

        if (prefix_ckpt && usable(it->prompt, lcp_cur) < n_base) {
            SRV_TRC("   - skipped, resumes fewer tokens than the slot's own prompt (%" PRId64 ")\n", n_base);
            continue;
        }

        if (f_keep_best < f_keep_cur && f_sim_best < f_sim_cur) {
            f_keep_best = f_keep_cur;
            f_sim_best  = f_sim_cur;
            lcp_best    = lcp_cur;

            it_best = it;
        }
    }

    // p7: no entry beats the slot's own prompt, but an entry's pinned prefix checkpoint resumes further. It is
    // copied, not moved: the entry is usually a long conversation (f_keep < 0.25) that must stay cached, and
    // update_slots() then cuts the slot's copy back to the checkpoint. The slot shares the entry's checkpoints,
    // which stay counted by the cache while the entry holds them
    if (it_best == states.end()) {
        int64_t n_copy = 0;
        const auto it_c = find_prefix_copy(tokens_new, n_base, n_copy);
        if (it_c != states.cend()) {
            auto it = states.erase(it_c, it_c); // non-const iterator to the same entry

            SRV_INF(" - copying prompt with length %d for its pinned prefix checkpoint (%" PRId64 " tokens, own prompt %" PRId64 ")\n",
                    it->prompt.n_tokens(), n_copy, n_base);

            bool ok = true;
            {
                const auto & data = it->data.main;
                const size_t n = llama_state_seq_set_data_ext(ctx_tgt, data.data(), data.size(), id_slot, 0);
                if (n != data.size()) {
                    SRV_ERR("failed to restore state with size %zu\n", data.size());
                    ok = false;
                }
            }
            if (ok && !it->data.drft.empty()) {
                GGML_ASSERT(ctx_dft);
                const auto & data = it->data.drft;
                const size_t n = llama_state_seq_set_data_ext(ctx_dft, data.data(), data.size(), id_slot, 0);
                if (n != data.size()) {
                    SRV_WRN("failed to restore state with size %zu\n", data.size());
                    ok = false;
                }
            }
            if (!ok) {
                metrics.n_misses++;
                return false; // the caller clears the slot
            }

            prompt = it->prompt.clone();
            // a different conversation from here on: do not carry the entry's history over
            prompt.t_created = 0;
            prompt.n_hits    = 0;

            it->t_last_used = ggml_time_us();

            metrics.n_hits++;
            n_prefix_copies++;

            return true;
        }
    }

    if (it_best == states.end()) {
        metrics.n_misses++;

        if (it_match != states.end()) {
            // the slot keeps a better prompt, but the entry matched: it is still in use
            metrics.n_matched++;
            it_match->t_last_used = ggml_time_us();
        }

        // the slot's prompt is about to be replaced by an unrelated one: do not carry its history over
        if (f_keep_best < 0.5f) {
            prompt.t_created = 0;
            prompt.n_hits    = 0;
        }
    }

    if (it_best != states.end()) {
        SRV_TRC(" - found better prompt with f_keep = %.3f, f_sim = %.3f\n", f_keep_best, f_sim_best);

        // a failed restore below is a miss (the caller clears the slot)
        metrics.n_misses++;

        {
            auto & data = it_best->data.main;

            const size_t size = data.size();
            const size_t n = llama_state_seq_set_data_ext(ctx_tgt, data.data(), size, id_slot, 0);
            if (n != size) {
                SRV_ERR("failed to restore state with size %zu\n", size);

                return false;
            }

            data.clear();
            data.shrink_to_fit();
        }

        {
            auto & data = it_best->data.drft;

            if (!data.empty()) {
                GGML_ASSERT(ctx_dft);

                const size_t size = data.size();
                const size_t n = llama_state_seq_set_data_ext(ctx_dft, data.data(), size, id_slot, 0);
                if (n != size) {
                    SRV_WRN("failed to restore state with size %zu\n", size);

                    return false;
                }

                data.clear();
                data.shrink_to_fit();
            }
        }

        SRV_DBG("%s", "__TEST_TAG_PROMPT_CACHE_RESTORE__\n");

        // the whole cached prompt is a prefix of the new one
        if (lcp_best == (int) it_best->prompt.tokens.size()) {
            SRV_DBG("%s", "__TEST_TAG_PROMPT_CACHE_FULL_RESTORE__\n");
        }

        prompt = std::move(it_best->prompt);
        prompt.n_hits++;

        metrics.n_misses--;
        metrics.n_hits++;

        states.erase(it_best);

        // the checkpoints are held by the slot now
        for (const auto & ckpt : prompt.checkpoints) {
            common_prompt_checkpoint_set_owner(*ckpt, COMMON_HOST_POOL_SLOT);
        }
    }

    return true;
}

void server_prompt_cache::update() {
    if (limit_size > 0) {
        // overflow entries are bounded by limit_overflow, everything else by limit_size
        while (!states.empty() && (size() - size_overflow() > limit_size || size_overflow() > limit_overflow)) {
            const auto it = victim();
            SRV_WRN(" - cache size limit reached, removing oldest entry (size = %.3f MiB)\n", it->size() / (1024.0 * 1024.0));

            evict(it, SERVER_PROMPT_CACHE_EVICT_BUDGET);
        }
    }

    // average size per token
    const float size_per_token = std::max<float>(1.0f, float(size()) / (std::max<size_t>(1, n_tokens())));

    // dynamically increase the token limit if it can fit in the memory limit
    const size_t limit_tokens_cur = limit_size > 0 ? std::max<size_t>(limit_tokens, limit_size/size_per_token) : limit_tokens;

    if (limit_tokens > 0) {
        while (!states.empty() && n_tokens() > limit_tokens_cur) {
            const auto it = victim();
            SRV_WRN(" - cache token limit (%zu, est: %zu) reached, removing oldest entry (size = %.3f MiB)\n",
                    limit_tokens, limit_tokens_cur, it->size() / (1024.0 * 1024.0));

            evict(it, SERVER_PROMPT_CACHE_EVICT_TOKENS);
        }
    }

    SRV_TRC(" - cache state: %zu prompts, %.3f MiB (limits: %.3f MiB, %zu tokens, %zu est)\n",
            states.size(), size() / (1024.0 * 1024.0), limit_size / (1024.0 * 1024.0), limit_tokens, limit_tokens_cur);

    for (const auto & state : states) {
        SRV_TRC("   - prompt %p: %7d tokens, checkpoints: %2zu, %9.3f MiB\n",
                (const void *)&state, state.prompt.n_tokens(), state.prompt.checkpoints.size(), state.size() / (1024.0 * 1024.0));
    }
}

//
// server_slot_sched
//

static std::vector<const server_slot_info *> get_free_slot_infos(const std::vector<server_slot_info> & slots, size_t n_slots_needed, int exclude_id_slot) {
    std::vector<const server_slot_info *> free_slots;
    for (const auto & slot : slots) {
        if (!slot.is_processing && slot.id != exclude_id_slot) {
            free_slots.push_back(&slot);
        }
    }

    std::stable_sort(free_slots.begin(), free_slots.end(), [](const server_slot_info * a, const server_slot_info * b) {
        if ((a->prompt->n_tokens() == 0) != (b->prompt->n_tokens() == 0)) {
            return a->prompt->n_tokens() == 0;
        }
        return a->t_last_used < b->t_last_used;
    });

    free_slots.resize(std::min(free_slots.size(), n_slots_needed));

    return free_slots;
}

std::vector<int> server_slot_sched::get_free_slots(const std::vector<server_slot_info> & slots, size_t n_slots_needed, int exclude_id_slot) const {
    std::vector<int> res;
    for (const auto * slot : get_free_slot_infos(slots, n_slots_needed, exclude_id_slot)) {
        res.push_back(slot->id);
    }

    return res;
}

server_slot_plan server_slot_sched::select(const std::vector<server_slot_info> & slots, const server_task & task) const {
    const server_slot_info * ret = nullptr;

    bool update_cache = false;

    // if a specific slot is requested, use it (still goes through cache update logic below)
    if (task.id_slot != -1 && !slots.empty()) {
        // note: allow id_slot to be out of bounds (wrap around)
        const int id_slot = task.id_slot % slots.size();

        for (const auto & slot : slots) {
            if (slot.id == id_slot) {
                ret = &slot;
                break;
            }
        }

        if (ret) {
            // a busy slot is returned untouched, the caller defers the task
            if (ret->is_processing) {
                return { ret->id, false };
            }

            SLT_INF(*ret, "selected slot by id (%d)\n", task.id_slot);
        }
    }

    // find the slot that has at least n% prompt similarity
    if (slot_prompt_similarity != 0.0f) {
        float f_sim_best = 0;

        for (const auto & slot : slots) {
            if (task.id_slot != -1 && slot.id != task.id_slot) {
                continue;
            }

            // skip the slot if it is not available
            if (slot.is_processing) {
                SLT_TRC(slot, " - skipping, is_processing = %d\n", slot.is_processing);
                continue;
            }

            const auto & tokens = slot.prompt->tokens;

            // skip the slot if it does not contains cached tokens
            if (tokens.empty()) {
                SLT_TRC(slot, "%s", " - skipping, slot is empty\n");
                continue;
            }

            // fraction of the Longest Common Prefix length with respect to the input prompt length
            const size_t lcp_len = tokens.get_common_prefix(task.tokens);
            const float f_sim_cur = float(lcp_len) / task.tokens.size();

            SLT_TRC(slot, " - checking sim = %.3f (%zu/%zu) > %.3f\n", f_sim_cur, lcp_len, task.tokens.size(), slot_prompt_similarity);

            // select the current slot if the criteria match
            if (f_sim_cur > f_sim_best && f_sim_cur > slot_prompt_similarity) {
                f_sim_best = f_sim_cur;

                ret = &slot;
            }
        }

        if (ret != nullptr) {
            const float f_keep = (f_sim_best*task.tokens.size()) / ret->prompt->tokens.size();

            if (task.id_slot == -1) {
                SLT_INF(*ret, "selected slot by LCP similarity, f_sim_best = %.3f (> %.3f thold), f_keep = %.3f\n",
                        f_sim_best, slot_prompt_similarity, f_keep);
            }

            // if we are about to lose a large portion of the existing context - save it in the prompt cache
            if (f_keep < 0.5f) {
                update_cache = true;
            }
        }
    }

    // find an empty slot, or the one that has been least recently used
    if (ret == nullptr) {
        const auto free_slots = get_free_slot_infos(slots, 1, -1);

        if (!free_slots.empty()) {
            ret = free_slots[0];

            SLT_INF(*ret, "selected slot by LRU, n_tokens = %d, t_last = %" PRId64 "\n", ret->prompt->n_tokens(), ret->t_last_used);

            update_cache = true;
        }
    }

    if (ret == nullptr) {
        return {};
    }

    // cache prompts only for the tasks that reuse them
    update_cache = update_cache && !task.is_stateless();

    return { ret->id, update_cache };
}
