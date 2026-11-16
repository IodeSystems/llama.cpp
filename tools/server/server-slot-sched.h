#pragma once

#include "server-common.h"
#include "server-task.h"

#include <list>
#include <memory>
#include <vector>

// which prompt goes to which slot: slot prompt state, prompt cache and slot selection

// what a cached prompt can give to a new prompt
struct server_prompt_reuse {
    int32_t   n_past   = 0; // tokens to keep
    llama_pos pos_next = 0;

    // restore this checkpoint first, nullptr when the live state is used as it is
    std::shared_ptr<const common_prompt_checkpoint> ckpt;
};

struct server_prompt {
    server_tokens tokens;

    // a checkpoint does not change after it is created, so the prompt cache shares it with the slot instead of a copy
    std::list<std::shared_ptr<const common_prompt_checkpoint>> checkpoints;

    // prompt cache bookkeeping, carried from a cache entry into the slot and back on the next save
    int64_t  t_created = 0; // us, first save of this prompt to the cache (0 = never saved)
    uint32_t n_hits    = 0; // times the cache restored this prompt into a slot

    void clear() {
        tokens.clear();
        checkpoints.clear();
        t_created = 0;
        n_hits    = 0;
    }

    int n_tokens() const {
        return tokens.size();
    }

    server_prompt clone() const {
        return server_prompt {
            tokens.clone(),
            checkpoints,
            t_created,
            n_hits,
        };
    }

    // n_prefix: tokens shared with the new prompt, must be > 0
    // n_unshifted: tokens shared with the new prompt before cache reuse shifted any chunk (<= n_prefix)
    // has_new_tokens: the new prompt is longer than n_prefix
    // pos_min: min position of the live state
    // n_past == 0 without a checkpoint means that nothing can be reused
    server_prompt_reuse get_reuse(int32_t n_prefix, int32_t n_unshifted, bool has_new_tokens, int32_t n_swa, llama_pos pos_min) const;
};

struct server_prompt_data {
    common_state_data main;
    common_state_data drft;

    bool overflow = false; // allocated as --cache-overflow heap

    // real allocated bytes
    size_t size() const {
        return common_state_data_bytes(main) + common_state_data_bytes(drft);
    }
};

struct server_prompt_cache_state {
    server_prompt prompt;
    server_prompt_data data;

    int64_t t_last_used = 0; // us, last save, load or match of this entry

    size_t size() const {
        size_t res = data.size();

        for (const auto & ckpt : prompt.checkpoints) {
            res += ckpt->real_size();
        }

        return res;
    }
};

struct server_prompt_cache {
    server_prompt_cache(int32_t limit_size_mib, size_t limit_tokens, common_cache_ram_policy policy = COMMON_CACHE_RAM_POLICY_FIFO) {
        this->policy       = policy;
        this->limit_size   = 1024ull*1024ull*(limit_size_mib < 0 ? 0 : limit_size_mib);
        this->limit_tokens = limit_tokens;
    }

    std::list<server_prompt_cache_state> states;

    // in bytes, 0 = no limit
    size_t limit_size = 0;

    // in tokens, 0 = no limit
    size_t limit_tokens = 0;

    // which entry to evict first
    common_cache_ram_policy policy = COMMON_CACHE_RAM_POLICY_FIFO;

    // counters for /metrics (gauges are filled by get_metrics())
    server_prompt_cache_metrics metrics;

    // --cache-overflow heap: entries beyond limit_size go to plain heap, up to limit_overflow bytes
    bool   overflow_heap  = false;
    size_t limit_overflow = 0;

    // throttled overflow log
    int64_t  t_ovf_log       = 0;
    uint64_t n_ovf_saves     = 0; // saves that did not fit, since the last log
    uint64_t n_ovf_evicted   = 0;
    size_t   bytes_ovf_evicted = 0;
    uint64_t n_ovf_heap      = 0;
    size_t   bytes_ovf_heap  = 0;
    uint64_t n_ovf_rel_sync0 = 0; // host pool budget releases at the last log
    size_t   bytes_ovf_rel_sync0 = 0;
    uint64_t n_ovf_frag0     = 0; // host pool arena fragmentation count at the last log

    // real allocated bytes (a checkpoint shared by several entries is counted once)
    size_t size() const;
    size_t size_overflow() const;

    // erase an entry; checkpoints still referenced elsewhere go back to slot accounting
    std::list<server_prompt_cache_state>::iterator erase_state(std::list<server_prompt_cache_state>::iterator it);

    void log_overflow(const char * what, size_t bytes);

    // the entry to evict next under the policy (states must not be empty)
    std::list<server_prompt_cache_state>::iterator victim();

    // host pool fragmentation: the first entry in policy order whose eviction leaves a free extent of
    // `size` bytes, or victim() if none does
    std::list<server_prompt_cache_state>::iterator victim_hole(size_t size);

    // blocks only the cache holds: entry state buffers and checkpoints no slot shares
    std::vector<const void *> movable_blocks() const;

    // erase an entry to make room (or a superseded one) and count it; returns the next entry
    std::list<server_prompt_cache_state>::iterator evict(std::list<server_prompt_cache_state>::iterator it, server_prompt_cache_evict_reason reason);

    server_prompt_cache_metrics get_metrics() const;

    size_t n_tokens() const;

    server_prompt_cache_state * alloc(const server_prompt & prompt, size_t state_size_main, size_t state_size_drft);

    bool load(server_prompt & prompt, const server_tokens & tokens_new, llama_context * ctx_tgt, llama_context * ctx_dft, int32_t id_slot);

    void update();
};

// what the scheduler needs to know about a slot
struct server_slot_info {
    int id = -1;
    const server_task * task = nullptr; // only for logging

    bool is_processing = false;
    int64_t t_last_used = -1;

    const server_prompt * prompt = nullptr;
};

struct server_slot_plan {
    // -1 if no slot is available; a busy slot if the task asks for it by id, the caller defers the task
    int id_slot = -1;

    // save the slot state to the prompt cache, then load the best match for the task
    bool update_cache = false;
};

// decides only, the caller applies the plan
struct server_slot_sched {
    float slot_prompt_similarity = 0.0f;

    // idle slots that lose the least by being reused: empty slots first, then the least recently used ones
    std::vector<int> get_free_slots(const std::vector<server_slot_info> & slots, size_t n_slots_needed, int exclude_id_slot) const;

    server_slot_plan select(const std::vector<server_slot_info> & slots, const server_task & task) const;
};
