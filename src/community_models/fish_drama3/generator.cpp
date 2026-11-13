#include "engine/community_models/fish_drama3/generator.h"

#include "engine/framework/debug/profiler.h"
#include "engine/framework/runtime/errors.h"
#include "engine/framework/runtime/options.h"
#include "engine/framework/sampling/hf_sampler.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <mutex>
#include <optional>
#include <random>
#include <stdexcept>
#include <string_view>
#include <thread>

namespace engine::models::fish_drama3 {
namespace {

// Free device memory at a phase boundary. Drama 3 allocates its phase graphs in
// spikes, so the peak is only attributable when each phase reports what it left.
void log_free_vram(core::ExecutionContext & execution, const char * name) {
    const auto snapshot = execution.memory_snapshot();
    if (!snapshot.available) return;
    debug::timing_log_scalar(std::string("drama3.vram.free_") + name + "_mb",
        static_cast<double>(snapshot.free_bytes) / (1024.0 * 1024.0));
}

struct PerfPreset {
    std::string_view name;
    bool fused_moe;
    ggml_type kv_cache_type;
    int64_t kv_cache_alignment;
    bool device_sampling;
};

constexpr std::array<PerfPreset, 3> kPerfPresets{{
    {"off", false, GGML_TYPE_F16, 1, false},
    {"cuda_perf", true, GGML_TYPE_F16, 256, true},
    // Same arithmetic as cuda_perf. HIP runs the same ggml-cuda kernels, and
    // the fused tower graph removes thirteen per-frame host round trips there
    // for the same reason it does on CUDA.
    {"hip_perf", true, GGML_TYPE_F16, 256, true},
}};

// HIP decodes one frame at a time through the same ggml-cuda kernels CUDA uses,
// so it starts from the fused preset. Every field is still overridable.
std::string_view default_perf_mode(core::BackendType backend_type) {
    return backend_type == core::BackendType::Hip ? "hip_perf" : "off";
}

struct SamplerScratch {
    std::vector<int32_t> order;
    std::vector<double> weights;
    std::vector<double> probabilities;
};

int32_t sample(const std::vector<float> & logits, float temperature, int top_k,
               float top_p, std::mt19937 & random, SamplerScratch & scratch) {
    if (logits.empty()) throw std::runtime_error("Drama 3 sampler has no logits");
    if (temperature == 0 || top_k == 1)
        return sampling::HfLogitsProcessor::argmax(logits.data(), logits.size(), "Drama 3");
    auto & order = scratch.order;
    order.resize(logits.size());
    std::iota(order.begin(), order.end(), 0);
    const auto count = std::min(logits.size(), static_cast<size_t>(top_k));
    const auto compare = [&](int32_t left, int32_t right) {
        return logits[left] == logits[right] ? left < right : logits[left] > logits[right];
    };
    const double maximum = *std::max_element(logits.begin(), logits.end()) / temperature;
    double total = 0;
    // Top-p is defined against the full distribution, before top-k filtering.
    auto & weights = scratch.weights;
    weights.resize(logits.size());
    for (size_t index = 0; index < logits.size(); ++index) {
        weights[index] = std::exp(logits[index] / temperature - maximum);
        total += weights[index];
    }
    if (!(total > 0) || !std::isfinite(total)) throw std::runtime_error("Drama 3 sampler encountered nonfinite logits");
    size_t sorted = std::min(count, size_t{64});
    for (;;) {
        std::partial_sort(order.begin(), order.begin() + sorted, order.end(), compare);
        double mass = 0;
        for (size_t rank = 0; rank < sorted; ++rank) mass += weights[order[rank]] / total;
        if (sorted == count || mass > top_p) break;
        sorted = std::min(count, sorted * 2);
    }
    auto & probabilities = scratch.probabilities;
    probabilities.assign(count, 0);
    double cumulative = 0;
    for (size_t rank = 0; rank < sorted; ++rank) {
        cumulative += weights[order[rank]] / total;
        if (rank > 0 && cumulative > top_p) break;
        probabilities[rank] = weights[order[rank]];
    }
    std::discrete_distribution<size_t> choice(probabilities.begin(), probabilities.end());
    return order[choice(random)];
}

std::vector<float> last_hidden(const std::vector<float> & values, int64_t width) {
    if (values.size() < static_cast<size_t>(width)) throw std::runtime_error("Drama 3 backbone returned too few hidden values");
    return {values.end() - width, values.end()};
}

// ---------------------------------------------------------------------------
// Reference cache
//
// A session pays the voice reference once: ~1.3 s to encode it, then ~4 s to
// evaluate its ~1250 prompt rows into backbone and tower state. Both are pure
// functions of the reference and the package, so they are persisted and a new
// process starts as if the prefix had already been evaluated.
//
// The payload is a few tens of megabytes of state, so entries are validated on
// read against the package identity, the numerics options and the prompt
// frames; anything that does not match is ignored and recomputed.
// ---------------------------------------------------------------------------

constexpr uint32_t kReferenceCacheMagic = 0x31524433u;  // "D3R1"
constexpr uint32_t kReferenceCodesMagic = 0x31434433u;  // "D3C1"
constexpr uint32_t kReferenceCacheVersion = 1;
constexpr size_t kReferenceFrameValues = 11;

struct Drama3ReferenceCacheEntry {
    std::vector<Drama3Frame> frames;
    std::vector<float> hidden;
    Drama3BackboneState backbone;
    Drama3TowerState tower;
};

uint64_t fnv1a64(const void * data, size_t size, uint64_t hash = 1469598103934665603ull) {
    const auto * bytes = static_cast<const uint8_t *>(data);
    for (size_t i = 0; i < size; ++i) hash = (hash ^ bytes[i]) * 1099511628211ull;
    return hash;
}

std::string hex64(uint64_t value) {
    static constexpr char digits[] = "0123456789abcdef";
    std::string out(16, '0');
    for (int i = 15; i >= 0; --i) {
        out[static_cast<size_t>(i)] = digits[value & 0xFu];
        value >>= 4;
    }
    return out;
}

// `${fish_drama3.reference_cache_dir}`, else `${XDG_CACHE_HOME:-~/.cache}/audio.cpp`.
std::filesystem::path reference_cache_root(const runtime::SessionOptions & options) {
    if (const auto value = runtime::find_option(options.options, {"fish_drama3.reference_cache_dir"})) {
        if (value->empty()) throw std::runtime_error("fish_drama3.reference_cache_dir must not be empty");
        return std::filesystem::path(*value);
    }
    std::filesystem::path base;
    if (const char * xdg = std::getenv("XDG_CACHE_HOME"); xdg && *xdg) base = xdg;
    else if (const char * home = std::getenv("HOME"); home && *home) base = std::filesystem::path(home) / ".cache";
    else throw std::runtime_error("HOME is required for the Drama 3 reference cache");
    return base / "audio.cpp" / "drama3-reference";
}

// Everything about the run that changes the stored state. The package identity
// covers the weights; these cover the shapes and the kernels that produced the
// floats.
std::string reference_cache_config(const Drama3BackboneDecoderConfig & config, int64_t context_length,
                                   bool device_sampling) {
    std::string text = "drama3-reference-v1";
    for (const auto value : {config.hidden_size, config.layers, config.heads, config.kv_heads, config.head_dim,
             config.full_attention_interval, config.experts_per_token, config.linear_conv_kernel_dim,
             config.linear_num_key_heads, config.linear_num_value_heads, config.linear_value_head_dim,
             config.decode_cache_steps, context_length, static_cast<int64_t>(config.kv_cache_type),
             config.kv_cache_alignment, static_cast<int64_t>(config.fused_moe),
             static_cast<int64_t>(config.round_bf16_activations), static_cast<int64_t>(device_sampling)}) {
        text += ":" + std::to_string(value);
    }
    return text;
}

std::string reference_cache_key(const std::string & identity, const std::string & config,
                                const std::vector<Drama3Frame> & frames) {
    uint64_t hash = fnv1a64(identity.data(), identity.size());
    hash = fnv1a64(config.data(), config.size(), hash);
    hash = fnv1a64(frames.data(), frames.size() * sizeof(Drama3Frame), hash);
    return hex64(hash);
}

void put_u32(std::vector<uint8_t> & out, uint32_t value) {
    for (int shift = 0; shift < 32; shift += 8) out.push_back(static_cast<uint8_t>(value >> shift));
}

void put_bytes(std::vector<uint8_t> & out, const void * data, size_t size) {
    out.insert(out.end(), static_cast<const uint8_t *>(data), static_cast<const uint8_t *>(data) + size);
}

void put_blob(std::vector<uint8_t> & out, const void * data, size_t size) {
    put_u32(out, static_cast<uint32_t>(size));
    put_bytes(out, data, size);
}

struct BlobReader {
    const uint8_t * data = nullptr;
    size_t size = 0;
    size_t offset = 0;

    bool u32(uint32_t & value) {
        if (offset + 4 > size) return false;
        value = static_cast<uint32_t>(data[offset]) | static_cast<uint32_t>(data[offset + 1]) << 8 |
                static_cast<uint32_t>(data[offset + 2]) << 16 | static_cast<uint32_t>(data[offset + 3]) << 24;
        offset += 4;
        return true;
    }
    bool bytes(void * out, size_t count) {
        if (offset + count > size) return false;
        std::memcpy(out, data + offset, count);
        offset += count;
        return true;
    }
    bool blob(std::vector<uint8_t> & out, size_t expected) {
        uint32_t length = 0;
        if (!u32(length) || (expected != 0 && length != expected) || offset + length > size) return false;
        out.assign(data + offset, data + offset + length);
        offset += length;
        return true;
    }
};

bool write_reference_cache(const std::filesystem::path & path, const std::string & identity,
                           const std::string & config, const Drama3ReferenceCacheEntry & entry) {
    std::vector<uint8_t> payload;
    put_blob(payload, identity.data(), identity.size());
    put_blob(payload, config.data(), config.size());
    put_u32(payload, static_cast<uint32_t>(entry.frames.size()));
    put_bytes(payload, entry.frames.data(), entry.frames.size() * sizeof(Drama3Frame));
    put_u32(payload, static_cast<uint32_t>(entry.hidden.size()));
    put_bytes(payload, entry.hidden.data(), entry.hidden.size() * sizeof(float));
    put_u32(payload, static_cast<uint32_t>(entry.backbone.steps));
    put_blob(payload, entry.backbone.bytes.data(), entry.backbone.bytes.size());
    put_blob(payload, entry.tower.bytes.data(), entry.tower.bytes.size());

    std::vector<uint8_t> file;
    put_u32(file, kReferenceCacheMagic);
    put_u32(file, kReferenceCacheVersion);
    put_u32(file, static_cast<uint32_t>(payload.size()));
    const uint64_t checksum = fnv1a64(payload.data(), payload.size());
    for (int shift = 0; shift < 64; shift += 8) file.push_back(static_cast<uint8_t>(checksum >> shift));
    file.insert(file.end(), payload.begin(), payload.end());

    std::error_code error;
    std::filesystem::create_directories(path.parent_path(), error);
    if (error) return false;
    const auto temp = path.string() + ".tmp" +
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
    {
        std::ofstream out(temp, std::ios::binary | std::ios::trunc);
        if (!out) return false;
        out.write(reinterpret_cast<const char *>(file.data()), static_cast<std::streamsize>(file.size()));
        if (!out) return false;
    }
    std::filesystem::rename(temp, path, error);
    if (error) {
        std::filesystem::remove(temp, error);
        return false;
    }
    return true;
}

// The reference codes are what the prompt is built from, so they have to be
// available before the state key can even be formed. They are keyed by the
// audio itself plus the encoder options that shape them.
std::string reference_codes_key(const std::string & identity, const std::string & config,
                                const runtime::AudioBuffer & audio) {
    uint64_t hash = fnv1a64(identity.data(), identity.size());
    hash = fnv1a64(config.data(), config.size(), hash);
    const int64_t header[3] = {audio.sample_rate, audio.channels, static_cast<int64_t>(audio.samples.size())};
    hash = fnv1a64(header, sizeof(header), hash);
    hash = fnv1a64(audio.samples.data(), audio.samples.size() * sizeof(float), hash);
    return hex64(hash);
}

bool write_reference_codes(const std::filesystem::path & path, const codecs::FishDacCodes & codes) {
    std::vector<uint8_t> payload;
    put_u32(payload, static_cast<uint32_t>(codes.codebooks));
    put_u32(payload, static_cast<uint32_t>(codes.frames));
    put_bytes(payload, codes.codes.data(), codes.codes.size() * sizeof(int32_t));

    std::vector<uint8_t> file;
    put_u32(file, kReferenceCodesMagic);
    put_u32(file, kReferenceCacheVersion);
    put_u32(file, static_cast<uint32_t>(payload.size()));
    const uint64_t checksum = fnv1a64(payload.data(), payload.size());
    for (int shift = 0; shift < 64; shift += 8) file.push_back(static_cast<uint8_t>(checksum >> shift));
    file.insert(file.end(), payload.begin(), payload.end());

    std::error_code error;
    std::filesystem::create_directories(path.parent_path(), error);
    if (error) return false;
    const auto temp = path.string() + ".tmp" +
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
    {
        std::ofstream out(temp, std::ios::binary | std::ios::trunc);
        if (!out) return false;
        out.write(reinterpret_cast<const char *>(file.data()), static_cast<std::streamsize>(file.size()));
        if (!out) return false;
    }
    std::filesystem::rename(temp, path, error);
    if (error) {
        std::filesystem::remove(temp, error);
        return false;
    }
    return true;
}

std::optional<codecs::FishDacCodes> read_reference_codes(const std::filesystem::path & path, int64_t codebooks) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return std::nullopt;
    std::vector<uint8_t> file((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    if (file.size() < 20) return std::nullopt;
    BlobReader header{file.data(), file.size()};
    uint32_t magic = 0;
    uint32_t version = 0;
    uint32_t payload_size = 0;
    if (!header.u32(magic) || !header.u32(version) || !header.u32(payload_size)) return std::nullopt;
    if (magic != kReferenceCodesMagic || version != kReferenceCacheVersion) return std::nullopt;
    uint64_t checksum = 0;
    for (int i = 0; i < 8; ++i) checksum |= static_cast<uint64_t>(file[12 + static_cast<size_t>(i)]) << (8 * i);
    const size_t offset = 20;
    if (offset + payload_size != file.size() ||
        fnv1a64(file.data() + offset, payload_size) != checksum) return std::nullopt;
    BlobReader reader{file.data() + offset, payload_size};
    uint32_t stored_books = 0;
    uint32_t stored_frames = 0;
    if (!reader.u32(stored_books) || !reader.u32(stored_frames)) return std::nullopt;
    if (stored_books != static_cast<uint32_t>(codebooks) || stored_frames == 0 ||
        stored_frames > 1u << 20) return std::nullopt;
    codecs::FishDacCodes codes;
    codes.codebooks = stored_books;
    codes.frames = stored_frames;
    codes.codes.resize(static_cast<size_t>(stored_books) * stored_frames);
    if (!reader.bytes(codes.codes.data(), codes.codes.size() * sizeof(int32_t))) return std::nullopt;
    return codes;
}

std::optional<Drama3ReferenceCacheEntry> read_reference_cache(
    const std::filesystem::path & path, const std::string & identity, const std::string & config,
    size_t frames, int64_t hidden_size) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return std::nullopt;
    std::vector<uint8_t> file((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    if (file.size() < 20) return std::nullopt;
    BlobReader header{file.data(), file.size()};
    uint32_t magic = 0;
    uint32_t version = 0;
    uint32_t payload_size = 0;
    if (!header.u32(magic) || !header.u32(version) || !header.u32(payload_size)) return std::nullopt;
    if (magic != kReferenceCacheMagic || version != kReferenceCacheVersion) return std::nullopt;
    uint64_t checksum = 0;
    for (int i = 0; i < 8; ++i) checksum |= static_cast<uint64_t>(file[12 + static_cast<size_t>(i)]) << (8 * i);
    const size_t offset = 20;
    if (offset + payload_size != file.size() ||
        fnv1a64(file.data() + offset, payload_size) != checksum) return std::nullopt;

    BlobReader reader{file.data() + offset, payload_size};
    std::vector<uint8_t> stored_identity;
    std::vector<uint8_t> stored_config;
    if (!reader.blob(stored_identity, identity.size()) || !reader.blob(stored_config, config.size())) return std::nullopt;
    if (std::string(stored_identity.begin(), stored_identity.end()) != identity ||
        std::string(stored_config.begin(), stored_config.end()) != config) return std::nullopt;

    Drama3ReferenceCacheEntry entry;
    uint32_t stored_frames = 0;
    if (!reader.u32(stored_frames) || stored_frames != frames) return std::nullopt;
    entry.frames.resize(stored_frames);
    if (!reader.bytes(entry.frames.data(), stored_frames * sizeof(Drama3Frame))) return std::nullopt;
    uint32_t values = 0;
    if (!reader.u32(values) || values == 0 || values % static_cast<uint32_t>(hidden_size) != 0) return std::nullopt;
    entry.hidden.resize(values);
    if (!reader.bytes(entry.hidden.data(), values * sizeof(float))) return std::nullopt;
    uint32_t steps = 0;
    if (!reader.u32(steps) || steps != stored_frames) return std::nullopt;
    entry.backbone.steps = steps;
    if (!reader.blob(entry.backbone.bytes, 0) || !reader.blob(entry.tower.bytes, 0)) return std::nullopt;
    if (entry.backbone.bytes.empty() || entry.tower.bytes.empty()) return std::nullopt;
    return entry;
}

}  // namespace

Drama3Generator::Drama3Generator(std::shared_ptr<const Drama3Assets> assets,
    core::ExecutionContext & execution, const runtime::SessionOptions & options, bool streaming)
    : assets_(std::move(assets)), tokenizer_(assets_->resources),
      cache_steps_(runtime::parse_positive_i64_option(options.options, {"fish_drama3.context_length"}, 2048)) {
    execution_ = &execution;
    if (cache_steps_ > 262144) throw std::runtime_error("Drama 3 context_length exceeds the checkpoint's maximum position count");
    if (const auto value = runtime::find_option(options.options, {"fish_drama3.profile_first_audio"})) {
        profile_first_audio_ = runtime::parse_bool_option(*value, "fish_drama3.profile_first_audio");
    }
    // Retaining the reference-prefix state splits the first prompt evaluation
    // in two segments, which changes the sampled take for a fixed seed on the
    // same terms as any other fused or re-tiled graph. Set this to false to keep
    // the single-segment prompt evaluation.
    if (const auto value = runtime::find_option(options.options, {"fish_drama3.reference_prefix_cache"})) {
        reference_prefix_cache_ = runtime::parse_bool_option(*value, "fish_drama3.reference_prefix_cache");
    }
    codec_batch_frames_ = static_cast<size_t>(runtime::parse_positive_i64_option(
        options.options, {"fish_drama3.codec_batch_frames"}, 4));
    if (const auto value = runtime::find_option(options.options, {"fish_drama3.codec_async_decode"})) {
        codec_async_decode_ = runtime::parse_bool_option(*value, "fish_drama3.codec_async_decode");
    }
    auto backbone_config = assets_->backbone;
    backbone_config.decode_cache_steps = cache_steps_;
    const auto perf_mode = runtime::find_option(options.options, {"fish_drama3.perf_mode"}).value_or(
        std::string(default_perf_mode(execution.backend_type())));
    const auto preset = std::find_if(kPerfPresets.begin(), kPerfPresets.end(),
        [&](const auto & entry) { return entry.name == perf_mode; });
    if (preset == kPerfPresets.end()) throw std::runtime_error("Invalid fish_drama3.perf_mode: " + perf_mode);
    device_sampling_ = preset->device_sampling;
    if (const auto value = runtime::find_option(options.options, {"fish_drama3.device_sampling"})) {
        device_sampling_ = runtime::parse_bool_option(*value, "fish_drama3.device_sampling");
    }
    backbone_config.fused_moe = preset->fused_moe;
    backbone_config.kv_cache_type = preset->kv_cache_type;
    backbone_config.kv_cache_alignment = preset->kv_cache_alignment;
    backbone_config.kv_cache_alignment = runtime::parse_positive_i64_option(
        options.options, {"fish_drama3.kv_cache_alignment"}, backbone_config.kv_cache_alignment);
    if (const auto value = runtime::find_option(options.options, {"fish_drama3.fused_moe"}))
        backbone_config.fused_moe = runtime::parse_bool_option(*value, "fish_drama3.fused_moe");
    if (const auto value = runtime::find_option(options.options, {"fish_drama3.kv_cache_type"})) {
        if (*value == "f16") backbone_config.kv_cache_type = GGML_TYPE_F16;
        else if (*value == "f32") backbone_config.kv_cache_type = GGML_TYPE_F32;
        else throw std::runtime_error("fish_drama3.kv_cache_type requires f32 or f16");
    }
    debug::trace_log_scalar("fish_drama3.perf_mode", perf_mode);
    debug::trace_log_scalar("fish_drama3.device_sampling", device_sampling_);
    debug::trace_log_scalar("fish_drama3.fused_moe", backbone_config.fused_moe);
    debug::trace_log_scalar("fish_drama3.kv_cache_type", std::string_view(backbone_config.kv_cache_type == GGML_TYPE_F16 ? "f16" : "f32"));
    debug::trace_log_scalar("fish_drama3.kv_cache_alignment", backbone_config.kv_cache_alignment);
    backbone_config.cpu_expert_layers = runtime::parse_i64_option(options.options, {"fish_drama3.cpu_expert_layers"}).value_or(4);
    if (const auto value = runtime::find_option(options.options, {"fish_drama3.disk_experts"}))
        backbone_config.disk_experts = runtime::parse_bool_option(*value, "fish_drama3.disk_experts");
    if (const auto value = runtime::find_option(options.options, {"fish_drama3.pin_experts"}))
        backbone_config.pin_experts = runtime::parse_bool_option(*value, "fish_drama3.pin_experts");
    if (backbone_config.pin_experts && !backbone_config.disk_experts)
        throw std::runtime_error("fish_drama3.pin_experts requires fish_drama3.disk_experts=true");
    backbone_config.expert_cache_size = runtime::parse_positive_i64_option(
        options.options, {"fish_drama3.expert_cache_size"}, 32);
    backbone_config.disk_expert_read_threads = runtime::parse_positive_i64_option(
        options.options, {"fish_drama3.disk_expert_read_threads"}, 1);
    if (backbone_config.disk_expert_read_threads > 64)
        throw std::runtime_error("Drama 3 disk_expert_read_threads must be between 1 and 64");
    const auto backbone_prefill_cache_mb = runtime::parse_i64_option(options.options, {"fish_drama3.backbone_prefill_cache_mb"}).value_or(256);
    if (backbone_prefill_cache_mb < 0 || backbone_prefill_cache_mb > 4096) throw std::runtime_error("Drama 3 backbone_prefill_cache_mb must be between 0 and 4096");
    backbone_config.prefill_cache_bytes = static_cast<size_t>(backbone_prefill_cache_mb) * 1024u * 1024u;
    backbone_config.profile_prefill_cache = profile_first_audio_;
    if (const auto value = runtime::find_option(options.options, {"fish_drama3.profile_skip_routed_experts"}))
        backbone_config.profile_skip_routed_experts = runtime::parse_bool_option(*value, "fish_drama3.profile_skip_routed_experts");
    if (const auto value = runtime::find_option(options.options, {"fish_drama3.profile_skip_attention"}))
        backbone_config.profile_skip_attention = runtime::parse_bool_option(*value, "fish_drama3.profile_skip_attention");
    backbone_ = std::make_unique<Drama3BackboneDecoderRuntime>(assets_->model_weights, backbone_config, execution,
        64 * 1024 * 1024, 8 * 1024 * 1024, assets::TensorStorageType::Native);
    disk_experts_ = backbone_config.disk_experts;
    decoder_ = backbone_->create_decode_session(cache_steps_);
    const auto tower_prefill_cache_mb = runtime::parse_i64_option(options.options, {"fish_drama3.tower_prefill_cache_mb"}).value_or(256);
    if (tower_prefill_cache_mb < 0 || tower_prefill_cache_mb > 4096) throw std::runtime_error("Drama 3 tower_prefill_cache_mb must be between 0 and 4096");
    tower_ = std::make_unique<Drama3Tower>(assets_->model_weights, assets_->tower, execution, cache_steps_,
        static_cast<size_t>(tower_prefill_cache_mb) * 1024u * 1024u);
    auto codec_backend = options.backend;
    if (const auto value = runtime::find_option(options.options, {"fish_drama3.codec_backend"})) {
        if (*value == "cpu") codec_backend.type = core::BackendType::Cpu;
        else if (*value != "same") throw std::runtime_error("fish_drama3.codec_backend requires cpu or same");
    }
    codec_execution_ = std::make_unique<core::ExecutionContext>(codec_backend);
    codecs::FishDacCodecRuntimeOptions codec_options;
    codec_options.weight_context_bytes = 16 * 1024 * 1024;
    codec_options.decoder_prepack_weights = streaming && codec_batch_frames_ > 1;
    codec_options.encoder_bf16_compute = true;
    if (const auto value = runtime::find_option(options.options, {"fish_drama3.codec_encoder_bf16_compute"})) {
        codec_options.encoder_bf16_compute = runtime::parse_bool_option(*value, "fish_drama3.codec_encoder_bf16_compute");
    }
    const auto component = codecs::FishDacCodecComponent::load_from_tensor_source(assets_->codec_weights,
        assets_->codec, {}, codec_execution_->backend(), codec_backend.type, codec_options);
    bool codec_encoder_f32_accumulation = false;
    if (const auto value = runtime::find_option(options.options, {"fish_drama3.codec_encoder_f32_accumulation"})) {
        codec_encoder_f32_accumulation = runtime::parse_bool_option(*value, "fish_drama3.codec_encoder_f32_accumulation");
    }
    const int64_t codec_encoder_chunk_samples = runtime::parse_i64_option(
        options.options, {"fish_drama3.codec_encoder_chunk_samples"}).value_or(0);
    codec_ = std::make_unique<codecs::FishDacCodecRuntime>(component, codec_backend,
        codec_backend.threads, codec_options.graph_arena_bytes, codec_encoder_f32_accumulation, codec_encoder_chunk_samples);
    if (const auto value = runtime::find_option(options.options, {"fish_drama3.reference_cache"})) {
        reference_cache_ = runtime::parse_bool_option(*value, "fish_drama3.reference_cache");
    }
    reference_cache_ = reference_cache_ && backbone_config.disk_experts && reference_prefix_cache_;
    if (reference_cache_) {
        try {
            reference_cache_dir_ = reference_cache_root(options);
            reference_cache_config_ = reference_cache_config(backbone_config, cache_steps_, device_sampling_);
            codec_cache_config_ = "drama3-reference-codes-v1:" +
                std::to_string(codec_encoder_chunk_samples) + ":" +
                std::to_string(static_cast<int64_t>(codec_options.encoder_bf16_compute)) + ":" +
                std::to_string(static_cast<int64_t>(codec_encoder_f32_accumulation)) + ":" +
                std::to_string(static_cast<int64_t>(codec_backend.type));
            std::error_code error;
            const auto & path = assets_->model_weights->source_path();
            const auto bytes = std::filesystem::file_size(path, error);
            reference_cache_identity_ = error ? path.string() : path.string() + ":" + std::to_string(bytes);
        } catch (const std::exception & error) {
            reference_cache_ = false;
            debug::trace_log_scalar("drama3.reference_cache_error", std::string_view(error.what()));
        }
    }
    debug::trace_log_scalar("drama3.reference_cache", std::string_view(reference_cache_ ? "enabled" : "disabled"));
    log_free_vram(execution, "after_load");
    vram_margin_mb_ = runtime::parse_i64_option(options.options, {"fish_drama3.vram_margin_mb"}).value_or(0);
    if (vram_margin_mb_ < 0 || vram_margin_mb_ > 16384)
        throw std::runtime_error("fish_drama3.vram_margin_mb must be between 0 and 16384");
    reserve_vram_margin(execution);
    assets_->model_weights->release_storage();
    assets_->codec_weights->release_storage();
}

// Hold device memory for the process lifetime. Every later allocation then has
// to fit under (device total - margin).
//
// MEASURED WARNING: on gfx1101 this is NOT a safe default. The reserve is taken
// out of the same headroom the reference-encoder graph needs (about 4.1 GiB for
// a 58 s reference), so a 1024 MiB reserve at expert_cache_size=96 turned a
// working run into an out-of-memory abort; on this card an allocation failure
// at the edge can wedge the GPU ring and force a reboot. Left opt-in; the
// usable fix is to bound the spike itself with
// fish_drama3.codec_encoder_chunk_samples and to refuse a configuration before
// loading it, not to hold memory back.
void Drama3Generator::reserve_vram_margin(core::ExecutionContext & execution) {
    debug::trace_log_scalar("fish_drama3.vram_margin_mb", vram_margin_mb_);
    if (vram_margin_mb_ <= 0) {
        debug::trace_log_scalar("drama3.vram_margin.status", std::string_view("disabled"));
        return;
    }
    if (execution.backend_type() == core::BackendType::Cpu) {
        debug::trace_log_scalar("drama3.vram_margin.status", std::string_view("cpu_backend"));
        return;
    }
    constexpr double kMib = 1024.0 * 1024.0;
    const auto snapshot = execution.memory_snapshot();
    if (!snapshot.available) {
        debug::trace_log_scalar("drama3.vram_margin.status", std::string_view("unavailable"));
        return;
    }
    const int64_t free_mb = static_cast<int64_t>(static_cast<double>(snapshot.free_bytes) / kMib);
    debug::timing_log_scalar("drama3.vram.total_mb", static_cast<double>(snapshot.total_bytes) / kMib);
    debug::timing_log_scalar("drama3.vram.free_mb", free_mb);
    if (free_mb < vram_margin_mb_) {
        debug::trace_log_scalar("drama3.vram_margin.status", std::string_view("insufficient"));
        throw runtime::CapacityError("Drama 3 loaded but only " + std::to_string(free_mb) +
            " MiB of VRAM stays free, below the fish_drama3.vram_margin_mb reserve of " +
            std::to_string(vram_margin_mb_) +
            " MiB; lower fish_drama3.expert_cache_size or fish_drama3.vram_margin_mb");
    }
    vram_margin_buffer_ = ggml_backend_alloc_buffer(execution.backend(),
        static_cast<size_t>(vram_margin_mb_) * 1024u * 1024u);
    if (vram_margin_buffer_ == nullptr) {
        debug::trace_log_scalar("drama3.vram_margin.status", std::string_view("failed"));
        throw runtime::CapacityError("Drama 3 could not reserve the fish_drama3.vram_margin_mb reserve of " +
            std::to_string(vram_margin_mb_) + " MiB; lower fish_drama3.expert_cache_size or fish_drama3.vram_margin_mb");
    }
    debug::trace_log_scalar("drama3.vram_margin.status", std::string_view("reserved"));
    debug::timing_log_scalar("drama3.vram_margin_reserved_mb", vram_margin_mb_);
    debug::timing_log_scalar("drama3.vram.free_after_margin_mb",
        static_cast<double>(execution.memory_snapshot().free_bytes) / kMib);
}

Drama3Generator::~Drama3Generator() {
    if (vram_margin_buffer_ != nullptr) ggml_backend_buffer_free(vram_margin_buffer_);
}

// Restore the reference rows, their hidden states and both snapshot states from
// the cache when this package and these options produced them.
void Drama3Generator::load_reference_cache(const std::vector<Drama3Frame> & prefix, int64_t rows) {
    const std::vector<Drama3Frame> frames(prefix.begin(), prefix.begin() + rows);
    const auto key = reference_cache_key(reference_cache_identity_, reference_cache_config_, frames);
    const auto path = reference_cache_dir_ / (key + ".bin");
    const auto started = std::chrono::steady_clock::now();
    auto entry = read_reference_cache(path, reference_cache_identity_, reference_cache_config_,
        frames.size(), assets_->backbone.hidden_size);
    if (!entry || entry->frames != frames ||
        entry->hidden.size() != frames.size() * static_cast<size_t>(assets_->backbone.hidden_size)) {
        debug::trace_log_scalar("drama3.reference_cache", std::string_view(entry ? "invalid" : "miss"));
        return;
    }
    reference_prefix_frames_ = std::move(entry->frames);
    reference_prefix_hidden_ = std::move(entry->hidden);
    reference_prefix_state_ = std::move(entry->backbone);
    reference_prefix_tower_state_ = std::move(entry->tower);
    debug::trace_log_scalar("drama3.reference_cache", std::string_view("hit"));
    debug::timing_log_scalar("drama3.reference_cache_load_ms", debug::elapsed_ms(started));
}

// Persist what the two-phase prompt evaluation produced. Entries are keyed by
// package, numerics options and the reference frames, so a different voice,
// package or graph shape never reads this one back.
void Drama3Generator::store_reference_cache() {
    if (!reference_cache_ || reference_prefix_frames_.empty() || reference_prefix_state_.bytes.empty() ||
        reference_prefix_tower_state_.bytes.empty()) return;
    Drama3ReferenceCacheEntry entry;
    entry.frames = reference_prefix_frames_;
    entry.hidden = reference_prefix_hidden_;
    entry.backbone = reference_prefix_state_;
    entry.tower = reference_prefix_tower_state_;
    const auto key = reference_cache_key(reference_cache_identity_, reference_cache_config_, entry.frames);
    const auto started = std::chrono::steady_clock::now();
    const bool stored = write_reference_cache(reference_cache_dir_ / (key + ".bin"),
        reference_cache_identity_, reference_cache_config_, entry);
    debug::trace_log_scalar("drama3.reference_cache_store", std::string_view(stored ? "ok" : "failed"));
    debug::timing_log_scalar("drama3.reference_cache_store_ms", debug::elapsed_ms(started));
}

codecs::FishDacCodes Drama3Generator::encode_reference(const runtime::AudioBuffer & audio) {
    const auto encode_started = std::chrono::steady_clock::now();
    if (cached_reference_audio_ && cached_reference_audio_->sample_rate == audio.sample_rate &&
        cached_reference_audio_->channels == audio.channels && cached_reference_audio_->samples == audio.samples) {
        return cached_reference_codes_;
    }
    // A later process with the same voice starts from the stored codes instead
    // of encoding the clip again.
    const auto codes_path = reference_cache_ && !audio.samples.empty()
        ? reference_cache_dir_ /
            ("codes-" + reference_codes_key(reference_cache_identity_, codec_cache_config_, audio) + ".bin")
        : std::filesystem::path{};
    if (!codes_path.empty()) {
        if (const auto stored = read_reference_codes(codes_path, assets_->codec.total_codebooks)) {
            debug::trace_log_scalar("drama3.reference_cache_codes", std::string_view("hit"));
            debug::timing_log_scalar("drama3.reference_encode_ms", debug::elapsed_ms(encode_started));
            cached_reference_audio_ = audio;
            cached_reference_codes_ = *stored;
            return *stored;
        }
    }
    auto result = codec_->encode_codes(audio);
    log_free_vram(*execution_, "after_encode");
    codec_->release_encode_graph();
    log_free_vram(*execution_, "after_encode_release");
    debug::timing_log_scalar("drama3.reference_encode_ms", debug::elapsed_ms(encode_started));
    validate_drama3_codes(result, assets_->tower.codebook_size);
    if (!codes_path.empty()) {
        const bool stored = write_reference_codes(codes_path, result);
        debug::trace_log_scalar("drama3.reference_cache_codes",
            std::string_view(stored ? "stored" : "store_failed"));
    }
    // Keep one short reference in host memory. Exact waveform equality avoids
    // stale path-based hits when an uploaded file is replaced or re-decoded.
    constexpr size_t max_cached_samples = 44100 * 30;
    if (audio.samples.size() <= max_cached_samples) {
        cached_reference_audio_ = audio;
        cached_reference_codes_ = result;
    } else {
        cached_reference_audio_.reset();
        cached_reference_codes_ = {};
    }
    return result;
}

void Drama3Generator::warm_streaming_codec(const codecs::FishDacCodes & reference) {
    validate_drama3_codes(reference, assets_->tower.codebook_size);
    codecs::FishDacCodes frame;
    frame.codebooks = reference.codebooks;
    frame.frames = 1;
    for (int64_t book = 0; book < reference.codebooks; ++book) {
        frame.codes.push_back(reference.codes[static_cast<size_t>(book * reference.frames)]);
    }
    // Initialize and capture the first-frame codec graph during preparation.
    // Discard decoded reference audio and reset all causal stream state.
    for (int pass = 0; pass < 4; ++pass) {
        codec_->reset_decode_stream();
        (void)codec_->decode_stream(frame);
    }
    codec_->reset_decode_stream();
}

Drama3Generation Drama3Generator::generate(const std::string & text, const std::vector<Drama3Take> & references,
    const std::vector<Drama3Take> & history, const Drama3Sampling & sampling, bool streaming,
    const std::function<void(const runtime::AudioBuffer &)> & on_audio, bool continuation) {
    if (!std::isfinite(sampling.temperature) || sampling.temperature < 0 ||
        !std::isfinite(sampling.top_p) || sampling.top_p <= 0 || sampling.top_p > 1 ||
        sampling.top_k <= 0 || sampling.max_frames <= 0) throw runtime::InvalidRequestError("invalid Drama 3 sampling parameters");
    using ProfileClock = std::chrono::steady_clock;
    std::array<double, 9> component_ms{};
    auto component_started = ProfileClock::now();
    const auto profile_start = profile_first_audio_ ? ProfileClock::now() : ProfileClock::time_point{};
    auto profile_marked = profile_start;
    std::vector<std::pair<const char *, double>> first_audio_phases;
    const auto mark_phase = [&](const char * name) {
        if (!profile_first_audio_) return;
        const auto now = ProfileClock::now();
        first_audio_phases.emplace_back(name, std::chrono::duration<double, std::milli>(now - profile_marked).count());
        profile_marked = now;
    };
    int64_t reference_rows = 0;
    const auto prompt = build_drama3_prompt(*assets_, tokenizer_, text, references, history, &reference_rows);
    if (sampling.max_frames > cache_steps_ ||
        prompt.size() > static_cast<size_t>(cache_steps_ - sampling.max_frames)) {
        throw runtime::CapacityError("Drama 3 prompt plus max_tokens exceeds fish_drama3.context_length; increase it or shorten history");
    }
    auto & decoder = decoder_;
    // A continuation may extend the exact frames already evaluated. Never
    // carry recurrent state across independent requests or changed histories.
    const size_t prefix_rows = disk_experts_ && continuation && !cached_prefix_.empty() &&
        cached_prefix_.size() <= prompt.size() && std::equal(cached_prefix_.begin(), cached_prefix_.end(), prompt.begin())
        ? cached_prefix_.size() : 0;
    auto prefix_hidden = std::move(cached_hidden_);
    cached_prefix_.clear();
    cached_hidden_.clear();
    // Start independent requests from the same expert placement, but preserve
    // LRU reuse between continuation chunks within a request.
    if (prefix_rows == 0) decoder->reset(disk_experts_ && !continuation);
    tower_->reset();
    mark_phase("setup_ms");
    component_ms[0] += debug::elapsed_ms(component_started);
    component_started = ProfileClock::now();
    std::vector<int32_t> ids;
    for (const auto & frame : prompt) ids.push_back(frame[0]);
    auto embeddings = backbone_->token_embedding(ids);
    tower_->mix_backbone_embeddings(embeddings, prompt);
    mark_phase("embedding_ms");
    component_ms[1] += debug::elapsed_ms(component_started);
    component_started = ProfileClock::now();
    const int64_t hidden_size = assets_->backbone.hidden_size;
    const int64_t prompt_rows = static_cast<int64_t>(prompt.size());
    Drama3BackboneDecoderForwardResult prefill;
    if (prefix_rows != 0) {
        prefill.hidden = std::move(prefix_hidden);
        prefill.steps = static_cast<int64_t>(prompt.size());
        for (size_t row = prefix_rows; row < prompt.size(); ++row) {
            const auto begin = embeddings.begin() + row * assets_->backbone.hidden_size;
            const auto hidden = decoder->run_embedding_step({begin, begin + assets_->backbone.hidden_size});
            prefill.hidden.insert(prefill.hidden.end(), hidden.begin(), hidden.end());
        }
    } else {
        // A second process with the same reference starts from the persisted
        // state instead of evaluating 1250 rows again.
        if (reference_cache_ && reference_prefix_frames_.empty() && reference_rows > 0 && reference_rows <= prompt_rows) {
            load_reference_cache(prompt, reference_rows);
        }
        // A request that starts with the reference rows already evaluated only
        // has to evaluate the text tail. The reference state is kept after the
        // first request of the session, so a server that answers many requests
        // with one voice pays the prompt cost once.
        int64_t reuse_rows = 0;
        if (disk_experts_ && reference_prefix_cache_ && !reference_prefix_frames_.empty() &&
            static_cast<int64_t>(reference_prefix_frames_.size()) <= prompt_rows &&
            std::equal(reference_prefix_frames_.begin(), reference_prefix_frames_.end(), prompt.begin())) {
            reuse_rows = static_cast<int64_t>(reference_prefix_frames_.size());
        }
        const auto rows_of = [&](int64_t first, int64_t last) {
            const auto begin = embeddings.begin() + first * hidden_size;
            return std::vector<float>(begin, embeddings.begin() + last * hidden_size);
        };
        const auto frames_of = [&](int64_t first, int64_t last) {
            return std::vector<Drama3Frame>(prompt.begin() + first, prompt.begin() + last);
        };
        // The tower prefill runs beside each backbone phase, so it is timed here
        // rather than after the block.
        const auto tower_prefill = [&](const std::vector<float> & values,
                                       const std::vector<Drama3Frame> & frames, int64_t start_row) {
            const auto started = ProfileClock::now();
            tower_->prefill(values, frames, start_row);
            component_ms[3] += debug::elapsed_ms(started);
        };
        // Per-row fallback for a tail longer than the batched input set.
        const auto append_steps = [&](std::vector<float> & all, int64_t first, int64_t last) {
            for (int64_t row = first; row < last; ++row) {
                const auto hidden = decoder->run_embedding_step(rows_of(row, row + 1));
                all.insert(all.end(), hidden.begin(), hidden.end());
            }
        };
        // The rows after a retained prefix were evaluated one cached step at a
        // time at 27 ms per row, because the batched prefill graph cannot see a
        // restored prefix. The batched tail can, so a warm request now pays one
        // pass for its text instead of one pass per row.
        const auto append_tail = [&](std::vector<float> & all, int64_t first, int64_t last) {
            const int64_t rows = last - first;
            if (rows <= 0) return;
            if (rows <= decoder->max_batched_rows()) {
                const auto hidden = decoder->prefill_rows(rows_of(first, last), rows);
                all.insert(all.end(), hidden.hidden.begin(), hidden.hidden.end());
                return;
            }
            append_steps(all, first, last);
        };
        std::vector<float> all_hidden;
        if (reuse_rows > 0) {
            decoder->restore_state(reference_prefix_state_);
            tower_->restore_state(reference_prefix_tower_state_);
            all_hidden = reference_prefix_hidden_;
            append_tail(all_hidden, reuse_rows, prompt_rows);
            if (reuse_rows < prompt_rows) {
                const std::vector<float> tail_hidden(all_hidden.begin() + (reuse_rows - 1) * hidden_size,
                    all_hidden.end());
                tower_prefill(tail_hidden, frames_of(reuse_rows - 1, prompt_rows), reuse_rows - 1);
            }
            mark_phase("reference_reuse_ms");
        } else if (disk_experts_ && reference_prefix_cache_ && reference_rows > 0 && reference_rows < prompt_rows) {
            // Two phases: the state at the reference boundary is what a later
            // request can restore, so it has to exist on its own.
            auto head = decoder->prefill_embeddings(rows_of(0, reference_rows), reference_rows, 0);
            tower_prefill(head.hidden, frames_of(0, reference_rows), 0);
            reference_prefix_frames_.assign(prompt.begin(), prompt.begin() + reference_rows);
            reference_prefix_hidden_ = head.hidden;
            reference_prefix_state_ = decoder->snapshot_state();
            reference_prefix_tower_state_ = tower_->snapshot_state();
            store_reference_cache();
            all_hidden = std::move(head.hidden);
            // Same tail path as a restored request, so a first request and a
            // cached one produce the same take from the same prefix state.
            append_tail(all_hidden, reference_rows, prompt_rows);
            const std::vector<float> tail_hidden(all_hidden.begin() + (reference_rows - 1) * hidden_size,
                all_hidden.end());
            tower_prefill(tail_hidden, frames_of(reference_rows - 1, prompt_rows), reference_rows - 1);
            mark_phase("reference_prefix_ms");
        } else {
            auto out = decoder->prefill_embeddings(embeddings, prompt_rows);
            all_hidden = std::move(out.hidden);
            tower_prefill(all_hidden, prompt, 0);
        }
        prefill.hidden = std::move(all_hidden);
        prefill.steps = prompt_rows;
    }
    log_free_vram(*execution_, "after_prefill");
    debug::trace_log_scalar("drama3.prefill_reused_rows", prefix_rows);
    mark_phase("backbone_prefill_ms");
    component_ms[2] += debug::elapsed_ms(component_started);
    component_started = ProfileClock::now();
    // The tower prefill runs with the backbone in the branches above: a
    // restored prefix must not be filled twice.
    mark_phase("tower_prefill_ms");
    component_ms[3] += debug::elapsed_ms(component_started);
    auto hidden = last_hidden(prefill.hidden, assets_->backbone.hidden_size);
    Drama3Frame current = prompt.back();
    std::mt19937 random(sampling.seed);
    SamplerScratch sampler_scratch;
    std::deque<int32_t> recent;
    std::vector<Drama3Frame> output_frames;
    Drama3Generation result;
    std::mutex codec_mutex;
    std::condition_variable codec_ready;
    std::deque<Drama3Frame> codec_queue;
    bool codec_done = false;
    bool codec_cancelled = false;
    std::exception_ptr codec_error;
    std::thread codec_worker;
    if (streaming) {
        runtime::StreamingAudioConfig config;
        config.frames_per_chunk = codec_batch_frames_;
        config.frame_width = 10;
        streaming_audio_.begin(config, [&](const auto & window) {
            codecs::FishDacCodes codes;
            codes.codebooks = 10;
            codes.frames = static_cast<int64_t>(window.new_frames);
            codes.codes.resize(window.new_frames * 10);
            for (size_t book = 0; book < 10; ++book) {
                for (size_t frame = 0; frame < window.new_frames; ++frame) {
                    codes.codes[book * window.new_frames + frame] = window.data[frame * 10 + book];
                }
            }
            auto audio = codec_->decode_stream(codes);
            if (window.first_frame == 0) {
                mark_phase("codec_first_frame_ms");
                if (profile_first_audio_) {
                    std::cerr << "[drama3.first_audio] prompt_rows=" << prompt.size();
                    for (const auto & phase : first_audio_phases) std::cerr << " " << phase.first << "=" << phase.second;
                    std::cerr << " total_ms=" << std::chrono::duration<double, std::milli>(ProfileClock::now() - profile_start).count() << '\n';
                }
            }
            return audio;
        }, [&](const runtime::StreamEvent & event) {
            if (on_audio && event.audio_output) on_audio(*event.audio_output);
        }, {}, [&] { codec_->reset_decode_stream(); });
    }
    auto stop_codec = [&](bool cancel) {
        if (!codec_worker.joinable()) return;
        {
            std::lock_guard<std::mutex> lock(codec_mutex);
            codec_done = true;
            codec_cancelled = cancel;
            if (cancel) codec_queue.clear();
        }
        codec_ready.notify_all();
        codec_worker.join();
    };
    struct CodecJoin {
        decltype(stop_codec) & stop;
        runtime::StreamingAudioController<int32_t> & stream;
        ~CodecJoin() { stop(true); stream.reset(); }
    } codec_join{stop_codec, streaming_audio_};
    if (streaming && codec_async_decode_) {
        codec_worker = std::thread([&] {
            try {
                for (;;) {
                    Drama3Frame frame;
                    {
                        std::unique_lock<std::mutex> lock(codec_mutex);
                        codec_ready.wait(lock, [&] { return codec_done || !codec_queue.empty(); });
                        if (codec_queue.empty()) break;
                        frame = codec_queue.front();
                        codec_queue.pop_front();
                    }
                    codec_ready.notify_all();
                    streaming_audio_.push({frame.begin() + 1, frame.end()});
                }
                if (!codec_cancelled) result.audio = streaming_audio_.finish();
            } catch (...) {
                std::lock_guard<std::mutex> lock(codec_mutex);
                codec_error = std::current_exception();
                codec_queue.clear();
                codec_ready.notify_all();
            }
        });
    }
    const double expert_stage_ms_before = backbone_->expert_stage_ms();
    for (int64_t step = 0; step < sampling.max_frames; ++step) {
        component_started = ProfileClock::now();
        auto logits = backbone_->lm_head(hidden);
        component_ms[4] += debug::elapsed_ms(component_started);
        component_started = ProfileClock::now();
        // Both drama3 (lm_head_token_ids) and drama3-small (lm_head_rows) return
        // compact [EOS, semantic 0..N-1] logits from the backbone.
        auto candidates = std::move(logits);
        const int32_t first = sample(candidates, sampling.temperature, sampling.top_k, sampling.top_p, random, sampler_scratch);
        const int32_t second = sample(candidates, 1.0F, 30, 0.95F, random, sampler_scratch);
        const int32_t first_token = first == 0 ? assets_->eos : assets_->tower.semantic_begin + first - 1;
        const bool repeats = first != 0 && std::find(recent.begin(), recent.end(), first_token) != recent.end();
        const int32_t selected = repeats ? second : first;
        component_ms[5] += debug::elapsed_ms(component_started);
        if (selected == 0) break;
        if (step == 0) mark_phase("semantic_ms");
        Drama3Frame next{};
        next[0] = assets_->tower.semantic_begin + selected - 1;
        next[1] = selected - 1;
        const auto tower_started = ProfileClock::now();
        const auto sampling_before_tower = component_ms[5];
        const int64_t position = decoder->valid_steps() - 1;
        const auto & config = assets_->tower;
        if (device_sampling_) {
            std::array<float, 9> uniforms{};
            if (sampling.temperature != 0 && sampling.top_k != 1) {
                for (auto & uniform : uniforms) uniform = static_cast<float>(std::generate_canonical<double, 53>(random));
            }
            next = tower_->run_frame(hidden, current, selected - 1, position,
                sampling.temperature, sampling.top_k, sampling.top_p, uniforms);
        } else {
            const auto shared_input = tower_->shared_input({current});
            auto tower_hidden = hidden;
            for (int64_t layer = 0; layer < config.shared_layers + config.residual_layers; ++layer) {
                const int64_t book = layer < config.shared_layers ? 0 : layer - config.shared_layers;
                const auto tower_output = tower_->run_layer(layer, tower_hidden, {next[1 + book]}, {1.0F},
                                                            layer == 0 ? shared_input : std::vector<float>{}, position);
                tower_hidden = tower_output.hidden;
                if (layer >= config.shared_layers) {
                    component_started = ProfileClock::now();
                    next[2 + book] = sample(tower_output.logits, sampling.temperature, sampling.top_k, sampling.top_p, random, sampler_scratch);
                    component_ms[5] += debug::elapsed_ms(component_started);
                }
            }
        }
        component_ms[6] += debug::elapsed_ms(tower_started) - (component_ms[5] - sampling_before_tower);
        output_frames.push_back(next);
        if (step == 0) mark_phase("tower_first_frame_ms");
        recent.push_back(next[0]);
        if (recent.size() > 16) recent.pop_front();
        if (streaming) {
            if (codec_async_decode_) {
                std::unique_lock<std::mutex> lock(codec_mutex);
                codec_ready.wait(lock, [&] { return codec_error || codec_queue.size() < 2 * codec_batch_frames_; });
                if (codec_error) std::rethrow_exception(codec_error);
                codec_queue.push_back(next);
                lock.unlock();
                codec_ready.notify_one();
            } else {
                streaming_audio_.push({next.begin() + 1, next.end()});
            }
        }
        if (step + 1 == sampling.max_frames) { result.reached_limit = true; break; }
        current = next;
        component_started = ProfileClock::now();
        auto embedding = backbone_->token_embedding({current[0]});
        tower_->mix_backbone_embeddings(embedding, {current});
        component_ms[1] += debug::elapsed_ms(component_started);
        component_started = ProfileClock::now();
        hidden = decoder->run_embedding_step(embedding);
        if (disk_experts_) prefill.hidden.insert(prefill.hidden.end(), hidden.begin(), hidden.end());
        component_ms[7] += debug::elapsed_ms(component_started);
    }
    if (streaming) {
        stop_codec(false);
        if (codec_error) std::rethrow_exception(codec_error);
        if (!codec_async_decode_) result.audio = streaming_audio_.finish();
        component_ms[8] += streaming_audio_.decode_ms() + streaming_audio_.publish_ms();
    }
    if (output_frames.empty()) throw std::runtime_error("Drama 3 ended before producing an audio frame");
    if (disk_experts_) {
        cached_prefix_ = prompt;
        cached_prefix_.insert(cached_prefix_.end(), output_frames.begin(), output_frames.end());
        // At max_tokens the final sampled frame has not entered the backbone.
        cached_prefix_.resize(static_cast<size_t>(decoder->valid_steps()));
        cached_hidden_ = std::move(prefill.hidden);
    }
    result.codes.codebooks = 10;
    result.codes.frames = static_cast<int64_t>(output_frames.size());
    result.codes.codes.resize(output_frames.size() * 10);
    for (int64_t book = 0; book < 10; ++book) {
        for (size_t row = 0; row < output_frames.size(); ++row) result.codes.codes[book * output_frames.size() + row] = output_frames[row][book + 1];
    }
    if (!streaming) {
        component_started = ProfileClock::now();
        if (disk_experts_) {
            // Preserve KV/recurrent state for continuation, but do not overlap
            // completed prompt workspaces with the waveform decoder workspace.
            backbone_->release_prefill_graphs();
            tower_->release_prefill_graphs();
        }
        result.audio = codec_->decode_codes(result.codes);
        codec_->release_runtime_graphs();
        component_ms[8] += debug::elapsed_ms(component_started);
    }
    const char * component_keys[] = {"drama3.setup_ms", "drama3.embedding_ms", "drama3.backbone_prefill_ms",
        "drama3.tower_prefill_ms", "drama3.lm_head_ms", "drama3.sampling_ms", "drama3.tower_ms",
        "drama3.backbone_step_ms", "drama3.codec_ms"};
    for (size_t i = 0; i < component_ms.size(); ++i) debug::timing_log_scalar(component_keys[i], component_ms[i]);
    debug::timing_log_scalar("drama3.generated_frames", output_frames.size());
    // Decode time splits into expert staging (disk or pinned-host reads plus
    // uploads) and everything else, which is the GPU work.
    debug::timing_log_scalar("drama3.backbone_step_stage_ms",
        backbone_->expert_stage_ms() - expert_stage_ms_before);
    if (result.reached_limit) std::cerr << "[fish_drama3] max_tokens reached; audio may be incomplete\n";
    return result;
}

}  // namespace engine::models::fish_drama3
