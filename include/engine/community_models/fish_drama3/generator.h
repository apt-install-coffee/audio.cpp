#pragma once

#include "engine/community_models/fish_drama3/prompt.h"
#include "engine/framework/runtime/streaming_audio.h"

#include "ggml-backend.h"

#include <filesystem>
#include <functional>
#include <optional>
#include <string>

namespace engine::models::fish_drama3 {

struct Drama3Sampling {
    float temperature = 1.0F;
    float top_p = 0.9F;
    int top_k = 4097;
    int64_t max_frames = 1024;
    uint32_t seed = 0;
};

struct Drama3Generation {
    codecs::FishDacCodes codes;
    runtime::AudioBuffer audio;
    bool reached_limit = false;
};

class Drama3Generator {
public:
    Drama3Generator(std::shared_ptr<const Drama3Assets> assets, core::ExecutionContext & execution,
                    const runtime::SessionOptions & options, bool streaming);
    ~Drama3Generator();
    codecs::FishDacCodes encode_reference(const runtime::AudioBuffer & audio);
    void warm_streaming_codec(const codecs::FishDacCodes & reference);
    Drama3Generation generate(const std::string & text, const std::vector<Drama3Take> & references,
        const std::vector<Drama3Take> & history, const Drama3Sampling & sampling, bool streaming,
        const std::function<void(const runtime::AudioBuffer &)> & on_audio = {}, bool continuation = false);

private:
    std::shared_ptr<const Drama3Assets> assets_;
    Drama3TextTokenizer tokenizer_;
    int64_t cache_steps_;
    bool profile_first_audio_ = false;
    bool disk_experts_ = false;
    bool device_sampling_ = false;
    size_t codec_batch_frames_ = 4;
    bool codec_async_decode_ = true;
    runtime::StreamingAudioController<int32_t> streaming_audio_;
    std::unique_ptr<Drama3BackboneDecoderRuntime> backbone_;
    std::unique_ptr<Drama3BackboneDecoderRuntime::DecodeSession> decoder_;
    std::vector<Drama3Frame> cached_prefix_;
    std::vector<float> cached_hidden_;
    // Reference-prefix state. The prompt rows that belong to the voice
    // reference are identical across requests, so their decoder and tower
    // state is kept and restored instead of being evaluated again.
    bool reference_prefix_cache_ = true;
    // Persisted reference codes and post-prefix state. A new process then skips
    // the ~1.3 s encode and the ~4 s prompt evaluation of the same voice.
    bool reference_cache_ = true;
    std::filesystem::path reference_cache_dir_;
    std::string reference_cache_identity_;
    std::string reference_cache_config_;
    std::string codec_cache_config_;
    std::vector<Drama3Frame> reference_prefix_frames_;
    std::vector<float> reference_prefix_hidden_;
    Drama3BackboneState reference_prefix_state_;
    Drama3TowerState reference_prefix_tower_state_;
    std::unique_ptr<Drama3Tower> tower_;
    std::unique_ptr<core::ExecutionContext> codec_execution_;
    std::unique_ptr<codecs::FishDacCodecRuntime> codec_;
    std::optional<runtime::AudioBuffer> cached_reference_audio_;
    codecs::FishDacCodes cached_reference_codes_;
    // Device bytes held for the process lifetime and never used. It keeps the
    // last free VRAM away from Drama 3's spiky per-request graphs, so an
    // over-large request fails with an allocation error instead of starving
    // the desktop compositor, which wedges the GPU ring on gfx1101.
    int64_t vram_margin_mb_ = 0;
    ggml_backend_buffer_t vram_margin_buffer_ = nullptr;
    void reserve_vram_margin(core::ExecutionContext & execution);
    void load_reference_cache(const std::vector<Drama3Frame> & prefix, int64_t rows);
    void store_reference_cache();
};

}  // namespace engine::models::fish_drama3
