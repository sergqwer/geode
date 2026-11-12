// yue2_view.hpp - what the track editor shows of a song (its waveform) and what places the score's bars on it (its
// pitch classes: YuE2-ComfyUI inpaint/grid.py _pitch_classes), taken every 10 ms from the finished audio and kept
// beside it (song.view): the app reads it in its own process without decoding the audio, and fits the score's
// clock to it (ui/music: BarGrid).
//
// The file: "GYV1", hops (u32), the samples a hop (u32: 480 at 48 kHz), the peaks (u8 a hop: the largest |sample| of
// either channel, 0..255 for 0..1), the pitch classes (12 u8 a hop: each hop's 12 values over its L2 norm, x 255).
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace geode::yue2 {

struct SongView {
    int hop = 480;                  // samples (10 ms at 48 kHz)
    std::vector<uint8_t> peaks;     // [hops]
    std::vector<uint8_t> classes;   // [hops * 12]
};

// [audio] interleaved [frames][channels] at [rate] -> its view (grid.py _pitch_classes: the mono mix, a 2048-sample
// periodic Hann window centred on every hop, the magnitudes' square roots of the bins between 80 Hz and 2 kHz summed
// into their nearest pitch class, each hop's 12 over their norm)
SongView view_of(const float *audio, int64_t frames, int channels, int rate);

bool write_view(const std::string &path, const SongView &v, std::string &err);
bool read_view(const std::string &path, SongView &v, std::string &err);

}  // namespace geode::yue2
