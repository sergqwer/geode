// yue2_master.hpp - a finished song's last steps on the phone, as the workstation's yue2_server.py does them with
// ffmpeg: the integrated loudness (ITU-R BS.1770-4 / EBU R128: K-weighting, 400 ms blocks every 100 ms, the absolute
// gate at -70 LUFS and the relative one 10 LU under), make-up gain toward the target capped at +max_gain dB (a cut
// when the song is louder), a look-ahead peak limiter (ffmpeg alimiter's limit 0.891 = -1 dBFS, attack 5 ms, release
// 60 ms, no auto level); then the file: WAV (16 / 24-bit PCM) or FLAC (our own encoder: fixed predictors, Rice
// coding, stereo decorrelation).
#pragma once

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace geode::yue2 {

// Integrated loudness of interleaved audio [frames][channels] at `rate` (the K-weighting's coefficients are
// recomputed for the rate); -inf for silence (every block under the absolute gate).
double integrated_lufs(const float *audio, int64_t frames, int channels, int rate);
// The largest |sample| in dBFS (not the true peak: no oversampling)
double sample_peak_db(const float *audio, int64_t frames, int channels);

struct MasterOptions {
    double target_lufs = -12.5, max_gain_db = 4.0;
    double limit = 0.891, attack_ms = 5.0, release_ms = 60.0;
    // a gain of its own instead of the loudness's (an edit's new audio: its song's gain, so that it is laid into the
    // song's mastered audio at the song's level); NaN: measured toward target_lufs
    double fixed_gain_db = std::numeric_limits<double>::quiet_NaN();
};
struct MasterResult {
    double lufs_before = 0, lufs_after = 0, gain_db = 0, wanted_db = 0, peak_before_db = 0, peak_after_db = 0;
    bool capped = false;
};
// In place: the gain, then the limiter (the output keeps the input's length: the look-ahead's delay taken out)
MasterResult master(std::vector<float> &audio, int channels, int rate, const MasterOptions &o);

// bits 16 or 24 (WAV: 32 = float too); false when the file could not be written
bool write_wav(const std::string &path, const float *audio, int64_t frames, int channels, int rate, int bits);
bool write_flac(const std::string &path, const float *audio, int64_t frames, int channels, int rate, int bits);

// A song's file read back (an edit's old audio): WAV (PCM 16 / 24 / 32-bit, float 32) or FLAC (constant, verbatim,
// fixed and LPC subframes, Rice and Rice2 partitions with escapes, wasted bits, every stereo assignment; 8 to 24-bit) -
// interleaved [frames][channels], each integer sample q as q / (2^(bits-1) - 1): the writers' own scale, so a file of
// ours reads back to exactly the values it was quantized to. header_only: the format and frames alone, audio empty.
struct AudioFile {
    int channels = 0, rate = 0, bits = 0;
    int64_t frames = 0;
    std::vector<float> audio;
};
bool read_audio(const std::string &path, AudioFile &out, std::string &err, bool header_only = false);

}  // namespace geode::yue2
