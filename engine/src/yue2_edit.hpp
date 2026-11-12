// yue2_edit.hpp - an edit of a sung song (YuE2-ComfyUI's Edit Track, its inpaint/ core; docs/music-edit.md), from the
// song's state (yue2_state.hpp) and its audio: the edited codes, latents and noise put together, the frames around
// what changed drawn again by the acoustic stage while every other frame is held on its old latent (NarDsp::solve's
// held frames), only windows of up to 40 s either side of them drawn, the new sound decoded around them alone and laid
// into the old audio with 100 ms crossfades - the rest of the song's samples are its own.
//
// Cut: frames [first, stop) out. Inside the song the seam is drawn again (12 frames either side); at the start or the
// end there is no seam: the audio is trimmed and faded (0.2 s in, 1.5 s out), no model opened - or, with
// redraw_edges, the new edge is drawn again as a seam is.
//
// Retake: frames [first, stop) sung again with the same words (core.py retakes, one take of the request's seed). The AR
// sings on from the song's prompt and its codes before `first` (its guidance, the request's sampling, the penalty
// remembering the codes before), up to `width` frames more than the stretch; inside the song the join picks how many of
// them come - the count after which the old codes that follow are likeliest (`join_frames` of them, among the codes);
// at the end the model may end the song itself. The new frames' noise is the first rows of the request's seed; the new
// frames and `margin` either side are drawn, the rest held, and laid in as a cut's seam is.
#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "yue2_song.hpp"

namespace geode::yue2 {

// make_song with req.edit.op set comes here
SongResult edit_song(const SongRequest &req, const SongCallbacks &cb);

// The windows [first, stop) the acoustic stage draws for the frames `held` does not hold (inpaint/core.py windows_of):
// runs closer than two `reach` share a window while it fits `size` frames, each reaching `reach` frames past its runs
// either way where the song, the next run and `size` allow. False (err) when a run alone is longer than `size`.
bool windows_of(const std::vector<uint8_t> &held, int size, int reach, std::vector<std::pair<int, int>> &out, std::string &err);

}  // namespace geode::yue2
