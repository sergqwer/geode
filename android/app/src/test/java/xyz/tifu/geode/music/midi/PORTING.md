# YuE2 MIDI and lyric port

Source: `work/ref/yue2-comfyui`, Apache-2.0. The reference was read only. No Python, Gradle, reference scripts,
network, models, or ComfyUI runtime were run. Compilation and JUnit verification used the cached Kotlin 2.0.21
compiler and JUnit 4.13.2 directly on the JVM.

## Files and API

The production package contains `MidiImport.kt`, `MidiReader.kt`, `MidiParts.kt`, `Karaoke.kt`, `MidiChords.kt`,
`ScoreEvents.kt`, `Phrasing.kt`, `Underlay.kt`, `LyricUnderlay.kt`, `LyricSyllables.kt`, and `Messages.kt`.
No Android imports are used.

The requested entry points are `MidiImport.read(bytes, mode, vocalTrack, instrumentTrack, withoutSections)`
and `Underlay.lay(abc, lyrics)`. `MidiError.code/args`, `Notice.code/args`, and `Why` carry localization data.
The other public objects expose the upstream readers and pure helpers for testing and future editors.
`LyricUnderlay.lay` draws syllables under an already named score; `Underlay.lay` rearranges an unnamed tune.
They implement the two distinct upstream operations, both using `AbcScore.parse`.

## Test inventory

Parameterized Python cases are loops inside JUnit methods. Counts below count source test functions and then
their expanded parameter cases; additional assertions within one source function are not counted as separate cases.

| Python file | Source functions ported | Expanded cases | Kotlin test file |
| --- | ---: | ---: | --- |
| test_midi_smf.py | 11 | 20 | MidiSmfTest.kt |
| test_midi_parts.py | 20 | 51 | MidiPartsTest.kt |
| test_midi_score.py | 19 | 26 | MidiScoreTest.kt |
| test_load_midi.py | 5 core portions | 6 | LoadMidiTest.kt |
| test_phrasing.py | 16 | 16 | PhrasingTest.kt |
| test_underlay.py | 20 | 20 | LyricUnderlayTest.kt |

There are 91 JUnit methods for the source cases, plus `exportedScoreRoundTrip` (92 total).
The latter uses `ScoreMidi.of` on the existing `abc_transpose.json` score `awkward`, and compares both voices'
exact onsets, pitches, durations, bar grids, tempo, and recovered chords after import in full mode.
`MidiFixtures.kt` ports `tests/midi_files.py`. `resources/midi/underlay_songs.json` is a verbatim copy of the
two-song upstream fixture; all four lyric variants in both languages are checked.

English warning/error assertions check enum codes and raw arguments. The four `_bar_list` cases check the first
eight bar numbers and the omitted count, allowing the application to format them in its language.
The source's legacy `facts` fields that are absent from the requested Result are checked through resulting notes,
ABC headers, notices, selected track roles, and helpers rather than duplicated as another facts object.

## Skipped ComfyUI plumbing

From `test_load_midi.py`, these nine functions depend on node widgets, the ComfyUI input folder, edit provenance,
node cache hooks, or HTTP route envelopes, which do not exist in the byte-array Kotlin API:

- `test_the_widgets_sit_in_their_final_order`: ComfyUI widget schema and saved workflow ordering.
- `test_only_midi_files_at_the_top_of_the_input_folder_are_offered`: ComfyUI input-folder picker.
- `test_an_edit_for_this_file_mode_and_tracks_is_output_and_one_for_another_choice_is_not`: node edit provenance.
- `test_an_edit_goes_out_bare_or_named_the_same_way`: node edit provenance; bare/named helpers are tested separately.
- `test_lyrics_belong_to_the_file`: node edit provenance.
- `test_a_path_instead_of_a_name_is_refused_before_the_disk_is_asked`: input-folder and HTTP route path validation.
- `test_a_file_that_cannot_be_read_keeps_its_path_in_the_console`: ComfyUI filesystem error reporting.
- `test_a_changed_file_runs_again_and_a_wrong_name_is_caught_before_the_run`: ComfyUI IS_CHANGED/VALIDATE_INPUTS.
- `test_the_tracks_route_lists_a_file_before_any_run`: ComfyUI route response and status codes; the part list is tested directly.

The five retained functions cover output score/lyrics/tracks, without_sections, karaoke, unreadable/unsingable
data, and saving a score to MIDI. Their UI provenance fields, filename wrappers, missing-input-folder-file case,
HTTP base64 envelope, and HTTP invalid-request responses are node plumbing and omitted.

From `test_underlay.py`, three functions exercise the ComfyUI editor's HTTP interface and are omitted:

- `test_the_words_route_lays_the_lyrics_under_the_score`: route response/JSON envelope; the same layout is tested directly.
- `test_the_words_route_refuses_what_the_editor_never_sends`: HTTP request shape and payload limit.
- `test_the_words_route_says_why_a_score_cannot_be_laid`: HTTP error response; bare scores are tested directly.

No pure MIDI, phrasing, or syllable layout cases were skipped. None of these suites needed torch or a model.

## Deliberate adaptations

- Exact beat and note arithmetic uses `AbcScore.Fraction`; the ABC renderer lives in this package because the
  existing parser and MIDI writer must not be edited. ABC whitespace and rest compression need not match Python;
  the tests compare the sounding score.
- Python's phrasing ceiling is rounded to one decimal place. `Laid.maxSeconds` must be an Int, so it rounds that
  decimal ceiling upward. The decimal value remains in `LYRICS_LAID` arguments. For a 31.2-second laid score this
  is 36.3 in the notice and 37 in maxSeconds. `Phrasing.ceiling` exposes the original decimal calculation.
- The optional unsung clause is a separate `UNSUNG_BARS` notice. All source bar numbers are preserved as data;
  localization may abbreviate long lists to the upstream six-section/six-range display limit.
- `CHORDS` and `DRUMS` are explicit track roles, and unselected parts have `Why.NOT_SELECTED`, where the original
  description sometimes used an empty string. No translated UI sentences are emitted.
