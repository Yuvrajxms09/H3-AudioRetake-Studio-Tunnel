# PX-365 dialogue and lip-sync work — fresh-chat handoff

**Read this before continuing the work.** It captures the user's goal, decisions made in the conversation, what is implemented, what was run, what succeeded technically, what has not been evaluated, and where the reusable graphs and run records live. Read it together with [`AGENTS.md`](../AGENTS.md), [`README.md`](../README.md), and the detailed [`Studio implementation notes`](implementation-notes.md).

## Current state at a glance

- The product problem is PX-365: H3 V2A can make useful scene sound but dialogue can be unintelligible. The product needs to distinguish semantic dialogue from exact supplied speech, and must not imply that silent-video inference can recover unheard original words by reading lips.
- Two workflow tracks are being pursued:
  1. **Text → H3 Joint Retake:** the user enters the desired words; H3 generates the voice and mouth together in the existing shot. This is the H3 Audio Retake Studio's **Joint retake** mode. The Studio now supports a LipForcing-aligned mask source for this path.
  2. **Supplied speech audio → localized FL2VA edit with Exact Audio Lock:** an existing recording drives the mouth, and the same supplied waveform is used as the final soundtrack. A new ComfyUI API graph combines this audio-lock path with preprocessed LipForcing masks and a crop/uncrop/composite path intended to preserve the rest of the source video. It was exercised in a 12-pairing, 5-second Colab batch.
- These are two implementation/test tracks, **not evidence that both original PX-365 product modes have passed evaluation**. The silent-video, context-only semantic-dialogue mode (A) has not been benchmarked. The Studio test supplied an exact line, and the 12-run prototype supplied speech audio; both are exact-target tests (mode B or an audio-driven extension of B).
- The supplied-audio graph is a separate prototype. **The Studio's new LipForcing option is a mask-preprocessing mode; it does not integrate the custom-audio Exact Audio Lock graph into the Studio UI.** Do not tell the user those are already one workflow.
- The Studio code changes are in the public repository [H3-AudioRetake-Studio-Tunnel](https://github.com/Yuvrajxms09/H3-AudioRetake-Studio-Tunnel), on `main`. The mask implementation is commit `51282a9`; implementation docs were pushed afterward. Check `git status` and `git log` before editing.
- Both the latest original-audio and silent-source Studio jobs completed. The user asked not to assess their content; the files and graph metadata were saved, but no subjective quality verdict was made.
- The broad 12–20 clip PX-365 benchmark is **not complete**. The 12 custom-audio pairings are a focused workflow execution check, not a measured quality benchmark.

## Product problem and boundaries

PX-365 exists because current H3 Video-to-Audio can produce good sound effects while dialogue is gibberish or irrelevant. The original plan separates two customer tasks:

- **Mode A — semantic dialogue from video:** silent video plus optional scene context should produce intelligible, scene-appropriate speech when the user does not supply an exact script. This can only infer plausible dialogue; it cannot reconstruct unheard original words from lips. This mode has not yet had a dedicated benchmark in the recorded runs.
- **Mode B — exact intended line:** user-supplied dialogue text drives generated speech, optionally with mouth-region repair. The Studio Joint Retake run is in this category because it supplied the exact words. A supplied speech recording is a useful additional input path that tests exact audio performance and visible repair, but it is not a substitute for evaluating text-to-speech Mode B or semantic Mode A.

Audio-only V2A and visible mouth-pixel editing are different capabilities and need different labels. A successful render queue is not proof of intelligible speech, script adherence, good lip sync, or preserved identity. The go/no-go criterion remains consistent dialogue quality at viable unit economics; if it is not met, keep the product promise SFX-only and do not imply lip-reading/reconstruction.

## Conversation decisions and workflow map

This is the decision trail behind the current scope. It is here so a new agent does not revive a path the user already set aside or mistake a preprocessing choice for an inference mode.

1. **Initial candidate review:** the conversation started with PX-365 and asked for deep assessment of H3 Audio Retake Studio, LTX Dub-It, AuK, existing H3/LTX V2A Colabs, H3 voice/SFX retake, dedicated LTX Foley, and existing lip-sync paths. The user then narrowed the active comparison to H3 Audio Retake Studio and LTX Dub-It. AuK and LTX Foley remained research context, not implemented paths.
2. **LTX was explored, then parked:** LTX Dub-It workflows and ICLoRA were examined, including a silent-audio test. The user rejected LTX mouth-region regeneration for this project because it changed too much of the input face. Do not fold LTX into the current H3 plan unless the user reopens it.
3. **H3 mask iterations:** a manually drawn oval produced poor results. SAM3 prompting/tracking and H3 inpainting workflow JSONs were investigated; earlier masks were too small or the inpainting result was poor. The user asked for ComfyUI's official one-frame video tracking logic, not a hand-built mask or paste. The Studio's SAM3 option uses interval 1 and explicit expansion defaulting to 0 px. This still does not make SAM3 output inherently correct.
4. **LipForcing was narrowed to preprocessing:** the user wanted LipForcing's tested face alignment and mask preparation, not its 14B generation model. The notebook's InsightFace/106-landmark `ImageProcessor` path and fixed mask template were adapted for H3 crops. That led to the Studio's new LipForcing mask source and to separately prepared full-frame assets for the custom-audio experiment. Those are two integrations of related preprocessing, not the same graph contract.
5. **The two active implementation tracks were kept distinct:**
   - **Workflow 1: text-driven Joint Retake in Studio.** Input is a video plus desired words and optional source audio/context. H3 generates speech and mouth changes together. It is the direct test for text-conditioned dialogue; it does not guarantee exact wording. Any of the Studio's supported mask sources can constrain the editable region.
   - **Workflow 2: supplied speech audio into localized FL2VA + Exact Audio Lock.** Input is a preprocessed silent source, aligned mask/reference assets and an exact speech recording. FL2VA edits the masked picture, while Exact Audio Lock supplies the target speech as the final soundtrack. This is the newer standalone ComfyUI API prototype described below, not currently a Studio UI mode.
6. **Older supplied-audio baseline:** the user had previously praised an H3 R2V Exact Audio Lock workflow for lip sync, while noting that it changed more of the picture. The user reports FL2VA is a stronger lip-sync path; this has not been measured in a controlled head-to-head test. Do not call the new localized FL2VA prototype the old R2V graph.
7. **Latest Studio silent-source run:** the same Hemsworth crop and text were run once with original source audio and once from a silent source. The silent run used the Studio-generated graph submitted through ComfyUI APIs after the embedded file picker failed. The user asked not to assess that run's output content, so the handoff records execution and metadata only.
8. **Custom-audio matrix:** after the standalone FL2VA graph and runtime mask-layout adapter were prepared in Colab, four videos were paired with three supplied voice recordings (12 jobs). All post-fix jobs completed and metadata was checked. This is an execution matrix, not the full PX-365 benchmark and not an objective sync-quality evaluation.

The user has repeatedly asked for existing repository/workflow logic to be reused, with no improvised manual mask/paste fallback. When a graph requires preprocessing, describe the exact source assets and where that preprocessing occurs; do not imply the H3 graph regenerated or re-aligned assets that were already prepared.

**Do not mark Mode A complete based on either track.** The silent Studio run still had the user's exact text prompt; the custom-audio run had an exact recorded performance. A dedicated Mode A test needs silent video plus scene context only, no script/transcript, and evaluation for intelligibility, semantic fit, and unwanted speech.

## Workflow 1 — text-driven Joint Retake in H3 Audio Retake Studio

### What it is

The user opens a source video, selects a time range, chooses **Joint retake**, enters the words under **Words to speak**, selects the visible speaker, tracks the face, and chooses a mask source. H3 generates new speech and the edited mouth in the same H3 pass. The selected words are prompt instructions; they are not an exact waveform lock. H3 can add, omit, repeat or mispronounce words. Studio Whisper ranking is a candidate-selection aid and is not a guarantee.

The source audio can provide context or ambience; the UI also supports a video with no audio stream by creating a zero-filled browser audio buffer. A silent source still provides no original words, voice, or ambience for the model to recover.

### Mask choices in the current Studio code

1. **Manual oval:** a hand-drawn mask in the selected speaker's 512 × 512 tracked crop. Earlier attempts were too tight and produced poor mouth results; this is why the other options were explored.
2. **SAM3 text prompt:** ComfyUI's built-in SAM3 video tracking and track-to-mask nodes, run on the Studio face crops. It is configured for one object, `detect_interval: 1`, and optional `GrowMask` expansion defaults to 0 px. Earlier attempts showed only a small lower-lip region, which prompted the user to ask for the official per-frame node path. The current implementation makes the requested prompt/growth explicit and previewable; it does not guarantee the returned segmentation is correct.
3. **LipForcing aligned mask:** a new Studio option using LipForcing's preprocessing source and its fixed LatentSync mask template. Each Studio tracked crop is face-aligned and the mask is mapped back to the crop coordinates H3 consumes. The preview and render use the same fingerprinted ComfyUI node. It needs InsightFace/preprocessing dependencies and the mask asset, but **not** LipForcing's 14B generation checkpoint.

These are mask-preparation choices, not three speech models or inference modes. The new LipForcing option does not itself improve script adherence or produce audio. It controls the visible region available for H3 to edit.

### Studio code and runtime changes

The custom Studio repository is a ComfyUI custom node package with a browser UI served on the same ComfyUI origin. The browser builds API-format graphs; Python nodes handle route checks, H3 sampling, timing masks, and LipForcing preprocessing. The Studio is not a separate inference service.

The custom changes in this branch are:

- **Quick Tunnel support:** exact HTTPS-origin checks and a live allowlist file let a Colab cell write the ephemeral Cloudflare hostname after ComfyUI starts. The same tunnel serves the Studio UI and ComfyUI endpoints. The user explicitly chose no authentication; tunnel origin validation is not authentication.
- **SAM3 mode:** built-in ComfyUI SAM3 nodes, text prompt, `detect_interval: 1`, one selected object, reviewable per-frame preview, optional expansion defaulting to 0 px, and mask softening.
- **LipForcing mode:** readiness endpoint, source/template resolution, SHA-256 configuration fingerprint, `AudioRetakeLipForcingMask` node, preview, and use of that same node at render time. Safeguards cover crop-space alignment, keep-vs-regenerate polarity, per-job alignment state, wrong Python module shadowing, unavailable CUDA ONNX Runtime/device, incompatible crop shape, face-detection failure and empty/non-finite masks.
- **State preservation:** the mask mode and preview metadata are retained for a dialogue turn and invalidated when the source, speaker track, relevant mask setting or preprocessing fingerprint changes.

Implementation details, exact file ownership, frame geometry and known limits are in [`implementation-notes.md`](implementation-notes.md). The practical install, tunnel and usage guide is in [`README.md`](../README.md).

### Studio test actually completed

On 2026-11-18, the Hemsworth crop was run in Joint Retake with text `yuvraj is a good boy, respect him`, one variation, seed 42, LipForcing mask, denoise label 0.60, mask softness 14 px, face color match 0.25, 200 ms edge fade, Keep original ambience enabled, and Match level enabled. The selected range was 0–5 seconds. The input is 640 × 344, 29.97 fps, about 8 seconds.

- **Run with source audio:** ComfyUI prompt `5de7fa9e-5219-4c4a-b131-a06d1ee8949e`; succeeded. H3 graph output `joint_repair_00002_.mp4`, 640 × 344, 24 fps, 175 frames (about 7.292 seconds).
- **Run with silent source:** an audio-stripped stream-copy of the same source was made and verified to have no audio stream. The second job used the same Studio-produced graph with only source video node `80` and source-audio node `83` replaced (silent video and zero-filled WAV context). Prompt `42a97381-d89e-4d68-85e2-186109b16b40`; succeeded. Output `joint_repair_00003_.mp4`, 640 × 344, 24 fps, 175 frames (about 7.292 seconds).
- Both graphs kept the LipForcing mask and Mel-Band RoFormer ambience nodes. The silent test proves that this graph completed with silent source context; the video content was not assessed in response to the user's instruction.
- These downloaded ComfyUI outputs are **H3 graph/context results**, not the Studio's separate full-duration **Export full video** result. They are shorter than the 8.008-second input because the graph emitted 175 frames at 24 fps.
- The embedded Studio file picker did not open during the silent run. Rather than claim a UI upload, the silent input and zero WAV were uploaded using the server's ComfyUI upload API, then the successful Studio graph was cloned and submitted through the prompt API. The mask, H3, stitch, stem and save topology came from that graph. This was an execution-path workaround, not a Studio code change.

The silent input and both outputs are in the local Downloads folder `H3_Studio_LipForcing_rerun/`. These files are not part of the public Studio repository.

## Workflow 2 — supplied audio into a masked FL2VA graph with Exact Audio Lock

### Why this is separate

The user previously tested H3 R2V with `MiniMaxH3TimedAudio`, `MiniMaxH3AddGuide`, and `MiniMaxH3ExactAudioLock`. They praised its lip sync, but it regenerated more of the picture than desired. That graph is an important supplied-audio baseline, not the source-preserving prototype. Its reusable local files are under `experiments/r2v_exact_audio_lock_best_v1/` in the earlier `707b` worktree; its README and `best_v1.json` explain its model settings and limitations. The user reports that FL2VA is a better sync baseline than R2V; treat this as their experience until a controlled comparison measures it.

The new prototype takes the useful supplied-audio conditioning/locking parts and combines them with the masked localized edit path. It is a ComfyUI API graph, not yet a new Studio UI mode. Its important distinction is:

- **The input video is preprocessed outside the graph:** a silent, 24 fps source, LipForcing-aligned mask video, and frame-0 reference are generated by the existing LipForcing notebook's `ImageProcessor` path. The video-generation checkpoint is not used for preprocessing.
- **The graph performs the edit:** FL2VA H3 sees the masked source crop and context, gets the supplied five-second speech through timed-audio conditioning and `MiniMaxH3AddGuide`, and uses `MiniMaxH3ExactAudioLock` to carry that supplied performance into the final MP4 soundtrack.
- **The picture is localized by the graph:** the source and aligned mask are loaded; the mask is resized, grown/softened, cropped to the subject, converted to the latent noise mask, then the decoded subject crop is uncropped and composited into the source plate. The H3 graph performs this crop/uncrop/composite; the side-by-side comparison movies are review artifacts only.

This is not the same as adding LipForcing preprocessing to the Studio. In the Studio mode, the ComfyUI node creates a mask from the tracked 512 × 512 crop at preview/render time. In the FL2VA prototype, the notebook writes frame-aligned source/mask/reference assets first and the API graph consumes those files. The two paths reuse LipForcing's preprocessing approach but have different input contracts and graph topology.

### Prototype graph and prompt

The 8-second Hemsworth template is in the parent workspace at `artifacts/avengers_hemsworth/H3_LipForcingMask_Hemsworth_CustomAudio_ExactLock_8s_api.json`; its design note is the adjacent `H3_LipForcingMask_Hemsworth_CustomAudio_ExactLock_8s_README.md`. The 8-second template itself was not run unchanged. The tested batch deep-copied its graph for each case, clipped the prepared videos/audio to 5 seconds, used 120 frames at 24 fps and set `ImageScaleToTotalPixels` to 0.18 MP. The workflow snapshots are in `artifacts/h3_custom_audio_5s_12_pairings/workflows/`.

The fixed prompt tells H3 to preserve subject/scene/framing, articulate the actual supplied audio's phonemes and timing, use that audio as the exact final soundtrack, and add no generated speech/SFX/music. The exact verbatim text is in `artifacts/h3_custom_audio_5s_12_pairings/README.md`; use it as the source of truth instead of retyping a remembered approximation.

Important graph stages and node roles:

| Stage | Nodes / function |
| --- | --- |
| Prepared inputs | Silent source via `VHS_LoadVideo`; aligned per-frame LipForcing mask via a second `VHS_LoadVideo`; frame-0 reference via `LoadImage`; five-second supplied voice via `LoadAudio`. |
| Mask treatment | `ImageResizeKJv2` → `ImageToMask` → `GrowMaskWithBlur` (64 px expansion, 32 px blur, tapered corners on) → mask/subject crop. These numbers make the actual editable region wider than the raw LipForcing template. |
| Source latent | Crop source, VAE encode, convert mask to latent space, set video noise mask; combine video latent with H3 audio scaffold using `LTXVConcatAVLatent`. |
| Audio conditioning/lock | `MiniMaxH3TimedAudio` starts at frame 0; `MiniMaxH3AddGuide` supplies speech guidance; `MiniMaxH3ExactAudioLock` supplies the locked AV latent and exact target audio. The final `VHS_VideoCombine` takes `exact_audio` from the lock node. |
| Inference / composite | FL2VA H3 sampler, decode video, `MVEx_SubjectUncrop`, then `ImageCompositeMasked` onto the source plate. H.264 output at 24 fps. |

The exact-audio-lock workflow replaces the soundtrack with the supplied speech track. It does not preserve/add source ambience, Foley or music. The mask/composite limits where the picture can change but cannot guarantee perfect identity or bit-identical pixels at/near the feathered edge.

### The 12-pairing run and its adapter

The 2026-11-18 Colab run used an NVIDIA L4, one fixed seed (0), 5-second silent 24 fps source clips at 0.18 MP, and each of three supplied five-second voice recordings:

| Source video | Chris Evans voice | Tom Cruise voice | RDJ voice |
| --- | --- | --- | --- |
| Hemsworth crop | completed | completed | completed |
| Chris Evans clip | completed | completed | completed |
| Tom Holland clip | completed | completed | completed |
| Tom Cruise clip | completed | completed | completed |

All 12 post-fix jobs completed. `ffprobe` confirmed an MP4 video stream plus AAC audio, 24 fps and 5-second duration; landscape sources produced 576 × 320, portrait Chris Evans produced 384 × 512. Recorded job time total was about 25m38s, excluding setup/preprocessing/package time. User said several outputs were particularly good, but there was no WER, phoneme-sync score, or controlled independent quality rating. Do not report all 12 as “perfect lipsync.”

#### Runtime adapter failure and status

The first six Exact Audio Lock submissions failed with `video noise_mask shape does not match the video latent`; the seventh attempt was stopped. Installed ComfyUI `SetLatentNoiseMask` returned a broadcastable mask such as `[T,1,H,W]`, while the H3 video latent was `[B,24,T,H,W]`; the plugin enforced strict equality. A narrow adapter was applied to the **active Colab copy** of `ComfyUI-H3-ExactAudioLock` to normalize only recognized compatible mask layouts and expand the mask over the 24 H3 video channels. It rejects unexpected shapes and does not change mask values or audio-lock behavior.

The original file, patched file and unified diff are in `artifacts/h3_custom_audio_5s_12_pairings/runtime_adapter/`. This patch was **not** committed to this Studio repo, to the local ComfyUI plugin, or upstream. The exact plugin/ComfyUI commits were not recorded. Existing plugin tests (65) and focused layout checks were reported passing before the 12 runs. A fresh runtime must inspect its installed plugin first; do not assume the patch is already installed or silently edit local ComfyUI.

### Saved custom-audio artifacts

The main run record is `artifacts/h3_custom_audio_5s_12_pairings/README.md`. It includes the runtime, prompt, settings, input mapping, per-run times, exact API graphs, runtime adapter and reproduction instructions. Generated source/mask/reference/audio assets and outputs are included there. The user-requested separated output videos are in the local Downloads folder `H3_custom_audio_5s_12_pairings_bundle/separated_videos/`; a ZIP is `H3_custom_audio_5s_12_pairings_bundle/separated_videos.zip`. The original combined reel is also present but should not be used in place of the individual outputs.

## Other investigated branches — do not confuse with the two active tracks

- **LTX-2.3 Dub-It ICLoRA:** examined earlier as a candidate and silent-audio experiment. The user later asked to focus on H3 and rejected LTX face-regeneration/mouth-mask workflows because they changed too much of the source face. LTX is parked; do not resume it unless the user explicitly reopens that comparison. Some experiment graphs and run records remain under `experiments/ltx_dub_it_silent_audio_v1/` and `runs/px365_ltx_dubit_silent_20261116/` in the parent workspace.
- **H3 inpainting-cropping 2-stage workflows:** reviewed as alternatives. The user tested a mask-based one-pass direction and disliked the results. Do not treat the sample face-replacement prompts as a validated lipsync path.
- **SAM3-only masks:** tried repeatedly; previous outputs tended to mask a small lip area or otherwise performed poorly. The code was changed to ComfyUI's built-in video tracking path, frame interval 1 and optional growth. This is now an available mask choice, not a proven best mask.
- **Manual oval:** remains available but the user's drawn mask produced poor results. Do not claim it matches the LipForcing preprocessing.
- **LipForcing generator:** not part of either active workflow. Only its existing face-alignment/mask preprocessing is reused. No LipForcing 14B generator weights are needed for this work.

## Tunnel/runtime context

The user starts ComfyUI in Colab, then starts `cloudflared`; a Quick Tunnel URL only becomes available after the tunnel starts and changes when a new runtime/tunnel is created. The Studio fork supports an exact-origin live allowlist file so the start cell can write the new URL after ComfyUI is already running. One HTTPS origin serves both Studio and ComfyUI API endpoints. The user explicitly said no authentication is needed; the code does not add auth. Do not reuse an old Cloudflare hostname. Restart ComfyUI after changing/installing Python custom-node code or dependencies; browser JS-only changes can reload without re-registering nodes.

The recent Hemsworth runs used an ephemeral tunnel hostname which may no longer work. Ask for or inspect the currently running URL before attempting another inference. The previous Colab runtime-adapter patch is also ephemeral; the local artifacts are its record, not evidence it persists in a later runtime.

## Decisions and collaboration preferences

- The user wants senior-engineer execution: inspect official ComfyUI nodes and existing repo logic, preserve existing behavior, reuse tested assets, make small explicit changes, and verify with actual graph/API results.
- Do not invent a separate manual mouth-paste/compositing path. The proven localized custom-audio graph uses its ComfyUI crop, latent mask, uncrop and `ImageCompositeMasked` nodes; the Studio graph uses its own `H3FaceStitch`. Explain which workflow actually did each operation.
- The user is comfortable making code changes inside Colab when dependencies or graph adapters are needed, but specifically wants to keep the connected L4 GPU unchanged.
- When running on the Studio server, API requests are acceptable and have been used successfully. Use the existing Studio/ComfyUI API graph; do not claim a UI upload occurred if the file picker failed.
- The user prefers output files saved in Downloads or directly shown in chat. For side-by-side custom-audio results they asked for separate files titled `input` and `output (audio name)`, not just a single comparison reel.
- When the user says “do not assess the output,” show files and verified metadata only. Do not invent a quality verdict or infer good sync from successful queue status/Whisper alone.
- Current scope is one visible person. Do not broaden the mask/tracking scope to multi-person handling without being asked.
- Avoid reopening LTX unless asked. Keep audio-only V2A, text-driven dialogue and custom-audio visual repair conceptually distinct.

## What remains open

1. **Product evaluation:** run the actual PX-365 benchmark (12–20 owned/licensed clips, visible/offscreen speaker, faces/animation/noise/quiet/action-only, short and longer durations, multiple voices/languages, at least two seeds) and record speech intelligibility, WER where applicable, semantic fit, unwanted speech, lip sync, SFX alignment and exact cost including failures. The present 12-run matrix is narrower: four clips, three voices, one seed, five seconds, 0.18 MP.
2. **Studio custom-audio integration:** decide whether to add a dedicated Exact Audio Lock mode to the Studio UI. The current Studio LipForcing option is only a mask source. The standalone FL2VA graph and its external prepared assets are not yet wired into `web/joint.js` or the UI as a new mode.
3. **Adapter durability:** establish exact ComfyUI/plugin versions and decide whether the mask-layout normalization belongs in an upstream fix, a maintained plugin release, or a documented runtime patch. Do not leave an untracked one-off patch as the only reproducible path.
4. **Browser upload path:** repeat the silent-source Studio test with the user's actual browser upload path if they want to verify it end-to-end. The prior successful silent run used the ComfyUI upload/prompt APIs after the embedded file picker failed.
5. **Quality evaluation:** assess the generated audio/video only when the user asks. The two Studio outputs were explicitly not reviewed for quality; the custom-audio run's user feedback was qualitative and not scored objectively.
6. **Longer outputs:** the reusable custom-audio JSON is an 8-second template, but the documented batch only ran 5-second overrides. A real unchanged 8-second run is not yet in the 12-run record.

## Suggested starting procedure for the next agent

1. Read this handoff, [`AGENTS.md`](../AGENTS.md), [`README.md`](../README.md), and [`implementation-notes.md`](implementation-notes.md). Inspect the repository branch and status; do not assume this thread's Cloudflare URL, Colab runtime, GPU, uploaded inputs or patch still exist.
2. Ask what the user wants to do next only if the goal is ambiguous. If they ask to continue the custom-audio workflow, inspect `artifacts/h3_custom_audio_5s_12_pairings/README.md`, the exact per-pair graph, and runtime adapter before touching Colab. Check the connected GPU and leave it unchanged.
3. If continuing the Studio text path, verify the current tunnel, ComfyUI object-info nodes, LipForcing readiness and model names. Use Joint Retake, a fresh mask preview, and the selected one-person track. Keep the source unchanged and save output separately.
4. For any new experiment, change one factor at a time, record the submitted graph and runtime versions, verify job completion plus output streams/duration, and separate execution success from subjective quality.
5. Update this handoff and the technical implementation note when the architecture or test evidence actually changes. Do not commit user media, models, secrets, local absolute paths or ephemeral tunnel URLs.
