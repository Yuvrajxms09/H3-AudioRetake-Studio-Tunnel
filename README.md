# H3 Audio Retake Studio

A local, browser-based audio and lip-sync repair studio for ComfyUI and MiniMax H3. Select a passage on a waveform, make or import speech takes, mark the speaker and mouth, compare a repair, and export the full video. The interface orchestrates ComfyUI jobs; you do not need to wire separate graphs for each operation.

This independent repository is based on [Nebuluss/H3-AudioRetake-studio](https://github.com/Nebuluss/H3-AudioRetake-studio) and adds exact-origin support for HTTPS reverse tunnels.

> **Early experimental software.** Check the generated words, voice, face tracking, lip-sync, transitions and colors. The model may change a face or produce artifacts.

## Demo

[Watch the narrated studio presentation](https://github.com/Yuvrajxms09/H3-AudioRetake-Studio-Tunnel/raw/refs/heads/main/demo/H3_Audio_Retake_Studio_presentation_narrated.mp4) (4 minutes). It walks through the interface, retake modes, sound and track retakes, and example results.

[Watch the studio walkthrough and edited cyclops scene](https://github.com/Yuvrajxms09/H3-AudioRetake-Studio-Tunnel/raw/refs/heads/main/demo/studio-walkthrough.mp4). The walkthrough shows actual interface captures followed by a generated dialogue edit. The longer final passage in this example was rendered with a custom ComfyUI workflow and assembled in Tesseract; the studio currently limits individual lip repairs to about seven seconds.

## AI coding agents

[AGENTS.md](AGENTS.md) explains the current workflow, limitations, troubleshooting, research and safe testing for coding agents. Claude can also discover [CLAUDE.md](CLAUDE.md).

For the custom tunnel, SAM3 and LipForcing changes, implementation rationale, mask-coordinate details, failure guards and the Hemsworth run record, see [the implementation notes](docs/implementation-notes.md).

For the broader PX-365 conversation handoff—including the text-driven Studio path, the supplied-audio FL2VA/Exact Audio Lock prototype, what was tested, what remains separate, and where the parent-workspace artifacts live—see [the fresh-chat handoff](docs/px365-fresh-chat-handoff.md).

## Install

1. Install and start an up-to-date [ComfyUI](https://github.com/comfyanonymous/ComfyUI) with a working MiniMax H3 setup. This version is tested locally, with ComfyUI and the browser on the same machine.
2. Clone [this repository](https://github.com/Yuvrajxms09/H3-AudioRetake-Studio-Tunnel) into `ComfyUI/custom_nodes`:

   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/Yuvrajxms09/H3-AudioRetake-Studio-Tunnel.git
   ```

   Install these ComfyUI node packs and their dependencies as directed by their authors:
   - [ComfyUI-H3-Motion-Context-MultiRef](https://github.com/seitanism/ComfyUI-H3-Motion-Context-MultiRef)
   - [ComfyUI-H3-FaceRefine](https://github.com/Carasibana/ComfyUI-H3-FaceRefine)
   - [ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite)
3. Install FFmpeg and FFprobe and make both available on the ComfyUI process PATH. Face tracking and optional export color matching also need OpenCV and NumPy in ComfyUI's Python environment; install FaceRefine's requirements there. The face detector may download its weights on first use.
4. Install compatible H3 Ref2VA, Qwen H3 text encoder, H3 video/audio VAEs and a **Pika H3 Flex Ref2VA** LoRA in their appropriate ComfyUI model folders. Open the studio's **H3 model setup** to choose the actual filenames. For **SAM3 text prompt** masks, also install a ComfyUI-compatible SAM3 checkpoint (for example [Comfy-Org SAM 3.1](https://huggingface.co/Comfy-Org/sam3.1)) in `ComfyUI/models/checkpoints`. The Studio uses ComfyUI's built-in `CheckpointLoaderSimple`, `SAM3_VideoTrack`, `SAM3_TrackToMask` and mask/image blur nodes; no extra SAM3 node pack is needed. The app does not include or automatically download models. Check their separate terms before use.
5. Restart ComfyUI and open **http://127.0.0.1:8188/audio-retake/**. If ComfyUI uses a different port, use that port instead. A separate studio server is not required.

To use **LipForcing aligned mask**, clone the official [LipForcing repository](https://github.com/cvlab-kaist/LipForcing) beside ComfyUI and install its preprocessing dependencies in the same Python environment ComfyUI uses: PyTorch/torchvision, OpenCV, Kornia, einops, InsightFace, and a CUDA-enabled ONNX Runtime. The Studio calls LipForcing's `ImageProcessor` and the notebook's `weights/mask(1).png`; it does **not** load LipForcing's 14B generation checkpoint or run LipForcing generation. The InsightFace `buffalo_l` detector/106-point landmark assets are still required. Configure these before starting ComfyUI:

```python
os.environ["LIPFORCING_ROOT"] = "/content/LipForcing"
os.environ["LIPFORCING_MASK_PATH"] = "/content/LipForcing/weights/mask(1).png"
```

The defaults discover the notebook's `/content/LipForcing` checkout and mask when present. The Studio's LipForcing readiness check reports missing source, template, dependencies, or CUDA provider before preprocessing. Restart ComfyUI after installing dependencies or changing these environment values.

The Pika sampler node bundled here follows an experimental six-step ComfyUI schedule. It does not include the Pika weights or claim exact parity with Pika's original inference code. See [Pika H3 Flex](https://huggingface.co/Pika-Labs/Pika-H3-Flex) and its model license. ComfyUI, the other node packs, their dependencies and model weights are independent projects with their own licenses.

## HTTPS tunnel access (optional)

The Studio UI and ComfyUI API share the same ComfyUI server and origin. A single tunnel to the ComfyUI HTTP address serves both; open `/audio-retake/` on that tunnel hostname. The browser submits workflows and reads results through that same origin.

By default, Studio routes accept localhost only. For a stable tunnel hostname, set `H3_AUDIO_RETAKE_ALLOWED_ORIGINS` in the ComfyUI process environment to one or more exact HTTPS origins, separated by commas:

```bash
export H3_AUDIO_RETAKE_ALLOWED_ORIGINS="https://studio.example.com,https://another-studio.example.net"
python main.py
```

Use origins only: include the scheme and hostname, with an optional port; do not include a path. The incoming `Host` must match a configured origin. Tunnel requests must reach ComfyUI from the same machine through a loopback connection, as with `cloudflared` running beside ComfyUI. Preserve the public hostname in the `Host` header; if the tunnel has an HTTP Host Header override, leave it unset or set it to that exact hostname. The Studio does not trust forwarded host or scheme headers. Restart ComfyUI after changing this environment setting.

For a Quick Tunnel whose URL is created after ComfyUI starts, configure `H3_AUDIO_RETAKE_ALLOWED_ORIGINS_FILE` before starting ComfyUI. The file can be updated while ComfyUI is running; Studio rereads it for each request. For example, in Colab:

```python
allowed_origins_file = COMFYUI / "user" / "default" / "audio_retake_allowed_origins.txt"
allowed_origins_file.parent.mkdir(parents=True, exist_ok=True)
allowed_origins_file.write_text("", encoding="utf-8")
os.environ.pop("H3_AUDIO_RETAKE_ALLOWED_ORIGINS", None)
os.environ["H3_AUDIO_RETAKE_ALLOWED_ORIGINS_FILE"] = str(allowed_origins_file)
```

In the tunnel output loop, replace the URL-printing block with:

```python
match = re.search(r"https://[A-Za-z0-9-]+\.trycloudflare\.com", line)
if match:
    origin = match.group(0)
    temporary_file = allowed_origins_file.with_suffix(".tmp")
    temporary_file.write_text(origin + "\n", encoding="utf-8")
    os.replace(temporary_file, allowed_origins_file)
    print("ComfyUI URL:", origin)
```

Keep the same `cloudflared` process running; the URL remains valid until that process stops. The file accepts one origin per line or comma-separated origins. Invalid entries are ignored. Quick Tunnel hostnames change when a new tunnel is created, so write the new exact URL after each new tunnel starts.

This setting adds no authentication. Anyone who can access the tunnel can use the ComfyUI server exposed through it, including its workflow and file APIs. Keep the URL private and stop the tunnel when finished.

## Make a retake

1. Open your original video or audio and drag over a short interval on the waveform. Drag a selection edge to adjust it, zoom up to 32×, type IN and OUT to the millisecond, nudge them by 1 ms, 10 ms, one frame or 100 ms, or use **Snap edges to pauses**.
2. Enter the words and optionally a voice reference, then generate takes. **Acting preset** fills the acting direction, with an intensity setting; edit the text freely or save your own presets in this browser. Presets are prompt guidance, not measured controls: in tests they changed Generate takes clearly, Free acting modestly, and a normal joint retake barely, because the voice reference and surrounding speech dominate. Generated takes follow the estimated line length, because H3 fills spare time with invented speech when a voice reference is used. The reference player lets you check the recording. You can edit the prompt directly. Alternatively, import a recording.
3. Listen and choose a take. Use **Take start** to skip unwanted leading silence. Audio-only export replaces the selected interval and adds the chosen short fades; a take shorter than the interval leaves silence after it.
4. For lip-sync, open a video, choose the speaker on the first frame, track their face and review the colored tracking box. **LipForcing aligned mask** is the default under **Mask source**; you can still choose **Manual oval** or **SAM3 text prompt**. For SAM3, select an installed checkpoint, enter a prompt such as “lower face including lips, mouth, jaw, and chin,” generate the mask preview, then play and scrub it beside the tracked crop across the passage. For LipForcing, generate and review its mask preview in the same way.
5. Adjust denoise strength, mouth mask softness and face color match. A low denoise may preserve more of the face but weaken lip-sync. The Pika six-step schedule rounds settings to whole steps. Scrub to a clear face frame and select it as the reference; the first tracked frame is the default. A thumbnail shows the selected image here and on each dialogue turn.
6. Render, compare the original and repair, then export the full video. The selected supplied audio is locked during lip-sync and used in the export. The optional **Reduce full-frame color jumps** calibration compares repair frames to the source before export. It is experimental, adds processing time, and can fail or change the image on challenging footage; turn it off if it makes a result worse.

### Voice only and off-screen voice (experimental)

**Voice only** repairs speech and keeps the source picture. Export attempts to copy the video stream; if that fails, it falls back to re-encoding. Enter the words, select the passage and render; tracking the face is optional and gives H3 a closer view of the lips. **Off-screen voice** does the same for a speaker outside the frame. **Match level to nearby speech** adjusts the retake by up to 6 dB, and **Free acting** hides the surrounding speech from H3 so the acting direction has more effect. **Variations** renders several seeds and ranks them by how many expected words local Whisper hears. Fill in **Language** for that score: left empty, Whisper detects the language phrase by phrase, so short noises can come out as another language and a correct take can score 0%. Leave **Keep the original ambience** off when the noise is the defect, because the separated background would bring it back. **Transcribe** uses PyTorch, torchaudio, Transformers and a local Hugging Face-format Whisper model folder under `models/whisper` to build a clickable timed-word list; click a word to select it and shift-click to extend. Below it, **Edit the full transcript** lets you change a copy of the recognized script. **Review changed passages** maps word-level edits back to the original word timings; adjust each IN/OUT time or copy the current waveform selection before queueing. Case and punctuation-only edits to the same words do not trigger speech regeneration. Insertions and deletions include adjacent retained words because Joint Retake needs an existing audio interval and replacement words; removing all speech in a passage is unsupported. Queueing creates one Joint Retake turn per changed passage, with LipForcing selected and five H3 variations per turn. Open each turn to track the speaker and review that turn's LipForcing mask before rendering. The original ASR transcript and the edited draft remain separate. The current project, including the draft and reviewed passage times, is saved in this browser and can be restored from the header; clearing site data removes it.

### Redo the SFX (experimental)

Choose **Redo the SFX** at the top of the Retake panel to replace the sound of a shot instead of a line. No mask or tracking is needed, and a video with no sound track opens too. The studio selects the whole clip; drag on the waveform to redo one passage instead. H3 keeps the real frames in its video latent and regenerates only the audio, so the export normally copies the source picture unchanged. One pass covers up to about 15 seconds.

Leave **Sound to generate** empty for the natural sound of the shot, or describe what you want. **Suggest from the video** asks a local [Ollama](https://ollama.com) vision model, such as Gemma, to look at frames across the selection and write a prompt for you to edit. It can name things that are not in the shot, so check it. Its wording also sets the level: on a clip with faint ambience, a suggestion that said "gentle" came out about 20 dB quieter than a direct request for strong wind. **Keep the original voices** separates the voices from the source with Mel-Band RoFormer and places the new sound under them. Use **Retake gain** to set the level of the new sound; it applies at export, so you can change it after rendering.

### Tracks (experimental)

**Split into voices and sound** separates the source with Mel-Band RoFormer and opens two tracks under the waveform: **Voices**, and **Sound** for everything else, music included. Sound is the source minus Voices, so at 0 dB the two add back to the source exactly. The separator is built for singing and speech, so laughter or a shout may land on either track; solo each one to check.

- Each track has **M** (mute), **S** (solo) and a level from -24 to +12 dB. Playback follows them live, and **Hear the source** switches back to the original for comparison.
- Select a passage on any waveform, then choose **Redo the sound here** or **Redo the voice here**. The passage opens in the Retake panel and appears on its track when rendered; a retake changes only that track. Sound passages use Redo the SFX, and H3 hears only the Sound track around the passage. Where lips move, H3 can still invent speech; voice separation keeps it out of the new sound. Voice passages use your line mode, or Voice only if you were on Lip-sync.
- Passages can overlap when they are on different tracks: you can redo the sound behind a line and the line itself. Passages on the same track still cannot overlap.
- **Export mix** puts what you hear on the source video, with your levels, Mute and Solo applied, and it works with no retake at all. The other export buttons use the tracks too once they are open. The picture stays identical unless a passage changed the mouth. **Download tracks** saves both as WAV files at full level, retakes included.

Tracks are saved with the project. Music and effects share the Sound track, because separating them needs another model.

### Joint retake (experimental)

Choose **Joint retake** under Repair mode to replace or add speech inside the shot. H3 regenerates the voice and the mouth in one pass, while the real picture and sound around the interval stay in the latent as context. Select only the words to replace, enter them under **Words to speak**, then track the speaker and choose a mask source. Review the full mask preview before rendering. SAM3 uses ComfyUI’s built-in `SAM3_VideoTrack` and `SAM3_TrackToMask` nodes with one tracked object and text detection on every frame. Its optional `GrowMask` expansion is 0 px by default and runs before edge softening; it needs an installed SAM3 checkpoint in the ComfyUI checkpoints folder. LipForcing uses its own InsightFace landmark alignment and fixed LatentSync mask on each Studio tracked crop, then maps that mask back into the crop coordinates Joint Retake consumes. The upstream mask's white means preserve, so the Studio inverts it to the H3 regenerate-mask convention; pixels outside the aligned face stay preserved. Both generated-mask modes show a preview, and the render graph runs the same configured mask-generation path. The LipForcing path fingerprints its source code and mask template so a changed installation invalidates the preview. LipForcing requires its preprocessing source, mask template, InsightFace assets and CUDA ONNX Runtime, but not its generation checkpoint. A window of at least 124 frames is built around the selection, so the source needs about 5.2 seconds of video. This mode needs a ComfyUI restart after installation or update for the **Audio Retake - Timed AV Mask** and **Audio Retake - LipForcing Mouth Mask** nodes. The audio mix keeps the source outside the retake before export; the final MP4 is encoded and is not guaranteed to be bit-identical. **Face direction** adds a visible reaction after the line; it can only show inside the mask. **Edge fade** sets how far the regeneration extends beyond the selection; shorten it between close words. Audio edges resolve to about 25 ms and video edges to about 170 ms. Keep the new line close to the original length: a static mask does not stop the mouth if the line ends early inside the interval.

The full-transcript editor is a reviewed orchestration layer over this existing turn workflow; it does not change the H3 graph or auto-detect speakers. You can compare five generated candidates for every queued edit and keep the best candidate per turn. Queueing does not submit a ComfyUI job until the turn is prepared and rendered.

For dialogue, or several separate areas of one clip, add each waveform passage as a turn, up to eight without overlap. Give passages the same speaker label to reuse the voice and acting. **Render all turns** renders whatever is missing in time order; a turn that is not ready is reported and the others still run. Passages can sit close together: when the fades cross, the row says so, and each picture repair stops halfway between the two lines, so check that join. **Export all completed turns** combines them. Labels are manual; they do not identify or isolate voices. A completed lip repair can also download its ComfyUI **API-format workflow**. That JSON refers to model names and local input files; it does not package those assets or represent the earlier interactive choices.

## What is preserved

- The original files are never overwritten. Export re-encodes the full MP4, so untouched pixels and audio are not bit-identical.
- Replacing an audio interval replaces everything audible there, including other voices and ambient sound. Use separate stems or a careful external mix when needed.
- Speaker selection uses FaceRefine detection and tracking. The mask can be a manually drawn oval, an optional SAM3 mask generated over the tracked crops, or a LipForcing aligned mouth template mapped into those crops. SAM3 segments the prompted region; it does not independently track lip motion or identify speakers. LipForcing mask generation detects/aligns the face in each Studio crop and fails if a frame has no face; it does not run LipForcing's video generator. Inspect the colored face-tracking preview and selected mask throughout the passage. The built-in SAM3 nodes/checkpoint and LipForcing preprocessing dependencies are separate from the FaceRefine custom nodes.
- The model can alter the face, even with a face reference, and the replacement may not be perfectly synchronized. The color calibration cannot guarantee continuity.
- Each lip repair is a short single-shot passage (about seven seconds maximum) and requires enough source frames after the interval for H3 padding. Browser-supported codecs and the upload limit apply. A project with takes, turns or tracks can be saved in this browser and restored from the header; browser storage limits and clearing site data can remove it. Opening a file alone does not replace the saved project. Downloaded files and ComfyUI output files remain separately.
- The app makes separate ComfyUI jobs for audio, tracking and rendering because the user chooses and reviews results between steps. It is one interface, not one unattended monolithic ComfyUI graph.

## Data and access

The interface and its export endpoints accept localhost requests by default. When exact HTTPS origins are configured, they are accepted only when the request arrives from a loopback client and the host and write-request origin match the configuration. This is an origin check, not authentication. Uploaded source files, voice references, masks and intermediate clips are copied to ComfyUI's `input/AudioRepairStudio` and `output/AudioRepairStudio` folders. Full exports go to `output/AudioRepairStudio/exports`. Remove files you no longer need using your usual file manager. The repository includes the public demonstration video; it does not include voice-reference files, model weights or tokens.

## Project status

This is an early community prototype, tested with a local Windows/NVIDIA setup. Other platforms, model variants, GPUs and unusual frame rates need more testing. Contributions and issue reports with reproducible, shareable examples are welcome.

## Acknowledgements

Special thanks to [Alisson Anjos](https://github.com/alisson-anjos) for [Henge](https://github.com/alisson-anjos/Henge), whose friendly interface helped inspire this studio, and to [seitanism](https://github.com/seitanism) for [ComfyUI-H3-Motion-Context-MultiRef](https://github.com/seitanism/ComfyUI-H3-Motion-Context-MultiRef) and its H3 audio/video tools.

## License

The application source is available under the [MIT License](LICENSE), the same license used by Henge. ComfyUI, third-party nodes, Pika H3 Flex and all model weights have their own licenses and are not redistributed here.
