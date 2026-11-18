"""Audio Retake Studio: a ComfyUI interface and Pika sampler."""
import asyncio
import hashlib
import importlib.util
import ipaddress
import json
import math
import os
import re
import subprocess
import tempfile
import threading
import uuid
from pathlib import Path
from urllib.parse import urlsplit

# LipForcing sets this before importing InsightFace/ONNX Runtime. Set it here,
# before ComfyUI can lazily import the optional preprocessing dependencies.
os.environ.setdefault("ORT_DISABLE_THREAD_AFFINITY", "1")

from aiohttp import web
import folder_paths
from server import PromptServer

ROOT = Path(__file__).resolve().parent
WEB = ROOT / "web"
ALLOWED_ASSETS = {"index.html", "app.js", "models.js", "lip.js", "joint.js", "acting.js", "transcript.js", "project.js", "turns.js", "style.css", "h3_audio_preset.json", "lip_preset.json"}
routes = PromptServer.instance.routes
NO_CACHE = {"Cache-Control": "no-cache"}  # Page and scripts must update together.
ALLOWED_ORIGINS_ENV = "H3_AUDIO_RETAKE_ALLOWED_ORIGINS"
ALLOWED_ORIGINS_FILE_ENV = "H3_AUDIO_RETAKE_ALLOWED_ORIGINS_FILE"
LIPFORCING_ROOT_ENV = "LIPFORCING_ROOT"
LIPFORCING_MASK_ENV = "LIPFORCING_MASK_PATH"
_LIPFORCING_PROCESSORS = {}
_LIPFORCING_PROCESSOR_LOCK = threading.RLock()


def normalize_origin(value, *, https_only=False):
    """Return a normalized origin, rejecting URLs with paths or credentials."""
    if not isinstance(value, str):
        return None
    try:
        parsed = urlsplit(value.strip())
        scheme = parsed.scheme.lower()
        if scheme not in (("https",) if https_only else ("http", "https")):
            return None
        if not parsed.hostname or parsed.username or parsed.password:
            return None
        if parsed.path not in ("", "/") or parsed.query or parsed.fragment:
            return None
        port = parsed.port
        hostname = parsed.hostname.lower()
        if "*" in hostname:
            return None
        if ":" in hostname:
            hostname = f"[{hostname}]"
        default_port = 443 if scheme == "https" else 80
        authority = hostname if port in (None, default_port) else f"{hostname}:{port}"
        return f"{scheme}://{authority}"
    except ValueError:
        return None


def configured_tunnel_origins():
    """Read exact HTTPS origins from startup settings and a live config file."""
    configured_values = [os.environ.get(ALLOWED_ORIGINS_ENV, "")]
    config_path = os.environ.get(ALLOWED_ORIGINS_FILE_ENV, "").strip()
    if config_path:
        try:
            configured_values.append(Path(config_path).read_text(encoding="utf-8"))
        except OSError:
            pass

    origins = set()
    for configured in configured_values:
        for value in re.split(r"[,\r\n]+", configured):
            origin = normalize_origin(value, https_only=True)
            if origin:
                origins.add(origin)
    return origins


def lipforcing_installation():
    """Resolve the notebook's LipForcing source and mask without its generator weights."""
    configured_root = os.environ.get(LIPFORCING_ROOT_ENV, "").strip()
    comfy_root = Path(getattr(folder_paths, "base_path", None) or Path.cwd()).resolve()
    candidates = [Path(configured_root)] if configured_root else [
        comfy_root.parent / "LipForcing",
        Path.cwd() / "LipForcing",
        ROOT.parent.parent.parent / "LipForcing",
        Path("/content/LipForcing"),
        Path("/workspace/LipForcing"),
    ]
    root = next((p.resolve() for p in candidates
                 if p.is_dir() and (p / "OmniAvatar/utils/latentsync/image_processor.py").is_file()), None)
    if root is None:
        raise FileNotFoundError(
            f"LipForcing source was not found. Set {LIPFORCING_ROOT_ENV} to the cloned LipForcing repository."
        )

    configured_mask = os.environ.get(LIPFORCING_MASK_ENV, "").strip()
    masks = [Path(configured_mask)] if configured_mask else [
        root / "weights/mask(1).png",  # exact custom mask path used by Lip_Sync (1).ipynb
        root / "weights/mask.png",
        root / "OmniAvatar/utils/latentsync/mask.png",
    ]
    mask_path = next((p.resolve() for p in masks if p.is_file()), None)
    if mask_path is None:
        raise FileNotFoundError(
            f"LipForcing's mask template was not found. Set {LIPFORCING_MASK_ENV} to the notebook's mask(1).png."
        )
    return root, mask_path


def lipforcing_config_key(root, mask_path):
    """Fingerprint the exact upstream alignment code and mask used for preview/render."""
    digest = hashlib.sha256(b"audio-retake-lipforcing-mask-v1\0")
    for relative in (
        "OmniAvatar/utils/latentsync/image_processor.py",
        "OmniAvatar/utils/latentsync/affine_transform.py",
        "OmniAvatar/utils/latentsync/face_detector.py",
    ):
        path = root / relative
        digest.update(relative.encode("utf-8"))
        digest.update(path.read_bytes())
    digest.update(mask_path.read_bytes())
    return digest.hexdigest()


def lipforcing_readiness():
    """Return an opaque config fingerprint and actionable readiness information."""
    try:
        root, mask_path = lipforcing_installation()
        key = lipforcing_config_key(root, mask_path)
    except (FileNotFoundError, OSError) as exc:
        return {"available": False, "message": str(exc)}

    missing = [name for name in ("torch", "torchvision", "cv2", "kornia", "einops", "insightface", "onnxruntime")
               if importlib.util.find_spec(name) is None]
    if missing:
        return {
            "available": False,
            "config_key": key,
            "message": "LipForcing preprocessing dependencies are missing: " + ", ".join(missing) + ".",
        }
    try:
        import onnxruntime
        providers = onnxruntime.get_available_providers()
    except Exception as exc:
        return {"available": False, "config_key": key, "message": f"ONNX Runtime could not load: {exc}"}
    if "CUDAExecutionProvider" not in providers:
        return {
            "available": False,
            "config_key": key,
            "message": "LipForcing's InsightFace detector needs ONNX Runtime with CUDAExecutionProvider.",
        }
    return {"available": True, "config_key": key, "message": "LipForcing preprocessing is ready."}


def local_request(request, *, write=False):
    """Allow localhost or an explicitly configured HTTPS origin via loopback."""
    try:
        if not ipaddress.ip_address(request.remote).is_loopback:
            return False

        request_host = request.host.strip().lower()
        hostname = request_host.rsplit(":", 1)[0].strip("[]")
        is_local = hostname in ("localhost", "127.0.0.1", "::1")
        tunnel_origins = configured_tunnel_origins()
        host_origin = normalize_origin(f"https://{request_host}", https_only=True)
        tunnel_origin = host_origin if host_origin in tunnel_origins else None
        if not is_local and not tunnel_origin:
            return False

        if write:
            request_origin = normalize_origin(request.headers.get("Origin", ""))
            expected_origin = (
                normalize_origin(f"{request.scheme}://{request.host}")
                if is_local
                else tunnel_origin
            )
            if request_origin != expected_origin:
                return False
        return True
    except (ValueError, AttributeError):
        return False


def require_local(request, *, write=False):
    if not local_request(request, write=write):
        raise web.HTTPForbidden(
            text=(
                "Audio Retake Studio accepts localhost requests or an exact HTTPS "
                f"origin listed in {ALLOWED_ORIGINS_ENV}."
            )
        )


@routes.get("/audio-retake/")
async def index(request):
    require_local(request)
    return web.FileResponse(WEB / "index.html", headers=NO_CACHE)


@routes.get("/audio-retake/api/info")
async def info(request):
    require_local(request)
    return web.json_response({"comfy_input": folder_paths.get_input_directory(), "path_separator": __import__("os").sep})


@routes.get("/audio-retake/api/lipforcing/info")
async def lipforcing_info(request):
    require_local(request)
    return web.json_response(lipforcing_readiness())


@routes.get("/audio-retake/api/result/{name}")
async def result(request):
    require_local(request)
    name = request.match_info["name"]
    if not re.fullmatch(r"[0-9a-f]{32}\.mp4", name):
        raise web.HTTPNotFound()
    path = Path(folder_paths.get_output_directory()) / "AudioRepairStudio" / "exports" / name
    if not path.is_file():
        raise web.HTTPNotFound()
    return web.FileResponse(path, headers={"Content-Disposition": "attachment; filename=audio-retake.mp4"})


def safe_file(root, name):
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_. -]+", name) or name in (".", ".."):
        raise ValueError("Invalid local filename.")
    root = Path(root).resolve()
    expected = (root / "AudioRepairStudio").resolve()
    path = (expected / name).resolve()
    if path.parent != expected or not path.is_file():
        raise ValueError("A required local media file is missing.")
    return path


def build_export(args):
    source = safe_file(folder_paths.get_input_directory(), args.get("source"))
    audio = safe_file(folder_paths.get_input_directory(), args.get("audio"))
    turns = args.get("turns")
    if turns is None:
        turns = [{"repair": args.get("repair"), "start": args.get("start"), "end": args.get("end"), "clipStart": args.get("clipStart")}]
    if not isinstance(turns, list) or not 1 <= len(turns) <= 8:
        raise ValueError("Choose between one and eight turns.")
    resolved = []
    for turn in turns:
        if not isinstance(turn, dict):
            raise ValueError("Invalid turn.")
        # A voice-only retake has no repair video; the source picture stays as it is.
        repair = turn.get("repair") and safe_file(folder_paths.get_output_directory(), turn.get("repair"))
        start, end, clip_start = (float(turn[k]) for k in ("start", "end", "clipStart"))
        if not all(math.isfinite(x) for x in (start, end, clip_start)) or not 0 <= clip_start <= start < end <= 100000:
            raise ValueError("Invalid selected interval.")
        resolved.append((repair, start, end, clip_start))
    resolved.sort(key=lambda item: item[1])
    if any(a[2] > b[1] for a, b in zip(resolved, resolved[1:])):
        raise ValueError("Turn intervals cannot overlap.")
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "json", str(source)], capture_output=True, text=True, timeout=30, check=True, creationflags=flags)
    stream = json.loads(probe.stdout)["streams"][0]
    w, h = int(stream["width"]), int(stream["height"])
    if not (16 <= w <= 16384 and 16 <= h <= 16384):
        raise ValueError("Unsupported video dimensions.")
    output = Path(folder_paths.get_output_directory()) / "AudioRepairStudio" / "exports"
    output.mkdir(parents=True, exist_ok=True)
    name = uuid.uuid4().hex + ".mp4"
    if not isinstance(args.get("matchColors", False), bool):
        raise ValueError("Invalid color matching setting.")
    try:
        with tempfile.TemporaryDirectory(prefix="audio-retake-") as tmp:
            resolved = [item for item in resolved if item[0]]
            if not resolved:
                base = ["ffmpeg", "-v", "error", "-y", "-i", str(source), "-i", str(audio), "-map", "0:v:0", "-map", "1:a:0", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart"]
                proc = subprocess.run(base + ["-c:v", "copy", str(output / name)], capture_output=True, text=True, timeout=1200, creationflags=flags)
                if proc.returncode:
                    proc = subprocess.run(base + ["-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p", str(output / name)], capture_output=True, text=True, timeout=1200, creationflags=flags)
                if proc.returncode:
                    raise RuntimeError("Video export failed: " + proc.stderr[-1200:])
                return "/audio-retake/api/result/" + name
            if args.get("matchColors", False):
                from .colour_match import match_repair_color
                resolved = [(match_repair_color(source, repair, start, end, clip_start, tmp, i), start, end, clip_start)
                            for i, (repair, start, end, clip_start) in enumerate(resolved)]
            cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(source)]
            for repair, _, _, _ in resolved:
                cmd += ["-i", str(repair)]
            cmd += ["-i", str(audio)]
            filters, previous = [], "0:v"
            for i, (_, start, end, clip_start) in enumerate(resolved):
                filters.append(f"[{i+1}:v]scale={w}:{h}:flags=lanczos,setpts=PTS-STARTPTS+{clip_start:.9f}/TB[repair{i}]")
                filters.append(f"[{previous}][repair{i}]overlay=0:0:eof_action=pass:repeatlast=0:enable=\'between(t,{start:.9f},{end:.9f})\'[stage{i}]")
                previous = f"stage{i}"
            cmd += ["-filter_complex", ";".join(filters), "-map", f"[{previous}]", "-map", f"{len(resolved)+1}:a:0", "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(output / name)]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1200, creationflags=flags)
            if proc.returncode:
                raise RuntimeError("Video export failed: " + proc.stderr[-1200:])
    except Exception:
        (output / name).unlink(missing_ok=True)
        raise
    return "/audio-retake/api/result/" + name


@routes.post("/audio-retake/api/export")
@routes.post("/audio-retake/api/export-turns")
async def export(request):
    require_local(request, write=True)
    try:
        if request.content_type != "application/json" or (request.content_length or 0) > 10000:
            raise ValueError("Invalid export request.")
        args = await request.json()
        if not isinstance(args, dict):
            raise ValueError("Invalid export request.")
        url = await asyncio.to_thread(build_export, args)
        return web.json_response({"url": url})
    except (ValueError, KeyError, TypeError, FileNotFoundError, RuntimeError, subprocess.SubprocessError) as exc:
        return web.json_response({"error": str(exc)}, status=400)


@routes.get("/audio-retake/{name}")
async def asset(request):
    require_local(request)
    name = request.match_info["name"]
    if name not in ALLOWED_ASSETS:
        raise web.HTTPNotFound()
    return web.FileResponse(WEB / name, headers=NO_CACHE)


class AudioRetakePikaSampler:
    """Six-step experimental Pika H3 Flex sampling recipe."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",)}}

    RETURN_TYPES = ("MODEL", "SAMPLER", "SIGMAS")
    RETURN_NAMES = ("model", "sampler", "sigmas")
    FUNCTION = "configure"
    CATEGORY = "Audio Retake Studio"

    def configure(self, model):
        import torch
        import comfy.model_sampling
        import comfy.samplers
        if model.model.__class__.__name__ != "MiniMaxH3":
            raise ValueError("The Audio Retake Pika sampler requires a MiniMax H3 model.")
        patched = model.clone()
        class Sampling(comfy.model_sampling.ModelSamplingAV, comfy.model_sampling.CONST):
            pass
        original = patched.get_model_object("model_sampling")
        sampling = Sampling(patched.model.model_config)
        sampling.set_parameters(shift=12.0, audio_shift=3.0)
        if hasattr(original, "noise_scale"):
            sampling.set_noise_scale(original.noise_scale)
        patched.add_object_patch("model_sampling", sampling)
        options = patched.model_options["transformer_options"] = patched.model_options.get("transformer_options", {}).copy()
        options["minimax_h3_sigma_shift_video"] = 12.0
        options["minimax_h3_sigma_shift_audio"] = 3.0
        knots = torch.tensor([1.0, .75, .5, .25, .1, .04, 0.0], dtype=torch.float32)
        sigmas = 12.0 * knots / (1.0 + 11.0 * knots)
        return (patched, comfy.samplers.sampler_object("lcm"), sigmas)


def timed_ramp(frames, start, end, feather, fps=24.0):
    """Per-frame strength: 1 inside [start, end], easing to 0 over feather seconds."""
    import torch
    t = (torch.arange(frames, dtype=torch.float32) + 0.5) / fps
    if feather <= 0:
        return ((t >= start) & (t <= end)).float()
    x = torch.minimum(((t - start + feather) / feather), ((end + feather - t) / feather)).clamp(0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


class AudioRetakeTimedMask:
    """Limit a spatial mask to one interval, for the video and audio noise masks."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "mask": ("MASK",),
            "frames": ("INT", {"default": 124, "min": 1, "max": 3600}),
            "start_seconds": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3600.0, "step": 0.001, "tooltip": "Interval start, relative to the first frame of this clip."}),
            "end_seconds": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 3600.0, "step": 0.001, "tooltip": "Interval end, relative to the first frame of this clip."}),
            "feather_seconds": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 5.0, "step": 0.01, "tooltip": "Fade outside the interval. The interval itself stays fully regenerated."}),
        }}

    RETURN_TYPES = ("MASK", "MASK")
    RETURN_NAMES = ("video_mask", "audio_mask")
    FUNCTION = "build"
    CATEGORY = "Audio Retake Studio"

    def build(self, mask, frames, start_seconds, end_seconds, feather_seconds):
        if end_seconds <= start_seconds:
            raise ValueError("The interval end must be after its start.")
        spatial = mask if mask.ndim == 3 else mask.unsqueeze(0)
        if spatial.shape[0] not in (1, frames):
            raise ValueError("The mask must hold one frame or one frame per clip frame.")
        ramp = timed_ramp(frames, start_seconds, end_seconds, feather_seconds).to(spatial)
        return (spatial.expand(frames, -1, -1) * ramp.view(-1, 1, 1), ramp.view(-1, 1, 1))


class AudioRetakeLipForcingMask:
    """Warp LipForcing's aligned LatentSync mouth mask into H3's tracked-crop space."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "config_key": ("STRING", {"default": "", "multiline": False}),
        }}

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("regenerate_mask",)
    FUNCTION = "build"
    CATEGORY = "Audio Retake Studio"

    def build(self, images, config_key):
        import cv2
        import numpy as np
        import onnxruntime
        import torch
        import comfy.model_management as management

        root, mask_path = lipforcing_installation()
        current_key = lipforcing_config_key(root, mask_path)
        if not config_key or config_key != current_key:
            raise ValueError("LipForcing source or mask changed. Refresh and review the mask preview before rendering.")
        if "CUDAExecutionProvider" not in onnxruntime.get_available_providers():
            raise RuntimeError("LipForcing face alignment requires ONNX Runtime with CUDAExecutionProvider.")
        if images.ndim != 4 or images.shape[0] < 1 or images.shape[1:3] != (512, 512):
            raise ValueError("LipForcing mask preparation expects the Studio's 512×512 tracked face crops.")

        device = management.get_torch_device()
        if device.type != "cuda":
            raise RuntimeError("LipForcing's configured InsightFace detector requires a CUDA ComfyUI device.")

        cache_key = (str(root), str(mask_path), str(device), current_key)
        with _LIPFORCING_PROCESSOR_LOCK:
            processor = _LIPFORCING_PROCESSORS.get(cache_key)
            if processor is None:
                _LIPFORCING_PROCESSORS.clear()
                import sys
                if str(root) not in sys.path:
                    sys.path.insert(0, str(root))
                from OmniAvatar.utils.latentsync.image_processor import ImageProcessor, load_fixed_mask

                module_path = Path(sys.modules[ImageProcessor.__module__].__file__).resolve()
                if os.path.commonpath((str(root), str(module_path))) != str(root):
                    raise RuntimeError("A different OmniAvatar package is shadowing the configured LipForcing source.")
                processor = ImageProcessor(
                    resolution=512,
                    device=str(device),
                    mask_image=load_fixed_mask(512, mask_image_path=str(mask_path)),
                    insightface_root=str(root / "checkpoints" / "auxiliary"),
                )
                _LIPFORCING_PROCESSORS[cache_key] = processor

            # LipForcing smooths its affine transform sequentially. Reset between
            # jobs, but keep that exact within-clip behavior for this batch.
            processor.restorer.p_bias = None
            keep_mask = processor.mask_image[0].detach().cpu().numpy().astype(np.float32)
            face_width, face_height = processor.restorer.face_size
            scale_x, scale_y = 512.0 / face_width, 512.0 / face_height
            scale = np.array([
                # Match OpenCV resize's half-pixel-center convention used by
                # ImageProcessor when it resizes the affine crop to 512 square.
                [scale_x, 0.0, 0.5 * scale_x - 0.5],
                [0.0, scale_y, 0.5 * scale_y - 0.5],
                [0.0, 0.0, 1.0],
            ], dtype=np.float32)
            masks = []
            for frame_index, frame in enumerate(images):
                management.throw_exception_if_processing_interrupted()
                rgb = (frame[..., :3].detach().clamp(0, 1).cpu().numpy() * 255.0).round().astype(np.uint8)
                try:
                    _, _, affine = processor.affine_transform(rgb)
                except RuntimeError as exc:
                    raise RuntimeError(f"LipForcing face detection failed on tracked frame {frame_index}: {exc}") from exc
                source_to_aligned = affine.detach().cpu().numpy().reshape(2, 3).astype(np.float32)
                source_to_aligned = (scale @ np.vstack((source_to_aligned, [0.0, 0.0, 1.0])))[:2]

                # The upstream mask is 1=preserve and 0=regenerate. Warp it back
                # to the exact H3 crop coordinates; outside the aligned face is
                # preserved, so it cannot accidentally expand the repair area.
                keep_in_crop = cv2.warpAffine(
                    keep_mask,
                    source_to_aligned,
                    (512, 512),
                    flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=1.0,
                )
                regenerate = np.clip(1.0 - keep_in_crop, 0.0, 1.0)
                if not np.isfinite(regenerate).all() or float(regenerate.max()) < 0.01:
                    raise RuntimeError(f"LipForcing produced an empty mouth mask on tracked frame {frame_index}.")
                masks.append(regenerate)

            mask_batch = torch.from_numpy(np.stack(masks, axis=0).astype(np.float32))
            print(
                f"[Audio Retake Studio] LipForcing mask prepared: {len(masks)} tracked frames, "
                f"template={mask_path.name}, config={current_key[:12]}"
            )
            return (mask_batch,)


class AudioRetakeTranscribe:
    """Transcribe speech with word timings, using a local Whisper folder."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"audio": ("AUDIO",), "language": ("STRING", {"default": "", "tooltip": "Language name or code. Leave empty to detect it."})}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "transcribe"
    OUTPUT_NODE = True
    CATEGORY = "Audio Retake Studio"

    def transcribe(self, audio, language):
        import torch
        import comfy.model_management as management
        from .transcribe import find_model, transcribe_words
        device = management.get_torch_device()
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        words = transcribe_words(audio["waveform"], audio["sample_rate"], find_model(folder_paths.models_dir), language.strip().lower(), device, dtype)
        management.soft_empty_cache()
        text = " ".join(word["word"] for word in words)
        return {"ui": {"transcript": [json.dumps(words, ensure_ascii=False)]}, "result": (text,)}


NODE_CLASS_MAPPINGS = {"AudioRetakePikaSampler": AudioRetakePikaSampler, "AudioRetakeTimedMask": AudioRetakeTimedMask, "AudioRetakeLipForcingMask": AudioRetakeLipForcingMask, "AudioRetakeTranscribe": AudioRetakeTranscribe}
NODE_DISPLAY_NAME_MAPPINGS = {"AudioRetakePikaSampler": "Audio Retake - Pika H3 Flex (experimental)", "AudioRetakeTimedMask": "Audio Retake - Timed AV Mask", "AudioRetakeLipForcingMask": "Audio Retake - LipForcing Mouth Mask", "AudioRetakeTranscribe": "Audio Retake - Transcribe Words"}
