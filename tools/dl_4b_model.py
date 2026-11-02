"""dl_4b_model.py - download sorokin/qwen3_4b_grids15_sft139 (Transformers/bfloat16/1)."""
import os
import urllib.request

from kagglesdk.models.types.model_api_service import ApiDownloadModelInstanceVersionRequest
from kagglesdk.models.types.model_enums import ModelFramework
from kaggle.api.kaggle_api_extended import KaggleApi

DEST = "/home/taizo/Kaggle/models/qwen3_4b_sft"

api = KaggleApi()
api.authenticate()
with api.build_kaggle_client() as k:
    svc = k.models.model_api_client
    req = ApiDownloadModelInstanceVersionRequest()
    req.owner_slug = "sorokin"
    req.model_slug = "qwen3_4b_grids15_sft139"
    req.framework = ModelFramework.MODEL_FRAMEWORK_TRANSFORMERS
    req.instance_slug = "bfloat16"
    req.version_number = 1
    redir = svc.download_model_instance_version(req)
    url = getattr(redir, "redirect_url", None) or getattr(redir, "url", None)
    os.makedirs(DEST, exist_ok=True)
    out = os.path.join(DEST, "model.tar.gz")
    print("downloading to", out, flush=True)
    urllib.request.urlretrieve(url, out)
    print("DONE", out, flush=True)
