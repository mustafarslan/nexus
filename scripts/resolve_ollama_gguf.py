#!/usr/bin/env python3
import sys
import os
import json
import urllib.request

def resolve_model_path(model_name="gemma4:31b-cloud"):
    home = os.path.expanduser("~")
    
    # 1. Try to query the local Ollama API
    url = "http://localhost:11434/api/show"
    data = json.dumps({"name": model_name}).encode('utf-8')
    req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
    
    resolved_model = model_name
    try:
        with urllib.request.urlopen(req) as response:
            res = json.loads(response.read().decode('utf-8'))
            parent = res.get("details", {}).get("parent_model")
            if parent:
                resolved_model = parent
    except Exception as e:
        sys.stderr.write(f"Ollama API query failed: {e}. Falling back to disk manifest scanning.\n")
        # If API failed, we check if we can read the cloud model manifest directly from disk
        try:
            cloud_manifest = os.path.join(home, ".ollama", "models", "manifests", "registry.ollama.ai", "library", "gemma4", "31b-cloud")
            if os.path.exists(cloud_manifest):
                with open(cloud_manifest, "r") as f:
                    manifest_data = json.load(f)
                    config_digest = manifest_data.get("config", {}).get("digest", "")
                    if config_digest.startswith("sha256:"):
                        config_digest = config_digest.replace("sha256:", "sha256-")
                    config_path = os.path.join(home, ".ollama", "models", "blobs", config_digest)
                    if os.path.exists(config_path):
                        with open(config_path, "r") as cf:
                            config_json = json.load(cf)
                            # Fallback: if we see parent details
                            pass
        except Exception as ex:
            sys.stderr.write(f"Disk manifest config read failed: {ex}\n")
            
    # Normalize model name: e.g. "gemma4:31b" -> namespace="library", name="gemma4", tag="31b"
    if ":" in resolved_model:
        name_part, tag = resolved_model.split(":", 1)
    else:
        name_part = resolved_model
        tag = "latest"
        
    if "/" in name_part:
        namespace, name = name_part.split("/", 1)
    else:
        namespace = "library"
        name = name_part
        
    manifest_path = os.path.join(
        home, ".ollama", "models", "manifests", "registry.ollama.ai", namespace, name, tag
    )
    
    if not os.path.exists(manifest_path):
        # Scan manifest dir for fallback matches
        manifests_root = os.path.join(home, ".ollama", "models", "manifests")
        for root, dirs, files in os.walk(manifests_root):
            for file in files:
                if name in root and tag in file:
                    manifest_path = os.path.join(root, file)
                    break
            if manifest_path and os.path.exists(manifest_path) and not os.path.isdir(manifest_path):
                break
                
    if os.path.exists(manifest_path) and not os.path.isdir(manifest_path):
        try:
            with open(manifest_path, "r") as f:
                manifest = json.load(f)
                for layer in manifest.get("layers", []):
                    media_type = layer.get("mediaType", "")
                    if "model" in media_type or layer.get("size", 0) > 1024 * 1024 * 1024:
                        digest = layer.get("digest", "")
                        if digest.startswith("sha256:"):
                            digest = digest.replace("sha256:", "sha256-")
                        blob_path = os.path.join(home, ".ollama", "models", "blobs", digest)
                        if os.path.exists(blob_path):
                            return blob_path
        except Exception as e:
            sys.stderr.write(f"Error parsing manifest {manifest_path}: {e}\n")
            
    # Final fallback: locate largest blob in ~/.ollama/models/blobs/
    try:
        blobs_dir = os.path.join(home, ".ollama", "models", "blobs")
        if os.path.exists(blobs_dir):
            largest_file = None
            largest_size = 0
            for file in os.listdir(blobs_dir):
                filepath = os.path.join(blobs_dir, file)
                if os.path.isfile(filepath):
                    size = os.path.getsize(filepath)
                    if size > largest_size:
                        largest_size = size
                        largest_file = filepath
            if largest_file:
                return largest_file
    except Exception as e:
        sys.stderr.write(f"Final fallback failed: {e}\n")
        
    return ""

if __name__ == "__main__":
    path = resolve_model_path()
    if path:
        print(path)
        sys.exit(0)
    else:
        sys.stderr.write("Failed to resolve model path\n")
        sys.exit(1)
