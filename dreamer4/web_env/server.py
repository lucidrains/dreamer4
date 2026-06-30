import os
import json
import base64
import io

import torch
import numpy as np

from http.server import BaseHTTPRequestHandler, HTTPServer
from einops import repeat, rearrange
import imageio.v3 as iio

def _tensor_or_array_to_b64_png(obs):
    img_data = obs['image'] if isinstance(obs, dict) else obs

    if torch.is_tensor(img_data):
        img_data = img_data.detach().cpu().numpy()

    if img_data.ndim == 4:
        img_data = img_data[0]

    if img_data.dtype in (np.float32, np.float64):
        img_data = (np.clip(img_data, 0.0, 1.0) * 255).astype(np.uint8)

    if img_data.shape[0] == 3:
        img_data = rearrange(img_data, 'c h w -> h w c')

    img_data = repeat(img_data, 'h w c -> (h p1) (w p2) c', p1 = 16, p2 = 16)

    img_bytes = iio.imwrite('<bytes>', img_data, extension = '.png')
    return base64.b64encode(img_bytes).decode("utf-8")

class WebEnvServer:
    def __init__(self, env, port=8000):
        self.env = env
        self.port = port
        self.html_path = os.path.join(os.path.dirname(__file__), 'index.html')
        self.is_batched = hasattr(self.env, 'model') or hasattr(self.env, 'world_model') # Usually indicates DynamicsWorldModelWrapper

    def serve(self):
        env_ref = self.env
        is_batched = self.is_batched
        html_path = self.html_path

        def get_steps_left():
            if hasattr(env_ref, 'max_steps') and hasattr(env_ref, 'steps'):
                return env_ref.max_steps - env_ref.steps
            elif hasattr(env_ref, 'env') and hasattr(env_ref.env, 'max_steps') and hasattr(env_ref.env, 'steps'):
                return env_ref.env.max_steps - env_ref.env.steps
            return None

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == '/':
                    self.send_response(200)
                    self.send_header('Content-type', 'text/html')
                    self.end_headers()
                    with open(html_path, 'rb') as f:
                        self.wfile.write(f.read())
                else:
                    self.send_response(404)
                    self.end_headers()

            def do_POST(self):
                if self.path == '/reset':
                    if is_batched:
                        obs, _ = env_ref.reset(batch_size=1)
                    else:
                        obs_tuple = env_ref.reset()
                        obs = obs_tuple[0] if isinstance(obs_tuple, tuple) and len(obs_tuple) == 2 and isinstance(obs_tuple[1], dict) else obs_tuple

                    b64 = _tensor_or_array_to_b64_png(obs)

                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({'image': b64, 'steps_left': get_steps_left()}).encode())

                elif self.path == '/step':
                    content_length = int(self.headers['Content-Length'])
                    post_data = self.rfile.read(content_length)
                    body = json.loads(post_data.decode('utf-8'))
                    action_val = body.get('action', 0)

                    if is_batched:
                        import torch
                        action = torch.tensor([action_val], dtype=torch.long)
                        obs, reward, terminated, truncated, _ = env_ref.step(action)
                        reward_val = reward.item() if hasattr(reward, 'item') else float(reward)
                        term_val = bool(terminated.item()) if hasattr(terminated, 'item') else bool(terminated)
                        trunc_val = bool(truncated.item()) if hasattr(truncated, 'item') else bool(truncated)
                    else:
                        res = env_ref.step(action_val)
                        if len(res) == 5:
                            obs, reward, terminated, truncated, _ = res
                            term_val = bool(terminated)
                            trunc_val = bool(truncated)
                        else:
                            obs, reward, done, _ = res
                            term_val = bool(done)
                            trunc_val = False

                        reward_val = float(reward)

                    b64 = _tensor_or_array_to_b64_png(obs)

                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({
                        'image': b64,
                        'reward': reward_val,
                        'terminated': term_val,
                        'truncated': trunc_val,
                        'done': term_val or trunc_val,
                        'steps_left': get_steps_left()
                    }).encode())
                else:
                    self.send_response(404)
                    self.end_headers()

            def log_message(self, format, *args):
                pass

        httpd = HTTPServer(('', self.port), Handler)

        if self.is_batched:
            print(f"Open browser to interact with world model: http://localhost:{self.port}")
        else:
            print(f"Open browser to interact with env: http://localhost:{self.port}")

        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server.")
            httpd.server_close()
