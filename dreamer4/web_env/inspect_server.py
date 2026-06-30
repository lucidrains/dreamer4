import json
import base64
import webbrowser
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import imageio.v3 as iio
from einops import rearrange, repeat

from memmap_replay_buffer import ReplayBuffer

def exists(val):
    return val is not None

def array_to_b64_png(img_data):
    if hasattr(img_data, 'numpy'):
        img_data = img_data.numpy()

    if img_data.ndim == 4:
        img_data = img_data[0]

    if img_data.dtype in (np.float32, np.float64):
        img_data = (np.clip(img_data, 0., 1.) * 255).astype(np.uint8)

    if img_data.shape[0] == 3:
        img_data = rearrange(img_data, 'c h w -> h w c')

    if img_data.shape[:2] == (16, 16):
        img_data = repeat(img_data, 'h w c -> (h p1) (w p2) c', p1 = 16, p2 = 16)
    elif img_data.shape[0] < 64:
        img_data = repeat(img_data, 'h w c -> (h p1) (w p2) c', p1 = 4, p2 = 4)

    img_bytes = iio.imwrite('<bytes>', img_data, extension = '.png')
    return base64.b64encode(img_bytes).decode('utf-8')

class InspectRequestHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def send_json(self, data, status = 200):
        self.send_response(status)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    @property
    def app(self):
        return self.server.app

    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(self.app.html_path.read_bytes())
            return

        if self.path == '/api/stats':
            return self.send_json(dict(
                num_episodes = self.app.num_episodes,
                fields = self.app.fields,
                schema = self.app.schema
            ))

        if self.path == '/api/episodes':
            episodes_info = []
            for i in range(self.app.num_episodes):
                ep_data = self.app.dataset[i]
                length = int(ep_data.get('_lens', len(ep_data.get('rewards', []))))

                cum_reward = None
                total_apples = None
                if 'rewards' in ep_data:
                    rewards = ep_data['rewards'][:length]
                    if hasattr(rewards, 'numpy'): rewards = rewards.numpy()
                    cum_reward = float(np.sum(rewards))
                    total_apples = float(np.sum(rewards > 0.0))

                episodes_info.append(dict(
                    id = i,
                    length = length,
                    cum_reward = cum_reward,
                    total_apples = total_apples
                ))

            return self.send_json(dict(episodes = episodes_info))

        if self.path.startswith('/api/episode/'):
            try:
                ep_id = int(self.path.split('/')[-1])
                if not (0 <= ep_id < self.app.num_episodes):
                    return self.send_json(dict(error = "Not found"), 404)

                ep_data = self.app.dataset[ep_id]
                length = int(ep_data.get('_lens', len(ep_data.get('video', ep_data.get('rewards', [])))))

                frames = []
                for i in range(length):
                    frame_data = {}
                    for field in self.app.fields:
                        if field not in ep_data: continue

                        is_meta = hasattr(self.app.replay_buffer, 'meta_fieldnames') and field in self.app.replay_buffer.meta_fieldnames

                        val = ep_data[field]
                        if not is_meta:
                            val = val[i]

                        if field == 'video':
                            frame_data[field] = array_to_b64_png(val)
                        else:
                            frame_data[field] = val.item() if hasattr(val, 'item') else val

                    frames.append(frame_data)

                return self.send_json(dict(id = ep_id, length = length, frames = frames))
            except Exception as e:
                print(f"Error loading episode: {e}")
                return self.send_json(dict(error = str(e)), 500)

        self.send_response(404)
        self.end_headers()

class InspectReplayBufferServer:
    def __init__(self, buffer_path, port = 8081):
        self.port = port
        self.html_path = Path(__file__).parent / 'inspect_index.html'

        self.replay_buffer = ReplayBuffer.from_folder(buffer_path)
        self.dataset = self.replay_buffer.dataset(slice_by_episode_len = True)
        self.num_episodes = len(self.dataset)

        self.fields = []
        self.schema = {}
        if self.num_episodes > 0:
            first = self.dataset[0]
            if isinstance(first, dict):
                for k, v in first.items():
                    if k == '_lens': continue
                    self.fields.append(k)
                    is_meta = hasattr(self.replay_buffer, 'meta_fieldnames') and k in self.replay_buffer.meta_fieldnames
                    shape = list(v.shape) if hasattr(v, 'shape') else []
                    if not is_meta and len(shape) > 0:
                        shape = shape[1:]

                    self.schema[k] = dict(
                        type = str(v.dtype) if hasattr(v, 'dtype') else type(v).__name__,
                        shape = shape,
                        is_meta = is_meta
                    )
            else:
                self.fields = ['video']
                v = first
                self.schema['video'] = dict(
                    type = str(v.dtype) if hasattr(v, 'dtype') else type(v).__name__,
                    shape = list(v.shape) if hasattr(v, 'shape') else []
                )

        print(f"Loaded {self.num_episodes} episodes. Fields: {self.fields}")

    def serve(self):
        httpd = HTTPServer(('', self.port), InspectRequestHandler)
        httpd.app = self

        url = f"http://localhost:{self.port}"
        print(f"Open browser to inspect replay buffer: {url}")
        webbrowser.open(url)

        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down inspector server.")
            httpd.server_close()
