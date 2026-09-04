# Moondream Python Quickstart

Get started with Moondream's vision AI in Python in minutes.

## Setup

1. Choose your deployment:

   - **Local**: Install [Moondream Station](https://moondream.ai/station)
   - **Cloud**: Get an API key from [Moondream Cloud](https://moondream.ai/cloud)

### Accessing Station over LAN

If Moondream Station runs on another computer on your local network,
replace `localhost` with that machine's LAN IP address (Station serves
port 2020, same as the default `http://localhost:2020/v1` in `main.py`):

```python
import moondream as md

# <station-ip> is the LAN IP of the machine running Station,
# e.g. "http://192.168.1.10:2020/v1"
model = md.vl(endpoint="http://<station-ip>:2020/v1")
```

To keep the address out of code, read it from the environment:

```shell
export MOONDREAM_ENDPOINT="http://<station-ip>:2020/v1"
```

```python
import os
import moondream as md

model = md.vl(endpoint=os.environ.get("MOONDREAM_ENDPOINT"))
```

Make sure both machines are on the same network and the Station host's
firewall allows inbound connections on port 2020. The Station host must
also listen on its LAN interface — community members report setting
`service_host` in the Station `config.json` (e.g.
`~/.moondream-station/config.json` on macOS) works.

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run Moondream Station locally or tweak the [main.py](quickstart/python/main.py) or run on the cloud using an [API key](https://moondream.ai/c/cloud/api-keys).

4. Run the example to see Moondream in action!

   ```bash
   python main.py
   ```

5. Try a full tutorial:

   Open the `detect_cars.ipynb` notebook in Jupyter and run all the cells to try caption, query, detect and point capabilities:

   ```bash
   jupyter notebook detect_cars.ipynb
   ```