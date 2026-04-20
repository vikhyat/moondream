import modal
import subprocess

app = modal.App("moondream")

# This will create a container image on Modal Labs and install Moondream Station.
image = modal.Image.from_registry("ubuntu:22.04", add_python="3.11").run_commands(
    [
        "apt-get update && apt-get install -y curl",
        "curl -fsSL https://depot.moondream.ai/station/install.sh | bash",
    ]
)


@app.function(
    image=image,
    memory=4096,  # Change your alloted memory here.
    gpu="L4",  # Change your GPU here.
    timeout=86400,
    # Scaling parameters.
    min_containers=1,  # Control the minimum containers which are warm at any time.
    max_containers=1,  # The max number of containers which your app can scale up to.
    scaledown_window=300,  # The max number of seconds containers remain idle before scaling down.
)

# This exposes the port Moondream Station is running on.
@modal.web_server(2020, startup_timeout=300.0)
def server():
    subprocess.Popen(["/moondream_station"])
