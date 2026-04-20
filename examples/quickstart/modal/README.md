# Deploy Moondream Station on Modal Labs

Get Moondream Station running in the cloud with Modal Labs—no hassle, just fast, scalable AI you can access from anywhere. This quick guide shows you how.

## Setting Up

### **1. Create Modal account**
Go to [modal.com](https://modal.com) to sign up.

### **2. Install Modal Labs Python Library**:
To access Modal Labs deployment tools, you need to install the library:

   ```bash
   pip install modal
   ```

### **3. Login**
   To connect your computer to your Modal Labs account, run:
   ```bash
   modal token new
   ```
   This will open an authorization page in your web browser where you can authenticate your system with a login token.

### **4. Launch a Moondream Station Instance**

#### **1. Deploying your Instance**:

To launch Moondream Station on Modal Labs, first make sure you're in the correct repository:

```
cd quickstart/modal
```
You have two deployment options:

**A) Production Deployment**

   ```bash
   modal deploy deploy_moondream.py
   ```
This creates a permanent deployment on Modal Labs that continues running even after you close your terminal.


**B)  Development Mode**

For development and testing with live logs in terminal:

```
modal serve deploy_moondream.py
```
This runs the app while streaming logs directly to your terminal. The deployment stops when you terminate the command `Ctrl+C`.

#### **2. Get your URL**
Modal will show your server URL at the start of execution. Make sure to copy this URL as you will need this to access the Moondream Station instance.

![alt text](../../images/modal/example-image.png)

It may look something like:
   ```
   https://<username>--moondream-server-dev.modal.run
   ```
   or
   ```
   https://<username>--moondream-server.modal.run
   ```
   This is your permanent service endpoint.

#### **4. Stopping your Instance**:

To stop your instance, run:

```
modal app stop moondream
```

or if you ran it with `modal serve` you can exit by inputting `CTRL+C` on your keyboard.

You may also locate and stop the app from your Modal Labs dashboard.


## Inference

### Using Moondream Station CLI
To use Moondream Station from the Moondream Station CLI, launch Moondream Station locally and then set the inference URL to the endpoint. **Make sure to add `/v1` to the end of the url that you copied.**

```
moondream> admin set-inference-url https://<username>--moondream-server-dev.modal.run/v1
```

Then continue to use Moondream Station normally:

```
moondream> query "What's in this image?" path/to/image.jpg
```

### Using Python Client
You can also access the Modal Labs endpoint using our Python client.

You will need to install the required libraries first:

```
pip install moondream pillow
```

Then, you can use the call the python client. Here's an example of what that may look like:

```python
# modal_infer.py
import moondream as md
from PIL import Image

# To use with the example code, make sure to add /v1 at the end.
model = md.vl(endpoint="https://<username>--moondream-server-dev.modal.run/v1")
image = Image.open("/path/to/image")

# query
answer = model.query(image, "What's in this image?")["answer"]
print(answer)
```

Check out the python demo for more examples [here](modal_infer.py).
