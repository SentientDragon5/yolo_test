# yolo_test

## Yolo
following this [tutorial](https://docs.ultralytics.com/quickstart/)

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python3 tests/yolo_simple.py
```

## Troubleshooting

To check if opencv is working, run 
```bash
python tests/opencv.py
```

To check if cameras are showing up, `sudo apt install v4l-utils` then run: 
```bash
v4l2-ctl --list-devices 
```

### WSL

WSL struggles to connect to cameras on the native hardware.
https://github.com/PINTO0309/wsl2_linux_kernel_usbcam_enable_conf?tab=readme-ov-file
