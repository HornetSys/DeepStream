# Command Templates

Run these on the target DeepStream Linux machine or inside the DeepStream container, not on a Windows laptop unless DeepStream/GStreamer is installed there. Replace placeholders and redact credentials before sharing output.

## Basic Environment

```bash
deepstream-app --version-all
gst-launch-1.0 --version
gst-inspect-1.0 rtspsrc
gst-inspect-1.0 nvv4l2decoder
```

Jetson:

```bash
cat /proc/device-tree/model
cat /etc/nv_tegra_release
tegrastats
```

dGPU:

```bash
nvidia-smi
nvidia-smi dmon -s pucvmet -d 1
```

RTSP connection and network signals:

```bash
ss -tan
ip -s link
```

## RTSP Session Only

Use this first to verify that the camera session itself stays healthy.

```bash
export RTSP_URL='rtsp://camera.example/stream'
GST_DEBUG=2,rtspsrc:6,rtpjitterbuffer:6 \
gst-launch-1.0 -e rtspsrc location="$RTSP_URL" latency=300 drop-on-latency=true ! fakesink sync=false
```

Compare TCP-only:

```bash
GST_DEBUG=2,rtspsrc:6,rtpjitterbuffer:6 \
gst-launch-1.0 -e rtspsrc location="$RTSP_URL" protocols=tcp latency=300 drop-on-latency=true ! fakesink sync=false
```

If UDP is required, inspect NIC drops while the test runs:

```bash
ip -s link
ethtool -S eth0
ping -c 50 camera.example
tracepath camera.example
```

## Codec-Specific Decode Tests

H.264:

```bash
GST_DEBUG=2,rtspsrc:5,rtpjitterbuffer:5,nvv4l2decoder:4 \
gst-launch-1.0 -e rtspsrc location="$RTSP_URL" protocols=tcp latency=300 drop-on-latency=true \
  ! rtph264depay ! h264parse ! nvv4l2decoder num-extra-surfaces=4 \
  ! fakesink sync=false
```

H.265:

```bash
GST_DEBUG=2,rtspsrc:5,rtpjitterbuffer:5,nvv4l2decoder:4 \
gst-launch-1.0 -e rtspsrc location="$RTSP_URL" protocols=tcp latency=300 drop-on-latency=true \
  ! rtph265depay ! h265parse ! nvv4l2decoder num-extra-surfaces=4 \
  ! fakesink sync=false
```

If decode is the bottleneck and frame loss is acceptable, test one load-shedding knob at a time:

```bash
gst-launch-1.0 -e rtspsrc location="$RTSP_URL" protocols=tcp latency=300 drop-on-latency=true \
  ! rtph264depay ! h264parse ! nvv4l2decoder drop-frame-interval=2 \
  ! fakesink sync=false
```

Skip-frame diagnostic (verify the property on the installed decoder first):

```bash
gst-launch-1.0 -e rtspsrc location="$RTSP_URL" protocols=tcp latency=300 drop-on-latency=true \
  ! rtph264depay ! h264parse ! nvv4l2decoder skip-frames=1 \
  ! fakesink sync=false
```

## DeepStream Latency Measurement

```bash
export NVDS_ENABLE_LATENCY_MEASUREMENT=1
export NVDS_ENABLE_COMPONENT_LATENCY_MEASUREMENT=1
deepstream-app -c config.txt
```

Before sharing logs, redact RTSP credentials, tokens, private hostnames, customer names, and internal registry paths.

## Isolation With Non-RTSP Sources

Use a file at similar resolution/FPS to remove the network/camera from the test. Match the parser to the file codec.

```bash
gst-launch-1.0 -e filesrc location=sample.mp4 ! qtdemux ! h264parse ! nvv4l2decoder ! fakesink sync=false
```

Use `videotestsrc` to remove encoded input and decoder behavior:

```bash
gst-launch-1.0 -e videotestsrc is-live=true ! video/x-raw,width=1920,height=1080,framerate=30/1 \
  ! nvvideoconvert ! 'video/x-raw(memory:NVMM),format=NV12' \
  ! fakesink sync=false
```

## Optional Packet Capture

Packet capture can expose credentials, camera IPs, and customer network details. It is client-mediated only: ask for explicit approval, provide the command for the user to run, and do not execute it autonomously. Keep captures local unless the user confirms what may be shared.

```bash
tcpdump -i eth0 host camera.example and port 554 -w rtsp-debug.pcap
```
