# Gardefender3000
AI has its place in society and everyone has their own needs.
I am using AI to keep squirrels out of my garden via a home tower defense turret.

The Gardefender3000 detects 80 different classes (Microsoft COCO model) at 30 frames per second using a Raspberry Pi 5. A water gun powered by an electric motor mounted on a pan/tilt servo assembly aims at the detected class. The squirrels never saw it coming... they should have invented AI first.

The Raspberry Pi hosts a server that is available through port forwarding. This allows the video to be streamed to a dashboard, where I can take manual control to check on the garden remotely. The server also allows me to water my plants remotely using dripline irrigation activated with a sprinkler valve that is also connected to the Pi.

![squirrel_jump](https://github.com/user-attachments/assets/4f990f4d-69c1-4592-8ab3-4e8363edf906)
