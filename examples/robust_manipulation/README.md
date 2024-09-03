The error happens

`mujoco.FatalError: Offscreen framebuffer is not complete, error 0x8cdd`

use 

`export MUJOCO_GL="osmesa"`

The is from the [link](https://github.com/google-deepmind/dm_control/issues/76)
