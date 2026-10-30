"""CPU tests for visualization semantics, independent of the generated pixels."""
import numpy as np
from scipy.spatial.transform import Rotation

from tools.visualize_h3_controls import (
    constant_controls, load_cosmos_drawer, numeric_controls, prompt_banner, text_is_used,
)


def test_text_flag_uses_actual_dit_flag_not_presence_of_source_prompt():
    meta={'prompt':'A tank moves backward.','text_condition_used':True,'scene_or_subject_text_used':False}
    assert not text_is_used(meta)
    _,prompt=prompt_banner(meta,832)
    assert prompt==''
    assert not text_is_used({'text_condition_used':'False'})


def test_up_pitch_uses_negative_x_rotation_and_left_yaw_negative_y():
    camera=np.repeat(np.eye(4)[None],37,axis=0)
    camera[:,:3,:3]=Rotation.from_euler('x',np.linspace(0,-15,37)[:,None],degrees=True).as_matrix()
    states=numeric_controls(camera,124)
    assert states[50]['pitch']>0
    assert abs(states[50]['yaw'])<1e-6
    camera[:,:3,:3]=Rotation.from_euler('y',np.linspace(0,-15,37)[:,None],degrees=True).as_matrix()
    assert numeric_controls(camera,124)[50]['yaw']<0


def test_constant_camera_key_is_not_subject_motion():
    assert constant_controls('A',124)[0]['keys']==['A']
    assert constant_controls('I',124)[0]['pitch']>0
    assert constant_controls('K',124)[0]['pitch']<0


def test_actual_cosmos_renderer_works_without_importing_mediapy():
    renderer=load_cosmos_drawer()
    frame=np.zeros((480,832,3),np.uint8)
    drawn=renderer(frame,{'W'},{'pitch_delta':0,'yaw_delta':0},panel_label='GEN',control_label='INPUT numeric C2W')
    assert drawn.shape==frame.shape
    assert bool(drawn.any())
    assert not bool(frame.any())
