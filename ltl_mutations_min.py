import numpy as np
from glumpy import app, gl, glm, gloo


render_vertex = """
attribute vec2 position;
attribute vec2 texcoord;
varying vec2 v_texcoord;
void main()
{
    gl_Position = vec4(position, 0.0, 1.0);
    v_texcoord = texcoord;
}
"""

render_fragment = """
uniform int pingpong;
uniform sampler2D texture;
varying vec2 v_texcoord;
void main()
{
    float v;
    v = texture2D(texture, v_texcoord)[pingpong];
    gl_FragColor = vec4(1.0-v, 1.0-v, 1.0-v, 1.0);
}
"""

compute_vertex = """
attribute vec2 position;
attribute vec2 texcoord;
varying vec2 v_texcoord;
void main()
{
    gl_Position = vec4(position, 0.0, 1.0);
    v_texcoord = texcoord;
}
"""

compute_fragment = """
uniform int pingpong;
uniform vec3 params;
uniform vec3 radii;
uniform sampler2D texture;
uniform float dx;          // horizontal distance between texels
uniform float dy;          // vertical distance between texels
varying vec2 v_texcoord;

float adj_count(float radius, vec2 p) { 
    float i, j;
    float count;
    count = 0;
    
    for (i = -radius; i <= radius; i++) {
        for (j = -radius; j <= radius; j++) {
            count = count + texture2D(texture, p + vec2(i*dx,j*dy))[pingpong];
        }
    }
    return count;
}

void main(void)
{
    vec2  p = v_texcoord;
    float old_state, new_state, count;

    old_state = texture2D(texture, p)[pingpong];

    count = adj_count(radii[0], p);

    new_state = old_state;
    
    if( count >= 0 && count < 121.0*params[0] )
        new_state = 0.0;
    
    else if( (count >= 121.0*params[0]) && (count <= 121.0*params[1]) )
        new_state = 1.0;

    else if( count >= 121.0*params[2] && count <= 121.0)
        new_state = 0.0;
    
    if( pingpong == 0 ) {
        gl_FragColor[1] = new_state;
        gl_FragColor[0] = old_state;
    } else {
        gl_FragColor[0] = new_state;
        gl_FragColor[1] = old_state;
    }
}
"""
WIDTH= 2048
HEIGHT = 1024
MUTATION_STEP = 0.01
window = app.Window(width=WIDTH, height=HEIGHT)

@window.event
def on_draw(dt):
    global pingpong

    width,height = WIDTH, HEIGHT

    pingpong = 1 - pingpong
    compute["pingpong"] = pingpong
    render["pingpong"] = pingpong

    gl.glDisable(gl.GL_BLEND)

    framebuffer.activate()
    gl.glViewport(0, 0, width, height)
    compute.draw(gl.GL_TRIANGLE_STRIP)
    framebuffer.deactivate()

    gl.glClear(gl.GL_COLOR_BUFFER_BIT)
    gl.glViewport(0, 0, width, height)
    render.draw(gl.GL_TRIANGLE_STRIP)                 

w, h = WIDTH,HEIGHT
Z = np.zeros((h, w, 4), dtype=np.float32)
Z[...] = np.random.rand(h, w, 4,)


pingpong = 1
compute = gloo.Program(compute_vertex, compute_fragment, count=4)
compute["texture"] = Z
compute["texture"].interpolation = gl.GL_NEAREST
compute["texture"].wrapping = gl.GL_CLAMP_TO_EDGE
compute["position"] = [(-1, -1), (-1, +1), (+1, -1), (+1, +1)]
compute["texcoord"] = [(0, 0), (0, 1), (1, 0), (1, 1)]
compute['dx'] = 1.0 / w
compute['dy'] = 1.0 / h
compute['pingpong'] = pingpong
compute['params'] = np.array([0.2801, 0.3719, 0.4793 ], dtype=np.float32)
compute['radii'] = np.array([5.0, 3.0, 1.0], dtype=np.float32)


render = gloo.Program(render_vertex, render_fragment, count=4)
render["position"] = [(-1, -1), (-1, +1), (+1, -1), (+1, +1)]
render["texcoord"] = [(0, 0), (0, 1), (1, 0), (1, 1)]
render["texture"] = compute["texture"]
render["texture"].interpolation = gl.GL_LINEAR
render["texture"].wrapping = gl.GL_CLAMP_TO_EDGE
render['pingpong'] = pingpong

framebuffer = gloo.FrameBuffer(color=compute["texture"],
                               depth=gloo.DepthBuffer(w, h))
app.run(framerate=60)

# -----------------------------------------------------------------------------
# Copyright (c) 2009-2016 Nicolas P. Rougier. All rights reserved.
# Distributed under the (new) BSD License.
# -----------------------------------------------------------------------------



# @window.event
# def on_draw(dt):
#     pass

# app.run(interactive=True)


