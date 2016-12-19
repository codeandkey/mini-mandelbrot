#pragma once

/* holds some shaders */
#define GLSL(x) "#version 120\n" #x

const char* vs_passthrough = GLSL(
	attribute vec2 position;
	attribute vec2 texcoord;

	varying vec2 fs_texcoord;

	void main(void) {
		fs_texcoord = texcoord;
		gl_Position = vec4(position, 0.0f, 1.0f);
	}
);

const char* fs_passthrough = GLSL(
	varying vec2 fs_texcoord;
	uniform sampler2D fs_texture;

	void main(void) {
		gl_FragColor = texture2D(fs_texture, fs_texcoord);
	}
);
