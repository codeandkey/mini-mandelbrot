/*
 * mini-mandelbrot : multithreaded mandelbrot renderer
 * the screen is recursively divided in a quadtree-like manner to divide the work between threads
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#include <gmp.h>
#include <mpfr.h>

#include <signal.h>
#include <pthread.h>

#include <GLXW/glxw.h>
#include <GLFW/glfw3.h>

#include "shaders.h"

/* window parameters */

#define WIDTH 1366
#define HEIGHT 768
#define TITLE "mandelbrot"
#define FS 1

/* mandelbrot generation parameters */

#define MBR_MAX_ITERATIONS 256
#define MBR_DIVERGE_THRESHOLD 4

#define BOUND_LEFT -2.5
#define BOUND_RIGHT 1
#define BOUND_TOP 1
#define BOUND_BOTTOM -1

#define PBITS 128

/* thread and calculation parameters */

#define THR_MAX_ACTIVE 4 /* maximum number of active CPU threads created to deal with each subdivision */
#define SUBDIV_MIN_SIZE 50 /* smallest area for a subdivision */

/* types */

typedef struct _pixel {
	uint8_t r, g, b, a;
} pixel;

typedef struct _mandelbrot_params {
	int left, right, top, bottom, thr_index;
} mandelbrot_params;

typedef struct _img {
	mpfr_t r, i;
} img;

/* globals */

GLFWwindow* win;
int r;
unsigned tex, vs, fs, prg;
mpfr_t bound_left, bound_right, bound_top, bound_bottom;

pixel pixbuf[WIDTH * HEIGHT];
pthread_mutex_t pixbuf_mutex = PTHREAD_MUTEX_INITIALIZER;

pthread_t threads[THR_MAX_ACTIVE];
int live_threads[THR_MAX_ACTIVE]; /* binary flag array signifying which thread slots are available */
int num_live_threads;
pthread_mutex_t live_threads_mutex = PTHREAD_MUTEX_INITIALIZER; /* each thread needs to know when they can make new threads, so we have to implement mutexes */

/* consts */
const pixel pix_white = { 0xFF, 0xFF, 0xFF, 0xFF };
const pixel pix_black = { 0x00, 0x00, 0x00, 0x00 };

/* decls */

void trap_sigint(int _);
void flush_pixels(pixel color);
int get_thr_slot(void);
void start_mandelbrot(void); /* starts all of the compute threads */
void* compute_mandelbrot(void* param); /* pthread main for compute threads */
void compute_mandelbrot_sub(int left, int right, int top, int bottom);
pixel get_color(int ind);
void key_callback(GLFWwindow* win, int key, int scancode, int action, int mods);

/* defs */

int main(int argc, char** argv) {
	/* prepare globals */

	num_live_threads = 0;

	mpfr_init2(bound_left, PBITS);
	mpfr_init2(bound_right, PBITS);
	mpfr_init2(bound_top, PBITS);
	mpfr_init2(bound_bottom, PBITS);

	mpfr_set_d(bound_left, BOUND_LEFT, MPFR_RNDD);
	mpfr_set_d(bound_right, BOUND_RIGHT, MPFR_RNDD);
	mpfr_set_d(bound_top, BOUND_TOP, MPFR_RNDD);
	mpfr_set_d(bound_bottom, BOUND_BOTTOM, MPFR_RNDD);

	for (int i = 0; i < THR_MAX_ACTIVE; ++i) {
		live_threads[i] = 0;
	}

	signal(SIGINT, trap_sigint);

	/* quickly prepare context info */

	if (!glfwInit()) return 1;
	if (!(win = glfwCreateWindow(WIDTH, HEIGHT, TITLE, FS ? glfwGetPrimaryMonitor() : NULL, NULL))) return 2;
	glfwMakeContextCurrent(win);
	if (glxwInit()) return 3;

	glfwSetKeyCallback(win, key_callback);

	/* prepare GL state */

	glViewport(0, 0, WIDTH, HEIGHT);
	glClearColor(0.0f, 0.0f, 1.0f, 0.0f);

	glGenTextures(1, &tex);
	glBindTexture(GL_TEXTURE_2D, tex);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, WIDTH, HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	vs = glCreateShader(GL_VERTEX_SHADER);
	fs = glCreateShader(GL_FRAGMENT_SHADER);

	glShaderSource(vs, 1, &vs_passthrough, NULL);
	glShaderSource(fs, 1, &fs_passthrough, NULL);

	glCompileShader(vs);
	glCompileShader(fs);

	glGetShaderiv(vs, GL_COMPILE_STATUS, &r);
	if (!r) return 4;

	glGetShaderiv(fs, GL_COMPILE_STATUS, &r);
	if (!r) return 5;

	prg = glCreateProgram();

	glAttachShader(prg, vs);
	glAttachShader(prg, fs);

	glLinkProgram(prg);
	glGetProgramiv(prg, GL_LINK_STATUS, &r);
	if (!r) return 6;

	glUseProgram(prg);
	glUniform1i(glGetUniformLocation(prg, "fs_texture"), 0);
	glActiveTexture(GL_TEXTURE0);

	float verts[24] = {
		-1.0f, 1.0f, 0.0f, 1.0f, /* simple quad mapping a texture to the screen */
		-1.0f, -1.0f, 0.0f, 0.0f,
		1.0f, -1.0f, 1.0f, 0.0f,
		-1.0f, 1.0f, 0.0f, 1.0f,
		1.0f, -1.0f, 1.0f, 0.0f,
		1.0f, 1.0f, 1.0f, 1.0f,
	};

	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(float) * 4, verts);

	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(float) * 4, verts + 2);

	flush_pixels(pix_white);

	/* start mainloop */

	start_mandelbrot();

	r = 1;
	while (r) {
		glfwPollEvents();

		r &= !glfwWindowShouldClose(win);
		r &= !glfwGetKey(win, GLFW_KEY_ESCAPE);

		glClear(GL_COLOR_BUFFER_BIT);

		pthread_mutex_lock(&pixbuf_mutex); /* ensure that the pixbuf is safe for reading */
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, WIDTH, HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, pixbuf);
		pthread_mutex_unlock(&pixbuf_mutex);

		glDrawArrays(GL_TRIANGLES, 0, 6);

		glfwSwapBuffers(win);
	}

	/* cleanup */

	printf("terminating cleanly\n");

	for (int i = 0; i < THR_MAX_ACTIVE; ++i) {
		pthread_mutex_lock(&live_threads_mutex);
		if (live_threads[i]) {
			printf("cancelling workthread %d\n", i);
			pthread_cancel(threads[i]);
			printf("done\n");
			num_live_threads--;
			live_threads[i] = 0;
		}
		pthread_mutex_unlock(&live_threads_mutex);
	}

	glDisableVertexAttribArray(0);
	glDisableVertexAttribArray(1);

	glUseProgram(0);
	glDetachShader(prg, vs);
	glDetachShader(prg, fs);
	glDeleteShader(vs);
	glDeleteShader(fs);
	glDeleteProgram(prg);

	glDeleteTextures(1, &tex);

	glfwDestroyWindow(win);
	glfwTerminate();

	mpfr_clear(bound_left);
	mpfr_clear(bound_right);
	mpfr_clear(bound_top);
	mpfr_clear(bound_bottom);

	return 0;
}

void* compute_mandelbrot(void* param) {
	mandelbrot_params* p = (mandelbrot_params*) param;
	printf("starting compute thread with sector (%d, %d, %d, %d) index %d\n", p->left, p->right, p->top, p->bottom, p->thr_index);

	compute_mandelbrot_sub(p->left, p->right, p->top, p->bottom);

	if (p->thr_index >= 0) {
		pthread_mutex_lock(&live_threads_mutex);
		live_threads[p->thr_index] = 0;
		num_live_threads--;
		pthread_mutex_unlock(&live_threads_mutex);
	}

	free(p); /* free the parameter set, allocated by start_mandelbrot() */
	return NULL;
}

void trap_sigint(int _) {
	r = 0; /* kill mainloop quietly */
	printf("caught SIGINT\n");
}

void flush_pixels(pixel c) {
	for (int i = 0; i < WIDTH * HEIGHT; ++i) {
		pixbuf[i] = c;
	}
}

int get_thr_slot(void) {
	for (int i = 0; i < THR_MAX_ACTIVE; ++i) {
		if (!live_threads[i]) {
			return i;
		}
	}

	return -1;
}

void start_mandelbrot(void) {
	mandelbrot_params* p;

	/* first, stop any computations in progress */
	for (int i = 0; i < THR_MAX_ACTIVE; ++i) {
		pthread_mutex_lock(&live_threads_mutex);
		if (live_threads[i]) {
			pthread_cancel(threads[i]);
			live_threads[i] = 0;
			num_live_threads--;
		}
		pthread_mutex_unlock(&live_threads_mutex);
	}

	/* we can assume that each thread will take roughly the same amt of time to complete, so we
	 * won't worry about switching inactive threads to help with other tasks */

	for (int i = 0; i < THR_MAX_ACTIVE; ++i) {
		p = malloc(sizeof *p);

		p->left = i * (WIDTH / THR_MAX_ACTIVE);
		p->right = (i + 1) * (WIDTH / THR_MAX_ACTIVE) - 1;
		p->top = HEIGHT - 1;
		p->bottom = 0;
		p->thr_index = i;

		pthread_mutex_lock(&live_threads_mutex);
		live_threads[i] = 1;
		num_live_threads++;
		pthread_mutex_unlock(&live_threads_mutex);

		printf("spawning child thread index %d with params (%d, %d, %d, %d)\n", i, p->left, p->right, p->top, p->bottom);

		if (pthread_create(threads + i, NULL, compute_mandelbrot, p)) {
			printf("failed to spawn thread..\n");
			exit(10);
		}
	}
}

void compute_mandelbrot_sub(int left, int right, int top, int bottom) {
	for (int y = bottom; y <= top; ++y) {
		for (int x = left; x <= right; ++x) {
			int diverge = 0, i;
			img cur, inp;

			mpfr_init2(cur.r, PBITS);
			mpfr_init2(cur.i, PBITS);
			mpfr_init2(inp.r, PBITS);
			mpfr_init2(inp.i, PBITS);

			mpfr_sub(inp.r, bound_right, bound_left, MPFR_RNDD);
			mpfr_sub(inp.i, bound_top, bound_bottom, MPFR_RNDD);

			mpfr_mul_d(inp.r, inp.r, (double) x / (double) (WIDTH - 1), MPFR_RNDD);
			mpfr_mul_d(inp.i, inp.i, (double) y / (double) (HEIGHT - 1), MPFR_RNDD);

			mpfr_add(inp.r, inp.r, bound_left, MPFR_RNDD);
			mpfr_add(inp.i, inp.i, bound_bottom, MPFR_RNDD);

			mpfr_set_d(cur.r, 0.0, MPFR_RNDD);
			mpfr_set_d(cur.i, 0.0, MPFR_RNDD);

			for (i = 0; i < MBR_MAX_ITERATIONS; ++i) {
				mpfr_t dist, dist2, rt;
				mpfr_init2(dist, PBITS);
				mpfr_init2(dist2, PBITS);
				mpfr_init2(rt, PBITS);

				mpfr_mul(dist, cur.r, cur.r, MPFR_RNDD);
				mpfr_mul(dist2, cur.i, cur.i, MPFR_RNDD);

				mpfr_add(dist, dist, dist2, MPFR_RNDD);

				if (mpfr_cmp_d(dist, MBR_DIVERGE_THRESHOLD) >= 0) {
					diverge = 1;
					break;
				}

				mpfr_sub(dist, dist, dist2, MPFR_RNDD);
				mpfr_sub(rt, dist, dist2, MPFR_RNDD);
				mpfr_add(rt, rt, inp.r, MPFR_RNDD);
				
				mpfr_mul(cur.i, cur.r, cur.i, MPFR_RNDD);
				mpfr_mul_d(cur.i, cur.i, 2.0, MPFR_RNDD);
				mpfr_add(cur.i, cur.i, inp.i, MPFR_RNDD);
				mpfr_set(cur.r, rt, MPFR_RNDD);

				mpfr_clear(dist);
				mpfr_clear(dist2);
				mpfr_clear(rt);
			}

			/* choose color from palette, where i=MBR_MAX_ITERATIONS should be black */
			pthread_mutex_lock(&pixbuf_mutex);

			if (memcmp(pixbuf + (y * WIDTH + x), &pix_white, sizeof pix_white)) {
				/* this sector is already being worked on or done */
				pthread_mutex_unlock(&pixbuf_mutex);
				mpfr_clear(cur.r);
				mpfr_clear(cur.i);
				mpfr_clear(inp.r);
				mpfr_clear(inp.i);
				return;
			}

			pixbuf[y * WIDTH + x] = get_color(i);
			pthread_mutex_unlock(&pixbuf_mutex);

			mpfr_clear(cur.r);
			mpfr_clear(cur.i);
			mpfr_clear(inp.r);
			mpfr_clear(inp.i);
		}
	}
}

pixel get_color(int ind) {
	pixel output = {0};
	int seg_size = MBR_MAX_ITERATIONS / 3;

	if (ind == MBR_MAX_ITERATIONS) return pix_black;

	/* transition to blue, green, and then red */
	if (ind >= seg_size * 2) {
		output.b = 0xFF - (ind - (seg_size * 3)) * 0xFF / seg_size;
		return output;
	}

	if (ind >= seg_size) {
		output.b = (ind - (seg_size * 2)) * 0xFF / seg_size;
		output.g = 0xFF - (ind - (seg_size * 2)) * 0xFF / seg_size;
		return output;
	}

	output.g = (ind - seg_size) * 0xFF / seg_size;
	output.r = 0xFF - (ind - seg_size) * 0xFF / seg_size;

	return output;
}

void key_callback(GLFWwindow* win, int key, int scancode, int action, int mods) {
	if (action != GLFW_PRESS) return;

	mpfr_t next_bl, next_br, next_bb, next_bt, hdiff, vdiff;

	mpfr_init2(next_bl, PBITS);
	mpfr_init2(next_br, PBITS);
	mpfr_init2(next_bb, PBITS);
	mpfr_init2(next_bt, PBITS);
	mpfr_init2(hdiff, PBITS);
	mpfr_init2(vdiff, PBITS);

	mpfr_set(next_bl, bound_left, MPFR_RNDD);
	mpfr_set(next_br, bound_right, MPFR_RNDD);
	mpfr_set(next_bb, bound_bottom, MPFR_RNDD);
	mpfr_set(next_bt, bound_top, MPFR_RNDD);

	mpfr_sub(hdiff, bound_right, bound_left, MPFR_RNDD);
	mpfr_sub(vdiff, bound_top, bound_bottom, MPFR_RNDD);

	mpfr_div_d(hdiff, hdiff, 2.0, MPFR_RNDD);
	mpfr_div_d(vdiff, vdiff, 2.0, MPFR_RNDD);

	switch (key) {
	case GLFW_KEY_LEFT:
		mpfr_sub(next_bl, bound_left, hdiff, MPFR_RNDD);
		mpfr_sub(next_br, bound_right, hdiff, MPFR_RNDD);
		break;
	case GLFW_KEY_RIGHT:
		mpfr_add(next_bl, bound_left, hdiff, MPFR_RNDD);
		mpfr_add(next_br, bound_right, hdiff, MPFR_RNDD);
		break;
	case GLFW_KEY_UP:
		mpfr_add(next_bb, bound_bottom, vdiff, MPFR_RNDD);
		mpfr_add(next_bt, bound_top, vdiff, MPFR_RNDD);
		break;
	case GLFW_KEY_DOWN:
		mpfr_sub(next_bb, bound_bottom, vdiff, MPFR_RNDD);
		mpfr_sub(next_bt, bound_top, vdiff, MPFR_RNDD);
		break;
	case GLFW_KEY_SPACE:
		/* zoom in 2x */
		mpfr_div_d(hdiff, hdiff, 2.0, MPFR_RNDD);
		mpfr_div_d(vdiff, vdiff, 2.0, MPFR_RNDD);
		mpfr_add(next_bl, bound_left, hdiff, MPFR_RNDD);
		mpfr_sub(next_br, bound_right, hdiff, MPFR_RNDD);
		mpfr_add(next_bb, bound_bottom, vdiff, MPFR_RNDD);
		mpfr_sub(next_bt, bound_top, vdiff, MPFR_RNDD);
		break;
	}

	mpfr_set(bound_left, next_bl, MPFR_RNDD);
	mpfr_set(bound_right, next_br, MPFR_RNDD);
	mpfr_set(bound_bottom, next_bb, MPFR_RNDD);
	mpfr_set(bound_top, next_bt, MPFR_RNDD);

	mpfr_clear(next_bl);
	mpfr_clear(next_br);
	mpfr_clear(next_bb);
	mpfr_clear(hdiff);
	mpfr_clear(vdiff);

	flush_pixels(pix_white);
	start_mandelbrot();
}
