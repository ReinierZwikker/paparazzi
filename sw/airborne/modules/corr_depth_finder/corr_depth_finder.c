#include "modules/datalink/telemetry.h"
#include "modules/corr_depth_finder/corr_depth_finder.h"

#include "modules/computer_vision/cv.h"
#include "modules/computer_vision/lib/vision/image.h"
#include "modules/core/abi.h"

#include "std.h"
#include <stdio.h>
//#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include "pthread.h"
#include "state.h"

#define AMOUNT_OF_SLICES 214
#define AMOUNT_OF_STEPS 15
#define AMOUNT_OF_SLICE_STEPS 3210  // 214 * 15

#define SIMD_ENABLED TRUE

#if SIMD_ENABLED == TRUE
#include "arm_neon.h"
#endif

#ifndef DEPTHFINDER_FPS
#define DEPTHFINDER_FPS 0       ///< Default FPS (zero means run at camera fps)
#endif

#ifndef HEADING_MODE
#define HEADING_MODE true       ///< Default navigation mode is heading mode
#endif

#define DEPTHFINDER_VERBOSE TRUE

#define PRINT(string,...) fprintf(stderr, "[corr_depth_finder->%s()] " string,__FUNCTION__ , ##__VA_ARGS__)
#if DEPTHFINDER_VERBOSE
#define VERBOSE_PRINT PRINT
#else
#define VERBOSE_PRINT(...)
#endif

// SETTINGS
const uint8_t slice_size = 16;
const uint8_t slice_extend = slice_size / 2;
const bool draw = true;

float cdf_max_std;
float cdf_threshold;

// Predefined evaluation location, direction and dependency on forward or sideways motion

// x-components of location in pixels
static uint32_t x_eval_locations[AMOUNT_OF_SLICES] = {285, 311, 337, 363, 381, 398, 411, 428, 288, 317, 346, 375, 395, 414, 429, 448, 290,
                                  320, 350, 380, 400, 420, 435, 455, 288, 317, 346, 375, 395, 414, 429, 448, 285, 311,
                                  337, 363, 381, 398, 411, 428, 234, 208, 182, 156, 138, 121, 108, 91, 231, 202, 173,
                                  144, 124, 105, 90, 71, 230, 200, 170, 140, 120, 100, 85, 65, 231, 202, 173, 144, 124,
                                  105, 90, 71, 234, 208, 182, 156, 138, 121, 108, 91, 371, 389, 408, 422, 440, 379, 398,
                                  418, 433, 453, 379, 398, 418, 433, 453, 371, 389, 408, 422, 440, 148, 130, 111, 97,
                                  79, 140, 121, 101, 86, 66, 140, 121, 101, 86, 66, 148, 130, 111, 97, 79, 276, 285,
                                  293, 260, 260, 260, 243, 234, 226, 288, 302, 316, 280, 290, 300, 270, 275, 280, 260,
                                  260, 260, 249, 244, 239, 240, 230, 220, 231, 217, 203, 260, 260, 260, 300, 220, 340,
                                  180, 380, 140, 420, 100, 460, 60, 260, 260, 260, 300, 220, 340, 180, 380, 140, 420,
                                  100, 460, 60, 260, 260, 260, 300, 220, 340, 180, 380, 140, 420, 100, 460, 60, 260,
                                  280, 300, 240, 220, 260, 280, 300, 240, 220, 260, 280, 300, 240, 220, 260, 280, 300,
                                  240, 220, 260, 280, 300, 240, 220};

// y-components of location in pixels
static uint32_t y_eval_locations[AMOUNT_OF_SLICES] = {105, 90, 75, 60, 50, 40, 32, 22, 112, 104, 96, 88, 83, 78, 74, 69, 120, 120, 120, 120,
                                  120, 120, 120, 120, 127, 135, 143, 151, 156, 161, 165, 170, 135, 150, 165, 180, 190,
                                  200, 207, 217, 135, 150, 165, 180, 190, 200, 207, 217, 127, 135, 143, 151, 156, 161,
                                  165, 170, 120, 120, 120, 120, 120, 120, 120, 120, 112, 104, 96, 88, 83, 78, 74, 69,
                                  105, 90, 75, 60, 50, 40, 32, 22, 75, 67, 60, 54, 46, 105, 102, 100, 98, 96, 134, 137,
                                  139, 141, 143, 164, 172, 179, 185, 193, 164, 172, 179, 185, 193, 134, 137, 139, 141,
                                  143, 105, 102, 100, 98, 96, 75, 67, 60, 54, 46, 156, 174, 192, 160, 180, 200, 156,
                                  174, 192, 91, 77, 63, 85, 68, 50, 81, 62, 42, 80, 60, 40, 81, 62, 42, 85, 68, 50, 91,
                                  77, 63, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 70, 70, 70,
                                  70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 170, 170, 170, 170, 170, 170, 170, 170, 170,
                                  170, 170, 170, 170, 145, 145, 145, 145, 145, 95, 95, 95, 95, 95, 25, 25, 25, 25, 25,
                                  45, 45, 45, 45, 45, 65, 65, 65, 65, 65};

// x-components of direction unit vector
static float_t x_eval_directions[AMOUNT_OF_SLICES] = {0.8660254037844387, 0.8660254037844387, 0.8660254037844387,
                                  0.8660254037844387, 0.8660254037844387, 0.8660254037844387,
                                  0.8660254037844387, 0.8660254037844387, 0.9659258262890683,
                                  0.9659258262890683, 0.9659258262890683, 0.9659258262890683,
                                  0.9659258262890683, 0.9659258262890683, 0.9659258262890683,
                                  0.9659258262890683, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                  1.0, 1.0, 0.9659258262890683, 0.9659258262890683, 0.9659258262890683,
                                  0.9659258262890683, 0.9659258262890683, 0.9659258262890683,
                                  0.9659258262890683, 0.9659258262890683, 0.8660254037844387,
                                  0.8660254037844387, 0.8660254037844387, 0.8660254037844387,
                                  0.8660254037844387, 0.8660254037844387, 0.8660254037844387,
                                  0.8660254037844387, -0.8660254037844387, -0.8660254037844387,
                                  -0.8660254037844387, -0.8660254037844387, -0.8660254037844387,
                                  -0.8660254037844387, -0.8660254037844387, -0.8660254037844387,
                                  -0.9659258262890682, -0.9659258262890682, -0.9659258262890682,
                                  -0.9659258262890682, -0.9659258262890682, -0.9659258262890682,
                                  -0.9659258262890682, -0.9659258262890682, -1.0, -1.0, -1.0,
                                  -1.0, -1.0, -1.0, -1.0, -1.0, -0.9659258262890682,
                                  -0.9659258262890682, -0.9659258262890682, -0.9659258262890682,
                                  -0.9659258262890682, -0.9659258262890682, -0.9659258262890682,
                                  -0.9659258262890682, -0.8660254037844387, -0.8660254037844387,
                                  -0.8660254037844387, -0.8660254037844387, -0.8660254037844387,
                                  -0.8660254037844387, -0.8660254037844387, -0.8660254037844387,
                                  0.9271838545667874, 0.9271838545667874, 0.9271838545667874,
                                  0.9271838545667874, 0.9271838545667874, 1.0,
                                  1.0, 1.0, 1.0,
                                  1.0, 1.0, 1.0,
                                  1.0, 1.0, 1.0,
                                  0.9271838545667874, 0.9271838545667874, 0.9271838545667874,
                                  0.9271838545667874, 0.9271838545667874, -0.9271838545667873,
                                  -0.9271838545667873, -0.9271838545667873, -0.9271838545667873,
                                  -0.9271838545667873, -1.0, -1.0,
                                  -1.0, -1.0, -1.0,
                                  -1.0, -1.0, -1.0,
                                  -1.0, -1.0, -0.9271838545667873,
                                  -0.9271838545667873, -0.9271838545667873, -0.9271838545667873,
                                  -0.9271838545667873, 0.42261826174069944, 0.42261826174069944,
                                  0.42261826174069944, 0.0, 0.0,
                                  0.0, -0.42261826174069933, -0.42261826174069933,
                                  -0.42261826174069933, 0.7071067811865476, 0.7071067811865476,
                                  0.7071067811865476, 0.5, 0.5,
                                  0.5, 0.25881904510252074, 0.25881904510252074,
                                  0.25881904510252074, 0.0, 0.0,
                                  0.0, -0.25881904510252085, -0.25881904510252085,
                                  -0.25881904510252085, -0.5, -0.5,
                                  -0.5, -0.7071067811865475, -0.7071067811865475,
                                  -0.7071067811865475, 1.0, -1.0, -1.0, 1.0, -1.0,
                                  1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0,
                                  1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0,
                                  -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -1.0,
                                  1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0,
                                  1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                  1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                                  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                  0.0, 0.0, 0.0};

// y-components of direction unit vector
static float_t y_eval_directions[AMOUNT_OF_SLICES] = {-0.5, -0.5, -0.5,
                                  -0.5, -0.5, -0.5,
                                  -0.5, -0.5, -0.25881904510252074,
                                  -0.25881904510252074, -0.25881904510252074, -0.25881904510252074,
                                  -0.25881904510252074, -0.25881904510252074, -0.25881904510252074,
                                  -0.25881904510252074, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                  0.0, 0.25881904510252074, 0.25881904510252074, 0.25881904510252074,
                                  0.25881904510252074, 0.25881904510252074, 0.25881904510252074,
                                  0.25881904510252074, 0.25881904510252074, 0.5,
                                  0.5, 0.5, 0.5,
                                  0.5, 0.5, 0.5,
                                  0.5, 0.5, 0.5,
                                  0.5, 0.5, 0.5,
                                  0.5, 0.5, 0.5,
                                  0.258819045102521, 0.258819045102521, 0.258819045102521,
                                  0.258819045102521, 0.258819045102521, 0.258819045102521,
                                  0.258819045102521, 0.258819045102521, 0.0,
                                  0.0, 0.0, 0.0,
                                  0.0, 0.0, 0.0,
                                  0.0, -0.258819045102521, -0.258819045102521,
                                  -0.258819045102521, -0.258819045102521, -0.258819045102521,
                                  -0.258819045102521, -0.258819045102521, -0.258819045102521,
                                  -0.5, -0.5, -0.5,
                                  -0.5, -0.5, -0.5,
                                  -0.5, -0.5, -0.374606593415912,
                                  -0.374606593415912, -0.374606593415912, -0.374606593415912,
                                  -0.374606593415912, -0.12186934340514748, -0.12186934340514748,
                                  -0.12186934340514748, -0.12186934340514748, -0.12186934340514748,
                                  0.12186934340514748, 0.12186934340514748, 0.12186934340514748,
                                  0.12186934340514748, 0.12186934340514748, 0.374606593415912,
                                  0.374606593415912, 0.374606593415912, 0.374606593415912,
                                  0.374606593415912, 0.37460659341591224, 0.37460659341591224,
                                  0.37460659341591224, 0.37460659341591224, 0.37460659341591224,
                                  0.12186934340514755, 0.12186934340514755, 0.12186934340514755,
                                  0.12186934340514755, 0.12186934340514755, -0.12186934340514755,
                                  -0.12186934340514755, -0.12186934340514755, -0.12186934340514755,
                                  -0.12186934340514755, -0.37460659341591224, -0.37460659341591224,
                                  -0.37460659341591224, -0.37460659341591224, -0.37460659341591224,
                                  0.9063077870366499, 0.9063077870366499, 0.9063077870366499,
                                  1.0, 1.0, 1.0, 0.90630778703665, 0.90630778703665,
                                  0.90630778703665, -0.7071067811865476, -0.7071067811865476,
                                  -0.7071067811865476, -0.8660254037844386, -0.8660254037844386,
                                  -0.8660254037844386, -0.9659258262890683, -0.9659258262890683,
                                  -0.9659258262890683, -1.0, -1.0, -1.0, -0.9659258262890683,
                                  -0.9659258262890683, -0.9659258262890683, -0.8660254037844387,
                                  -0.8660254037844387, -0.8660254037844387, -0.7071067811865476,
                                  -0.7071067811865476, -0.7071067811865476,
                                  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                  0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0,
                                  1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                  1.0, 1.0, 1.0};

//// Dependency on forward motion (0) or sideways motion (1)
//static bool eval_dependencies[AMOUNT_OF_SLICES] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
//                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
//                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
//                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
//                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
//                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
//                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
//                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
//                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
//                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
//                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
//                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
//                               0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
//                               1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
//                               1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
//                               1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
//                               1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
//                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
//                               0, 0, 0, 0};

static uint8_t eval_zones[AMOUNT_OF_SLICES] = {2, 3, 3, 3, 3, 3, 4, 4, 2, 3, 3, 3, 3, 4, 4, 4, 2, 3, 3, 3, 4, 4, 4, 4, 2, 3,
                                        3, 3, 3, 4, 4, 4, 2, 3, 3, 3, 3, 3, 4, 4, 2, 1, 1, 1, 1, 1, 0, 0, 2, 1, 1, 1,
                                        1, 0, 0, 0, 2, 1, 1, 1, 1, 0, 0, 0, 2, 1, 1, 1, 1, 0, 0, 0, 2, 1, 1, 1, 1, 1,
                                        0, 0, 3, 3, 4, 4, 4, 3, 3, 4, 4, 4, 3, 3, 4, 4, 4, 3, 3, 4, 4, 4, 1, 1, 0, 0,
                                        0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                        2, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 3,
                                        1, 3, 1, 4, 0, 4, 0, 2, 2, 2, 2, 2, 3, 1, 3, 1, 4, 0, 4, 0, 2, 2, 2, 2, 2, 3,
                                        1, 3, 1, 4, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                        0, 0, 0, 0, 0, 0};

static float zone_headings[5] = {-M_PI/4, -M_PI/16, 0, M_PI/16, M_PI/4};
static float zone_amount_of_markers[5] = {29, 37, 75, 36, 30};

struct image_t previous_image;
uint8_t **current_buf_locations_p;
uint8_t **previous_buf_locations_p;
uint8_t *previous_current_image_buf_p;
uint8_t *previous_previous_image_buf_p;

bool *color_shift_p;

static pthread_mutex_t mutex;

struct heading_object_t {
    float best_heading;
    float safe_length;
    float zone_busyness_ll;
    float zone_busyness_l;
    float zone_busyness_m;
    float zone_busyness_r;
    float zone_busyness_rr;
    float cycle_time;
    bool updated;
};
struct heading_object_t global_corr_heading_object;


//struct slice_t {
//  int16_t x_center;
//  int16_t y_center;
//  int16_t x_origin;
//  int16_t y_origin;
//  uint16_t buffer_origin;
//};

uint8_t *find_current_buf_locations(struct image_t *image_p, uint8_t **local_buf_locations_p) {
  uint8_t *img_buf = (uint8_t *) image_p->buf;

  for (uint8_t slice_i = 0; slice_i < AMOUNT_OF_SLICES; slice_i++) {
    for (uint8_t step_i = 0; step_i < AMOUNT_OF_STEPS; step_i++) {
      uint16_t x_center = y_eval_locations[slice_i] + (uint32_t)(step_i * y_eval_directions[slice_i]);
      uint16_t y_center = x_eval_locations[slice_i] + (uint32_t)(step_i * x_eval_directions[slice_i]);

      local_buf_locations_p[slice_i * step_i] =
              img_buf + 2 * image_p->w * (y_center - slice_extend) + 2 * (x_center - slice_extend);
    }
  }

  return img_buf;
}

uint8_t *find_previous_buf_locations(struct image_t *image_p, uint8_t **local_buf_locations_p) {
  uint8_t *img_buf = (uint8_t *) image_p->buf;

  for (uint8_t slice_i = 0; slice_i < AMOUNT_OF_SLICES; slice_i++) {
      uint16_t x_center = y_eval_locations[slice_i];
      uint16_t y_center = x_eval_locations[slice_i];

      local_buf_locations_p[slice_i] =
              img_buf + 2 * image_p->w * (y_center - slice_extend) + 2 * (x_center - slice_extend);

  }

  return img_buf;
}


/*
 * depth_finder_detector
 * @param img - input image to process
 * @return img
 */
static struct image_t *corr_depth_finder(struct image_t *current_image_p) {
  clock_t start = clock();

  uint8_t *current_buf = (uint8_t *) current_image_p->buf;
  uint8_t *previous_buf = (uint8_t *) previous_image.buf;


  if (current_buf != previous_current_image_buf_p) {
    previous_current_image_buf_p = find_current_buf_locations(current_image_p, current_buf_locations_p);
  }
  if (previous_buf != previous_previous_image_buf_p) {
    previous_previous_image_buf_p = find_previous_buf_locations(&previous_image, previous_buf_locations_p);
  }

  float depths[AMOUNT_OF_SLICES];
  memset(&depths, 0, AMOUNT_OF_SLICES * sizeof(float));

  float correlations[AMOUNT_OF_STEPS];
  float mean, std;

  //struct slice_t previous_slice, current_slice;

  for (uint8_t slice_i = 0; slice_i < AMOUNT_OF_SLICES; slice_i++) {
    memset(&correlations, 0.0f, AMOUNT_OF_STEPS * sizeof(float));
    mean = std = 0.0f;

    //previous_slice.x_center = y_eval_locations[slice_i];
    //previous_slice.y_center = x_eval_locations[slice_i];

    //previous_slice.buffer_origin =
    //        (uint16_t)(2 * current_image_p->w * (previous_slice.y_center - (int16_t) slice_extend)
    //        + 2 * (previous_slice.x_center - (int16_t) slice_extend));

    for (uint8_t step_i = 0; step_i < AMOUNT_OF_STEPS; step_i++) {
      //current_slice.x_center = previous_slice.x_center + (uint32_t)(step_i * y_eval_directions[slice_i]);
      //current_slice.y_center = previous_slice.y_center + (uint32_t)(step_i * x_eval_directions[slice_i]);

      //current_slice.buffer_origin =
      //        (uint16_t)(2 * current_image_p->w * (current_slice.y_center - (int16_t) slice_extend)
      //        + 2 * (current_slice.x_center - (int16_t) slice_extend));

      // Y,UV = color coordinates
      // x,y  = pixel coordinates

      uint8_t *previous_Y_slice_p, *previous_UV_slice_p, *current_Y_slice_p, *current_UV_slice_p;

      #if SIMD_ENABLED == TRUE  // === SIMD VARIANT OF THE FUNCTION ===

        uint8x16_t partial_sum = vdupq_n_u8(0);

        for (uint8_t slice_line = 0; slice_line < slice_size; slice_line++) {

          uint8_t *buffer_offset = 2 * current_image_p->w * slice_line;
          // Y_eval_e
          current_Y_slice_p   = current_buf_locations_p[slice_i * step_i] + buffer_offset + 1;
          previous_Y_slice_p  = previous_buf_locations_p[slice_i]         + buffer_offset + 1;
          previous_UV_slice_p = previous_buf_locations_p[slice_i]         + buffer_offset;
          current_UV_slice_p  = current_buf_locations_p[slice_i * step_i] + buffer_offset;
          uint8x16_t current_buf_y_vec, previous_buf_y_vec, current_buf_uv_vec, previous_buf_uv_vec;
          // TODO improve loading
          for (uint32_t buf_x = 0; buf_x < 16; buf_x++) {
            current_buf_y_vec[buf_x]   = *(current_Y_slice_p   + 2 * buf_x);
            previous_buf_y_vec[buf_x]  = *(previous_Y_slice_p  + 2 * buf_x);
            current_buf_uv_vec[buf_x]  = *(current_UV_slice_p  + 2 * buf_x + 2 * *color_shift_p[slice_i * step_i]);
            previous_buf_uv_vec[buf_x] = *(previous_UV_slice_p + 2 * buf_x);
          }
          partial_sum = vmlaq_u8(current_buf_y_vec, previous_buf_y_vec, partial_sum);
          partial_sum = vmlaq_u8(current_buf_uv_vec, previous_buf_uv_vec, partial_sum);
        }
        // TODO Improve to HADD
        for (uint8_t *vec_i = 0; vec_i < 16; vec_i++) {
          correlations[step_i] += (float) partial_sum[vec_i];
        }

      #else  // === NORMAL VARIANT OF THE FUNCTION ===

        for (uint32_t x_slice = 0; x_slice < slice_size; x_slice++) {
          for (uint32_t y_slice = 0; y_slice < slice_size; y_slice++) {
            uint32_t buffer_offset = 2 * current_image_p->w * y_slice + 2 * x_slice;
            // Y_eval_e
            current_Y_slice_p   = current_buf_locations_p[slice_i * step_i] + buffer_offset + 1;
            previous_Y_slice_p  = previous_buf_locations_p[slice_i]         + buffer_offset + 1;
            correlations[step_i] += (float) (*current_Y_slice_p * *previous_Y_slice_p);
            // UV
            previous_UV_slice_p = previous_buf_locations_p[slice_i]         + buffer_offset;
            current_UV_slice_p  = current_buf_locations_p[slice_i * step_i] + buffer_offset;
            correlations[step_i] += (float) (*(current_UV_slice_p + 2 * *color_shift_p[slice_i * step_i])
                    * *previous_UV_slice_p);
          }
        }

      #endif

      // Change range to 0...1 instead of 0...(2*50*50*255*255)
      correlations[step_i] /= 325125000.0f;

      mean += correlations[step_i];
    }

    // Find standard deviation of steps
    mean /= (float) AMOUNT_OF_STEPS;
    for (uint8_t step_i = 0; step_i < AMOUNT_OF_STEPS; step_i++) {
      std += powf((correlations[step_i] - mean), 2);
    }
    std = sqrtf(std / (float) AMOUNT_OF_STEPS);

    // VERBOSE_PRINT("mean: %f \t\t\t| std: %f > %f\n", mean, std, cdf_max_std);

    // Only consider this slice if the standard deviation is "high enough" ™
    if (std > cdf_max_std) {
      // Argmax of the correlations at the current eval location
      uint8_t max_i = 0;
      for (uint8_t step_i = 1; step_i < AMOUNT_OF_STEPS; step_i++) {
        if (correlations[step_i] > correlations[max_i]) {
          max_i = step_i;
        }
      }

      // TODO maybe divide by body velocity (forward or sideways dependent on dependency array)
      depths[slice_i] = (float) max_i / (float) AMOUNT_OF_STEPS;

      if (draw) {
        current_buf[2 * current_image_p->w * x_eval_locations[slice_i] + 2 * y_eval_locations[slice_i] + 1] =
            (uint8_t)(depths[slice_i] * 255);
      }
    }
  }

  float zone_busyness[5] = {0, 0, 0, 0, 0};
  for (uint8_t slice_i = 0; slice_i < AMOUNT_OF_SLICES; slice_i++) {
    zone_busyness[eval_zones[slice_i]] += depths[slice_i];
  }

  uint8_t min_zone_i = 2;
  for (uint8_t zone_i = 0; zone_i < 5; zone_i++) {
    zone_busyness[zone_i] /= zone_amount_of_markers[zone_i];
    // Only select zone if "significantly better" ™
    if (zone_busyness[zone_i] < zone_busyness[min_zone_i] && zone_busyness[zone_i] > cdf_threshold) {
      min_zone_i = zone_i;
    }
  }

  pthread_mutex_lock(&mutex);
  global_corr_heading_object.best_heading = zone_headings[min_zone_i];
  global_corr_heading_object.zone_busyness_ll = zone_busyness[0];
  global_corr_heading_object.zone_busyness_l = zone_busyness[1];
  global_corr_heading_object.zone_busyness_m = zone_busyness[2];
  global_corr_heading_object.zone_busyness_r = zone_busyness[3];
  global_corr_heading_object.zone_busyness_rr = zone_busyness[4];
  global_corr_heading_object.updated = true;
  pthread_mutex_unlock(&mutex);

  image_copy(current_image_p, &previous_image);

  pthread_mutex_lock(&mutex);
  clock_t end = clock();
  global_corr_heading_object.cycle_time = (end - start);
  pthread_mutex_unlock(&mutex);

  return current_image_p;
}

struct image_t *corr_depth_finder1(struct image_t *img, uint8_t camera_id);
struct image_t *corr_depth_finder1(struct image_t *img, uint8_t camera_id __attribute__((unused)))
{
  return corr_depth_finder(img);
}

// Create telemetry message
static void send_corr_depth_finder(struct transport_tx *trans, struct link_device *dev) {
  static struct heading_object_t local_corr_heading_object;
  pthread_mutex_lock(&mutex);
  memcpy(&local_corr_heading_object, &global_corr_heading_object, sizeof(struct heading_object_t));
  pthread_mutex_unlock(&mutex);

  pprz_msg_send_CORR_DEPTH_FINDER(trans, dev, AC_ID,
                                  &local_corr_heading_object.best_heading,
                                  &local_corr_heading_object.zone_busyness_ll,
                                  &local_corr_heading_object.zone_busyness_l,
                                  &local_corr_heading_object.zone_busyness_m,
                                  &local_corr_heading_object.zone_busyness_r,
                                  &local_corr_heading_object.zone_busyness_rr,
                                  &local_corr_heading_object.cycle_time);
}

void corr_depth_finder_init(void) {
  register_periodic_telemetry(DefaultPeriodic, PPRZ_MSG_ID_CORR_DEPTH_FINDER, send_corr_depth_finder);

  cdf_max_std = 0.008;
  cdf_threshold = 0.6f;

  current_buf_locations_p = malloc(AMOUNT_OF_SLICE_STEPS * sizeof(uint8_t*));
  previous_buf_locations_p = malloc(AMOUNT_OF_STEPS * sizeof(uint8_t*));

  color_shift_p = malloc(AMOUNT_OF_SLICE_STEPS * sizeof(bool));

  for (uint8_t slice_i = 0; slice_i < AMOUNT_OF_SLICES; slice_i++) {
    for (uint8_t step_i = 0; step_i < AMOUNT_OF_STEPS; step_i++) {
      // Shift color to correct for U and V unevenness
      if ((step_i % 2 == 0 && slice_i % 2 == 0) || (step_i % 2 != 0 && slice_i % 2 != 0)) {
        *(color_shift_p + slice_i * step_i) = false;
      } else {
        *(color_shift_p + slice_i * step_i) = true;
      }
    }
  }
  
  memset(&global_corr_heading_object, 0, sizeof(struct heading_object_t));

  pthread_mutex_init(&mutex, NULL);

  previous_current_image_buf_p = NULL;
  previous_previous_image_buf_p = NULL;

  image_create(&previous_image, 240, 520, IMAGE_YUV422);

  cv_add_to_device(&DEPTHFINDER_CAMERA, corr_depth_finder1, DEPTHFINDER_FPS, 0);
}

void corr_depth_finder_periodic(void) {
  static struct heading_object_t local_heading_object;

  pthread_mutex_lock(&mutex);
  memcpy(&local_heading_object, &global_corr_heading_object, sizeof(struct heading_object_t));
  pthread_mutex_unlock(&mutex);

  if(local_heading_object.updated) {
    AbiSendMsgDEPTH_FINDER_HEADING(CORR_DEPTH_ID, local_heading_object.best_heading);
    local_heading_object.updated = false;
  }
}
