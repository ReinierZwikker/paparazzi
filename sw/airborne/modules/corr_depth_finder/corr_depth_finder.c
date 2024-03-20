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
const uint8_t amount_of_steps = 15;
const uint8_t slice_size = 20;
const uint8_t slice_extend = slice_size / 2;
const bool draw = true;

float cdf_max_std;
float cdf_threshold;
// Predefined evaluation location, direction and dependency on forward or sideways motion

// x-components of location in pixels
uint32_t x_eval_locations[207] = {285, 311, 337, 363, 381, 398, 415, 288, 317, 346, 375, 395, 414, 433, 462, 290, 320,
                                  350, 380, 400, 420, 440, 470, 288, 317, 346, 375, 395, 414, 433, 462, 285, 311, 337,
                                  363, 381, 398, 415, 234, 208, 182, 156, 138, 121, 104, 231, 202, 173, 144, 124, 105,
                                   86,  57, 230, 200, 170, 140, 120, 100,  80,  50, 231, 202, 173, 144, 124, 105,  86,
                                   57, 234, 208, 182, 156, 138, 121, 104, 371, 389, 408, 426, 454, 379, 398, 418, 438,
                                  468, 379, 398, 418, 438, 468, 371, 389, 408, 426, 454, 148, 130, 111,  93,  65, 140,
                                  121, 101,  81,  51, 140, 121, 101,  81,  51, 148, 130, 111,  93,  65, 276, 285, 293,
                                  260, 260, 260, 243, 234, 226, 288, 302, 316, 280, 290, 300, 270, 275, 280, 260, 260,
                                  260, 249, 244, 239, 240, 230, 220, 231, 217, 203, 260, 260, 300, 220, 340, 180, 380,
                                  140, 420, 100, 460,  60, 260, 260, 300, 220, 340, 180, 380, 140, 420, 100, 460,  60,
                                  260, 260, 300, 220, 340, 180, 380, 140, 420, 100, 460,  60, 260, 280, 300, 240, 220,
                                  260, 280, 300, 240, 220, 260, 280, 300, 240, 220, 260, 280, 300, 240, 220, 260, 280,
                                  300, 240, 220};
// y-components of location in pixels
uint32_t y_eval_locations[207] = {105,  90,  75,  60,  50,  40,  30, 112, 104,  96,  88,  83,  78,  73,  65, 120, 120,
                                  120, 120, 120, 120, 120, 120, 127, 135, 143, 151, 156, 161, 166, 174, 135, 150, 165,
                                  180, 190, 200, 210, 135, 150, 165, 180, 190, 200, 210, 127, 135, 143, 151, 156, 161,
                                  166, 174, 120, 120, 120, 120, 120, 120, 120, 120, 112, 104,  96,  88,  83,  78,  73,
                                   65, 105,  90,  75,  60,  50,  40,  30,  75,  67,  60,  52,  41, 105, 102, 100,  98,
                                   94, 134, 137, 139, 141, 145, 164, 172, 179, 187, 198, 164, 172, 179, 187, 198, 134,
                                  137, 139, 141, 145, 105, 102, 100,  98,  94,  75,  67,  60,  52,  41, 156, 174, 192,
                                  160, 180, 200, 156, 174, 192,  91,  77,  63,  85,  68,  50,  81,  62,  42,  80,  60,
                                   40,  81,  62,  42,  85,  68,  50,  91,  77,  63, 120, 120, 120, 120, 120, 120, 120,
                                   120, 120, 120, 120, 120, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 170, 170,
                                   170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 145, 145, 145, 145, 145, 95, 95,
                                   95, 95, 95, 25, 25, 25, 25, 25, 45, 45, 45, 45, 45, 65, 65, 65, 65, 65};

// x-components of direction unit vector
float_t x_eval_directions[207] = {8.66025404e-01, 8.66025404e-01, 8.66025404e-01, 8.66025404e-01,
                                  8.66025404e-01, 8.66025404e-01, 8.66025404e-01, 9.65925826e-01,
                                   9.65925826e-01, 9.65925826e-01, 9.65925826e-01, 9.65925826e-01,
                                   9.65925826e-01, 9.65925826e-01, 9.65925826e-01, 1.00000000e+00,
                                   1.00000000e+00, 1.00000000e+00, 1.00000000e+00, 1.00000000e+00,
                                   1.00000000e+00, 1.00000000e+00, 1.00000000e+00, 9.65925826e-01,
                                   9.65925826e-01, 9.65925826e-01, 9.65925826e-01, 9.65925826e-01,
                                   9.65925826e-01, 9.65925826e-01, 9.65925826e-01, 8.66025404e-01,
                                   8.66025404e-01, 8.66025404e-01, 8.66025404e-01, 8.66025404e-01,
                                   8.66025404e-01, 8.66025404e-01, -8.66025404e-01, -8.66025404e-01,
                                   -8.66025404e-01, -8.66025404e-01, -8.66025404e-01, -8.66025404e-01,
                                   -8.66025404e-01, -9.65925826e-01, -9.65925826e-01, -9.65925826e-01,
                                   -9.65925826e-01, -9.65925826e-01, -9.65925826e-01, -9.65925826e-01,
                                   -9.65925826e-01, -1.00000000e+00, -1.00000000e+00, -1.00000000e+00,
                                   -1.00000000e+00, -1.00000000e+00, -1.00000000e+00, -1.00000000e+00,
                                   -1.00000000e+00, -9.65925826e-01, -9.65925826e-01, -9.65925826e-01,
                                   -9.65925826e-01, -9.65925826e-01, -9.65925826e-01, -9.65925826e-01,
                                   -9.65925826e-01, -8.66025404e-01, -8.66025404e-01, -8.66025404e-01,
                                   -8.66025404e-01, -8.66025404e-01, -8.66025404e-01, -8.66025404e-01,
                                   9.27183855e-01, 9.27183855e-01, 9.27183855e-01, 9.27183855e-01,
                                   9.27183855e-01, 9.92546152e-01, 9.92546152e-01, 9.92546152e-01,
                                   9.92546152e-01, 9.92546152e-01, 9.92546152e-01, 9.92546152e-01,
                                   9.92546152e-01, 9.92546152e-01, 9.92546152e-01, 9.27183855e-01,
                                   9.27183855e-01, 9.27183855e-01, 9.27183855e-01, 9.27183855e-01,
                                   -9.27183855e-01, -9.27183855e-01, -9.27183855e-01, -9.27183855e-01,
                                   -9.27183855e-01, -9.92546152e-01, -9.92546152e-01, -9.92546152e-01,
                                   -9.92546152e-01, -9.92546152e-01, -9.92546152e-01, -9.92546152e-01,
                                   -9.92546152e-01, -9.92546152e-01, -9.92546152e-01, -9.27183855e-01,
                                   -9.27183855e-01, -9.27183855e-01, -9.27183855e-01, -9.27183855e-01,
                                   4.22618262e-01, 4.22618262e-01, 4.22618262e-01, 6.12323400e-17,
                                   6.12323400e-17, 6.12323400e-17, -4.22618262e-01, -4.22618262e-01,
                                   -4.22618262e-01, 7.07106781e-01, 7.07106781e-01, 7.07106781e-01,
                                   5.00000000e-01, 5.00000000e-01, 5.00000000e-01, 2.58819045e-01,
                                   2.58819045e-01, 2.58819045e-01, 6.12323400e-17, 6.12323400e-17,
                                   6.12323400e-17, -2.58819045e-01, -2.58819045e-01, -2.58819045e-01,
                                   -5.00000000e-01, -5.00000000e-01, -5.00000000e-01, -7.07106781e-01,
                                   -7.07106781e-01, -7.07106781e-01, 1.00000000e+00, 1.00000000e+00,
                                   1.00000000e+00, 1.00000000e+00, 1.00000000e+00, 1.00000000e+00,
                                   1.00000000e+00, 1.00000000e+00, 1.00000000e+00, 1.00000000e+00,
                                   1.00000000e+00, 1.00000000e+00, 1.00000000e+00, 1.00000000e+00,
                                   1.00000000e+00, 1.00000000e+00, 1.00000000e+00, 1.00000000e+00,
                                   1.00000000e+00, 1.00000000e+00, 1.00000000e+00, 1.00000000e+00,
                                   1.00000000e+00, 1.00000000e+00, 1.00000000e+00, 1.00000000e+00,
                                   1.00000000e+00, 1.00000000e+00, 1.00000000e+00, 1.00000000e+00,
                                   1.00000000e+00, 1.00000000e+00, 1.00000000e+00, 1.00000000e+00,
                                   1.00000000e+00, 1.00000000e+00, 1.00000000e+00, 1.00000000e+00,
                                   1.00000000e+00, 1.00000000e+00, 1.00000000e+00, 1.00000000e+00,
                                   1.00000000e+00, 1.00000000e+00, 1.00000000e+00, 1.00000000e+00,
                                   0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                                   0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                                   0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                                   0.00000000e+00, 0.00000000e+00, 0.00000000e+00};

// y-components of direction unit vector
float_t y_eval_directions[207] = {-5.00000000e-01, -5.00000000e-01, -5.00000000e-01, -5.00000000e-01,
                             -5.00000000e-01, -5.00000000e-01, -5.00000000e-01, -2.58819045e-01,
                             -2.58819045e-01, -2.58819045e-01, -2.58819045e-01, -2.58819045e-01,
                             -2.58819045e-01, -2.58819045e-01, -2.58819045e-01, 0.00000000e+00,
                             0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                             0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 2.58819045e-01,
                             2.58819045e-01, 2.58819045e-01, 2.58819045e-01, 2.58819045e-01,
                             2.58819045e-01, 2.58819045e-01, 2.58819045e-01, 5.00000000e-01,
                             5.00000000e-01, 5.00000000e-01, 5.00000000e-01, 5.00000000e-01,
                             5.00000000e-01, 5.00000000e-01, 5.00000000e-01, 5.00000000e-01,
                             5.00000000e-01, 5.00000000e-01, 5.00000000e-01, 5.00000000e-01,
                             5.00000000e-01, 2.58819045e-01, 2.58819045e-01, 2.58819045e-01,
                             2.58819045e-01, 2.58819045e-01, 2.58819045e-01, 2.58819045e-01,
                             2.58819045e-01, 1.22464680e-16, 1.22464680e-16, 1.22464680e-16,
                             1.22464680e-16, 1.22464680e-16, 1.22464680e-16, 1.22464680e-16,
                             1.22464680e-16, -2.58819045e-01, -2.58819045e-01, -2.58819045e-01,
                             -2.58819045e-01, -2.58819045e-01, -2.58819045e-01, -2.58819045e-01,
                             -2.58819045e-01, -5.00000000e-01, -5.00000000e-01, -5.00000000e-01,
                             -5.00000000e-01, -5.00000000e-01, -5.00000000e-01, -5.00000000e-01,
                             -3.74606593e-01, -3.74606593e-01, -3.74606593e-01, -3.74606593e-01,
                             -3.74606593e-01, -1.21869343e-01, -1.21869343e-01, -1.21869343e-01,
                             -1.21869343e-01, -1.21869343e-01, 1.21869343e-01, 1.21869343e-01,
                             1.21869343e-01, 1.21869343e-01, 1.21869343e-01, 3.74606593e-01,
                             3.74606593e-01, 3.74606593e-01, 3.74606593e-01, 3.74606593e-01,
                             3.74606593e-01, 3.74606593e-01, 3.74606593e-01, 3.74606593e-01,
                             3.74606593e-01, 1.21869343e-01, 1.21869343e-01, 1.21869343e-01,
                             1.21869343e-01, 1.21869343e-01, -1.21869343e-01, -1.21869343e-01,
                             -1.21869343e-01, -1.21869343e-01, -1.21869343e-01, -3.74606593e-01,
                             -3.74606593e-01, -3.74606593e-01, -3.74606593e-01, -3.74606593e-01,
                             9.06307787e-01, 9.06307787e-01, 9.06307787e-01, 1.00000000e+00,
                             1.00000000e+00, 1.00000000e+00, 9.06307787e-01, 9.06307787e-01,
                             9.06307787e-01, -7.07106781e-01, -7.07106781e-01, -7.07106781e-01,
                             -8.66025404e-01, -8.66025404e-01, -8.66025404e-01, -9.65925826e-01,
                             -9.65925826e-01, -9.65925826e-01, -1.00000000e+00, -1.00000000e+00,
                             -1.00000000e+00, -9.65925826e-01, -9.65925826e-01, -9.65925826e-01,
                             -8.66025404e-01, -8.66025404e-01, -8.66025404e-01, -7.07106781e-01,
                             -7.07106781e-01, -7.07106781e-01, 0.00000000e+00, 0.00000000e+00,
                             0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                             0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                             0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                             0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                             0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                             0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                             0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                             0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                             0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                             0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                             0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                             1.00000000e+00, 1.00000000e+00, 1.00000000e+00, 1.00000000e+00,
                             1.00000000e+00, 1.00000000e+00, 1.00000000e+00, 1.00000000e+00,
                             1.00000000e+00, 1.00000000e+00, 1.00000000e+00, 1.00000000e+00,
                             1.00000000e+00, 1.00000000e+00, 1.00000000e+00};

// Dependency on forward motion (0) or sideways motion (1)
bool eval_dependencies[207] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

const float zone_headings[5] = {-M_PI/8, -M_PI/16, 0, M_PI/16, M_PI/8};
const float zone_amount_of_markers[5] = {29, 37, 75, 36, 30};

struct image_t previous_image;

static pthread_mutex_t mutex_depth, mutex_heading;

struct depth_object_t {
    float depth[207];
    bool updated;
};
struct depth_object_t global_depth_object;

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


struct slice_t {
  int16_t x_center;
  int16_t y_center;
  int16_t x_origin;
  int16_t y_origin;
  int16_t x_max;
  int16_t y_max;
  uint16_t buffer_origin;
};

// Create telemetry message
static void send_corr_depth_finder(struct transport_tx *trans, struct link_device *dev) {
  static struct heading_object_t local_corr_heading_object;
  pthread_mutex_lock(&mutex_heading);
  memcpy(&local_corr_heading_object, &global_corr_heading_object, sizeof(struct heading_object_t));
  pthread_mutex_unlock(&mutex_heading);

  pprz_msg_send_CORR_DEPTH_FINDER(trans, dev, AC_ID,
                                  &local_corr_heading_object.best_heading,
                                  &local_corr_heading_object.zone_busyness_ll,
                                  &local_corr_heading_object.zone_busyness_l,
                                  &local_corr_heading_object.zone_busyness_m,
                                  &local_corr_heading_object.zone_busyness_r,
                                  &local_corr_heading_object.zone_busyness_rr,
                                  &local_corr_heading_object.cycle_time);
}

/*
 * depth_finder_detector
 * @param img - input image to process
 * @return img
 */
static struct image_t *corr_depth_finder(struct image_t *current_image_p)
{
  //time_t start_time;
  clock_t start = clock();

  struct depth_object_t local_depth_object;
  memset(&local_depth_object.depth, 0.0f, 207 * sizeof(float));

  uint8_t *previous_buf = (uint8_t *)previous_image.buf;
  uint8_t *current_buf = (uint8_t *)current_image_p->buf;

  float correlations[amount_of_steps];
  float mean, std;

  struct slice_t previous_slice, current_slice;

  for (uint8_t slice_i = 0; slice_i < 207; slice_i++) {
    memset(&correlations, 0.0f, amount_of_steps * sizeof(float));
    mean = std = 0.0f;

    previous_slice.x_center = y_eval_locations[slice_i];
    previous_slice.y_center = x_eval_locations[slice_i];

    previous_slice.x_origin = previous_slice.x_center - (int16_t) slice_extend;
    previous_slice.y_origin = previous_slice.y_center - (int16_t) slice_extend;
    previous_slice.x_max    = previous_slice.x_center + (int16_t) slice_extend;
    previous_slice.y_max    = previous_slice.y_center + (int16_t) slice_extend;

    previous_slice.buffer_origin = (uint16_t) (2 * slice_size * previous_slice.y_origin + 2 * previous_slice.x_origin);

    for (uint8_t step_i = 0; step_i < amount_of_steps; step_i++) {
      // TODO check if correct
      current_slice.x_center = previous_slice.x_center + (uint32_t)(step_i * y_eval_directions[slice_i]);
      current_slice.y_center = previous_slice.y_center + (uint32_t)(step_i * x_eval_directions[slice_i]);

      current_slice.x_origin = current_slice.x_center - (int16_t) slice_extend;
      current_slice.y_origin = current_slice.y_center - (int16_t) slice_extend;
      current_slice.x_max    = current_slice.x_center + (int16_t) slice_extend;
      current_slice.y_max    = current_slice.y_center + (int16_t) slice_extend;

      current_slice.buffer_origin = (uint16_t) (2 * slice_size * current_slice.y_origin + 2 * current_slice.x_origin);

      // Bounds checking
      if (current_slice.x_origin > 0                                &&
          current_slice.x_max    < (int16_t) current_image_p->w - 1 &&
          current_slice.y_origin > 0                                &&
          current_slice.y_max    < (int16_t) current_image_p->h)
      {
        // Shift color to correct for U and V unevenness
        uint8_t color_shift;
        if ((step_i % 2 == 0 && slice_i % 2 == 0) || (step_i % 2 != 0 && slice_i % 2 != 0)) {
          color_shift = 0;
        } else {
          color_shift = 2;
        }

        // Y,UV = color coordinates
        // x,y  = pixel coordinates
        uint8_t previous_Y_location, previous_UV_location, current_Y_location, current_UV_location;
        for (uint32_t x_slice = 0; x_slice < slice_size; x_slice++) {
          for (uint32_t y_slice = 0; y_slice < slice_size; y_slice++) {
            // Y_eval_e
            previous_Y_location = previous_slice.buffer_origin + 2 * slice_size * y_slice + 2 * x_slice + 1;
            current_Y_location  = current_slice.buffer_origin  + 2 * slice_size * y_slice + 2 * x_slice + 1;
            correlations[step_i] += (float) current_buf[current_Y_location] * (float) previous_buf[previous_Y_location];
            // UV
            previous_UV_location = previous_slice.buffer_origin + 2 * slice_size * y_slice + 2 * x_slice;
            current_UV_location  = current_slice.buffer_origin  + 2 * slice_size * y_slice + 2 * x_slice;
            correlations[step_i] += (float)  current_buf[current_UV_location + color_shift] *
                                    (float) previous_buf[previous_UV_location];
          }
        }
        // Change range to 0...1 instead of 0...(2*50*50*255*255)
        correlations[step_i] /= 325125000.0f;

        mean += correlations[step_i];
      }
    }

    // Find standard deviation of steps
    mean /= (float) amount_of_steps;
    for (uint8_t step_i = 0; step_i < amount_of_steps; step_i++) {
      std += powf((correlations[step_i] - mean), 2);
    }
    std = sqrtf(std / (float) amount_of_steps);

    // VERBOSE_PRINT("mean: %f \t\t\t| std: %f > %f\n", mean, std, cdf_max_std);

    // Only consider this slice if the standard deviation is "high enough" ™
    if (std > cdf_max_std) {
      // Argmax of the correlations at the current eval location
      uint8_t max_i = 0;
      for (uint8_t step_i = 1; step_i < amount_of_steps; step_i++) {
        if (correlations[step_i] > correlations[max_i]) {
          max_i = step_i;
        }
      }

      // TODO maybe divide by body velocity (forward or sideways dependent on dependency array)
      local_depth_object.depth[slice_i] = (float) max_i / (float) amount_of_steps;

      if (draw) {

        current_buf[2 * current_image_p->w * previous_slice.y_center + 2 * previous_slice.x_center + 1] =
                    (uint8_t) (local_depth_object.depth[slice_i] * 255);
      }
    }
  }

  if (HEADING_MODE) {
    float zone_busyness[5] = {0, 0, 0, 0, 0};
    uint8_t zone_selection;

    for (uint8_t slice_i = 0; slice_i < 207; slice_i++) {
      if        (x_eval_locations[slice_i] < 120) {
        zone_selection = 0;
      } else if (x_eval_locations[slice_i] < 210) {
        zone_selection = 1;
      } else if (x_eval_locations[slice_i] < 310) {
        zone_selection = 2;
      } else if (x_eval_locations[slice_i] < 400) {
        zone_selection = 3;
      } else {
        zone_selection = 4;
      }

      zone_busyness[zone_selection] += local_depth_object.depth[slice_i];
      // VERBOSE_PRINT("%d: busyness += %f = %f\n", zone_selection, local_depth_object.depth[slice_i], zone_busyness[zone_selection]);

    }

    uint8_t min_zone_i = 0;
    for (uint8_t zone_i = 0; zone_i < 5; zone_i++) {
      zone_busyness[zone_i] /= zone_amount_of_markers[zone_i];
      // Only select zone if "significantly better" ™
      if (zone_busyness[zone_i] < zone_busyness[min_zone_i]) {
        min_zone_i = zone_i;
      }
    }
    if (zone_busyness[0] + zone_busyness[1] + zone_busyness[2] + zone_busyness[3] + zone_busyness[4] < cdf_threshold) {
      min_zone_i = 2;
    }

    // VERBOSE_PRINT("Busyness: %f, %f, %f, %f, %f\nSelected zone: %d, Angle: %f",
    //               zone_busyness[0], zone_busyness[1], zone_busyness[2], zone_busyness[3], zone_busyness[4],
    //               min_zone_i, zone_headings[min_zone_i]);

    pthread_mutex_lock(&mutex_heading);
    global_corr_heading_object.best_heading = zone_headings[min_zone_i];
    global_corr_heading_object.zone_busyness_ll = zone_busyness[0];
    global_corr_heading_object.zone_busyness_l = zone_busyness[1];
    global_corr_heading_object.zone_busyness_m = zone_busyness[2];
    global_corr_heading_object.zone_busyness_r = zone_busyness[3];
    global_corr_heading_object.zone_busyness_rr = zone_busyness[4];
    global_corr_heading_object.updated = true;
    pthread_mutex_unlock(&mutex_heading);
  } else {
    pthread_mutex_lock(&mutex_depth);
    memcpy(&global_depth_object.depth, &local_depth_object.depth, 207 * sizeof(float));
    global_depth_object.updated = true;
    pthread_mutex_unlock(&mutex_depth);
  }


  image_copy(current_image_p, &previous_image);

  // VERBOSE_PRINT("Correlation time: %f\n", start_time - time(NULL));

  pthread_mutex_lock(&mutex_heading);
  clock_t end = clock();
  global_corr_heading_object.cycle_time = (end - start);
  pthread_mutex_unlock(&mutex_heading);

  return current_image_p;
}

struct image_t *corr_depth_finder1(struct image_t *img, uint8_t camera_id);
struct image_t *corr_depth_finder1(struct image_t *img, uint8_t camera_id __attribute__((unused)))
{
  return corr_depth_finder(img);
}

void corr_depth_finder_init(void) {
  register_periodic_telemetry(DefaultPeriodic, PPRZ_MSG_ID_CORR_DEPTH_FINDER, send_corr_depth_finder);

  cdf_max_std = 0.008;
  cdf_threshold = 0.4f;

  memset(&global_depth_object, 0, sizeof(struct depth_object_t));
  memset(&global_corr_heading_object, 0, sizeof(struct heading_object_t));

  pthread_mutex_init(&mutex_heading, NULL);
  pthread_mutex_init(&mutex_depth, NULL);

  image_create(&previous_image, 240, 520, IMAGE_YUV422);

  cv_add_to_device(&DEPTHFINDER_CAMERA, corr_depth_finder1, DEPTHFINDER_FPS, 0);
}

void corr_depth_finder_periodic(void) {
  if (HEADING_MODE) {
    static struct heading_object_t local_heading_object;

    pthread_mutex_lock(&mutex_heading);
    memcpy(&local_heading_object, &global_corr_heading_object, sizeof(struct heading_object_t));
    pthread_mutex_unlock(&mutex_heading);

    if(local_heading_object.updated) {
      // TODO Implement correct ABI Message
      AbiSendMsgDEPTH_FINDER_HEADING(CORR_DEPTH_ID, local_heading_object.best_heading);
      // AbiSendMsgGREEN_DETECTION(GREEN_DETECTION_ID, local_heading_object.best_heading, 1000, 10000);
      local_heading_object.updated = false;
    }
  } else {
    static struct depth_object_t local_depth_object;

    pthread_mutex_lock(&mutex_depth);
    memcpy(&local_depth_object, &global_depth_object, sizeof(struct depth_object_t));
    pthread_mutex_unlock(&mutex_depth);

    if(local_depth_object.updated){
      // TODO Implement ABI Message
      // AbiSendMsgCORR_DEPTH_FINDER(CORR_DEPTH_ID, local_depth_object.depth);
      local_depth_object.updated = false;
    }
  }
}
