
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
const uint8_t amount_of_steps = 25;
const uint8_t slice_size = 50;
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

const float zone_headings[5] = {-M_PI/2, -M_PI/4, 0, M_PI/4, M_PI/2};
const float zone_amount_of_markers[5] = {29, 37, 75, 36, 30};

struct image_t previous_image;
struct image_t previous_slice, current_slice;

static pthread_mutex_t mutex_depth, mutex_heading;

struct depth_object_t {
    float depth[207];
    bool updated;
};
struct depth_object_t global_depth_object;

struct heading_object_t {
    float best_heading;
    float safe_length;
    bool updated;
};
struct heading_object_t global_corr_heading_object;

/*
 * depth_finder_detector
 * @param img - input image to process
 * @return img
 */
static struct image_t *corr_depth_finder(struct image_t *current_image_p)
{
  struct depth_object_t local_depth_object;
  memset(&local_depth_object.depth, 0.0f, 207 * sizeof(float));

  float correlations[amount_of_steps];
  float mean, std;

  previous_slice.w = slice_size;
  previous_slice.h = slice_size;
  current_slice.w = slice_size;
  current_slice.h = slice_size;

  struct point_t previous_slice_center, current_slice_center;

  for (uint8_t slice_i = 0; slice_i < 207; slice_i++) {
    memset(&correlations, 0.0f, amount_of_steps * sizeof(float));
    mean = std = 0.0f;

    previous_slice_center.y = x_eval_locations[slice_i];
    previous_slice_center.x = y_eval_locations[slice_i];

    // Get previous slice from previous image around the slice center
    image_window(&previous_image, &previous_slice, &previous_slice_center);

    uint8_t *previous_buf = (uint8_t *)previous_slice.buf;

    for (uint8_t step_i = 0; step_i < amount_of_steps; step_i++) {
      // TODO check if correct
      current_slice_center.y = previous_slice_center.y + (uint32_t)(step_i * x_eval_directions[slice_i]);
      current_slice_center.x = previous_slice_center.x + (uint32_t)(step_i * y_eval_directions[slice_i]);

      // Bounds checking
      if ((int32_t) current_slice_center.x - (int32_t) slice_extend > 0                                &&
          (int32_t) current_slice_center.x + (int32_t) slice_extend < (int32_t) current_image_p->w - 1 &&
          (int32_t) current_slice_center.y - (int32_t) slice_extend > 0                                &&
          (int32_t) current_slice_center.y + (int32_t) slice_extend < (int32_t) current_image_p->h)
      {
        image_window(current_image_p, &current_slice, &current_slice_center);

        uint8_t *current_buf = (uint8_t *)current_slice.buf;

        // Shift color to correct for U and V unevenness
        uint8_t color_shift;
        if ((step_i % 2 == 0 && slice_i % 2 == 0) || (step_i % 2 != 0 && slice_i % 2 != 0)) {
          color_shift = 0;
        } else {
          color_shift = 2;
        }

        uint8_t y_location, uv_location;
        for (uint8_t x = 0; x < current_slice.w; x++) {
          for (uint8_t y = 0; y < current_slice.h; y++) {
            // Y_eval_e
            y_location = 2 * current_slice.w * y + 2 * x + 1;
            correlations[step_i] += (float) current_buf[y_location] *
                                    (float) previous_buf[y_location];
            // UV
            uv_location = 2 * current_slice.w * y + 2 * x;
            correlations[step_i] += (float) current_buf[uv_location + color_shift] *
                                    (float) previous_buf[uv_location];
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
        uint8_t *current_image_buf = (uint8_t *)current_image_p->buf;

        current_image_buf[2 * current_image_p->w * previous_slice_center.y + 2 * previous_slice_center.x + 1] =
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

    VERBOSE_PRINT("Busyness: %f, %f, %f, %f, %f\nSelected zone: %d, Angle: %f",
                  zone_busyness[0], zone_busyness[1], zone_busyness[2], zone_busyness[3], zone_busyness[4],
                  min_zone_i, zone_headings[min_zone_i]);

    pthread_mutex_lock(&mutex_heading);
    global_corr_heading_object.best_heading = zone_headings[min_zone_i];
    global_corr_heading_object.updated = true;
    pthread_mutex_unlock(&mutex_heading);
  } else {
    pthread_mutex_lock(&mutex_depth);
    memcpy(&global_depth_object.depth, &local_depth_object.depth, 207 * sizeof(float));
    global_depth_object.updated = true;
    pthread_mutex_unlock(&mutex_depth);
  }


  image_copy(current_image_p, &previous_image);

  return current_image_p;
}

struct image_t *corr_depth_finder1(struct image_t *img, uint8_t camera_id);
struct image_t *corr_depth_finder1(struct image_t *img, uint8_t camera_id __attribute__((unused)))
{
  return corr_depth_finder(img);
}

void corr_depth_finder_init(void) {

  cdf_max_std = 0.008;
  cdf_threshold = 0.4f;

  memset(&global_depth_object, 0, sizeof(struct depth_object_t));
  memset(&global_corr_heading_object, 0, sizeof(struct heading_object_t));

  pthread_mutex_init(&mutex_heading, NULL);
  pthread_mutex_init(&mutex_depth, NULL);

  image_create(&previous_image, 240, 520, IMAGE_YUV422);
  image_create(&previous_slice, slice_size, slice_size, IMAGE_YUV422);
  image_create(&current_slice, slice_size, slice_size, IMAGE_YUV422);

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
