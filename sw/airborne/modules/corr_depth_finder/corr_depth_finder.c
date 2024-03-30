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


// === COMPILE TIME PARAMETERS ===

#define AMOUNT_OF_SLICES 214        ///< Size of slice LUT
#define AMOUNT_OF_STEPS 15          ///< Amount of steps per slice
#define AMOUNT_OF_SLICE_STEPS 3210  ///< 214 * 15
#define SLICE_SIZE 16               ///< Slice size in pixels

#define AMOUNT_OF_IMAGE_BUFFERS 6   ///< Amount of separate buffer locations to remember for the current image

#define SIMD_ENABLED TRUE           ///< Only enable when compiling for ARM Cortex processor!
                                    ///  Uses ARM NEON Intrinsics to speed up calculations

#if SIMD_ENABLED == TRUE
#include "arm_neon.h"
#define SLICE_SIZE 16 // Needs to be 16 if using SIMD
#endif

#ifndef DEPTHFINDER_FPS
#define DEPTHFINDER_FPS 0       ///< Default FPS (zero means run at camera fps)
#endif

#define DEPTHFINDER_VERBOSE FALSE

#define PRINT(string,...) fprintf(stderr, "[corr_depth_finder->%s()] " string,__FUNCTION__ , ##__VA_ARGS__)
#if DEPTHFINDER_VERBOSE
#define VERBOSE_PRINT PRINT
#else
#define VERBOSE_PRINT(...)
#endif


// === RUN TIME PARAMETERS ===
const uint8_t slice_extend = SLICE_SIZE / 2;
const bool draw = true;

// Tuning parameters in PPRZ GCS
float cdf_max_std;
float cdf_threshold;


// === Predefined evaluation location, direction LUTS ===

// x-components of location in pixels
static uint32_t x_eval_locations[AMOUNT_OF_SLICES] = {
        285, 311, 337, 363, 381, 398, 411, 428, 288, 317, 346, 375, 395, 414, 429, 448, 290, 320, 350, 380, 400, 420,
        435, 455, 288, 317, 346, 375, 395, 414, 429, 448, 285, 311, 337, 363, 381, 398, 411, 428, 234, 208, 182, 156,
        138, 121, 108, 91, 231, 202, 173, 144, 124, 105, 90, 71, 230, 200, 170, 140, 120, 100, 85, 65, 231, 202, 173,
        144, 124, 105, 90, 71, 234, 208, 182, 156, 138, 121, 108, 91, 371, 389, 408, 422, 440, 379, 398, 418, 433, 453,
        379, 398, 418, 433, 453, 371, 389, 408, 422, 440, 148, 130, 111, 97, 79, 140, 121, 101, 86, 66, 140, 121, 101,
        86, 66, 148, 130, 111, 97, 79, 276, 285, 293, 260, 260, 260, 243, 234, 226, 288, 302, 316, 280, 290, 300, 270,
        275, 280, 260, 260, 260, 249, 244, 239, 240, 230, 220, 231, 217, 203, 260, 260, 260, 300, 220, 340, 180, 380,
        140, 420, 100, 460, 60, 260, 260, 260, 300, 220, 340, 180, 380, 140, 420, 100, 460, 60, 260, 260, 260, 300,
        220, 340, 180, 380, 140, 420, 100, 460, 60, 260, 280, 300, 240, 220, 260, 280, 300, 240, 220, 260, 280, 300,
        240, 220, 260, 280, 300, 240, 220, 260, 280, 300, 240, 220
  };

// y-components of location in pixels
static uint32_t y_eval_locations[AMOUNT_OF_SLICES] = {
        105, 90, 75, 60, 50, 40, 32, 22, 112, 104, 96, 88, 83, 78, 74, 69, 120, 120, 120, 120, 120, 120, 120, 120, 127,
        135, 143, 151, 156, 161, 165, 170, 135, 150, 165, 180, 190, 200, 207, 217, 135, 150, 165, 180, 190, 200, 207,
        217, 127, 135, 143, 151, 156, 161, 165, 170, 120, 120, 120, 120, 120, 120, 120, 120, 112, 104, 96, 88, 83, 78,
        74, 69, 105, 90, 75, 60, 50, 40, 32, 22, 75, 67, 60, 54, 46, 105, 102, 100, 98, 96, 134, 137, 139, 141, 143,
        164, 172, 179, 185, 193, 164, 172, 179, 185, 193, 134, 137, 139, 141, 143, 105, 102, 100, 98, 96, 75, 67, 60,
        54, 46, 156, 174, 192, 160, 180, 200, 156, 174, 192, 91, 77, 63, 85, 68, 50, 81, 62, 42, 80, 60, 40, 81, 62,
        42, 85, 68, 50, 91, 77, 63, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 70, 70, 70, 70,
        70, 70, 70, 70, 70, 70, 70, 70, 70, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 170, 145, 145,
        145, 145, 145, 95, 95, 95, 95, 95, 25, 25, 25, 25, 25, 45, 45, 45, 45, 45, 65, 65, 65, 65, 65
  };

// x-components of direction unit vector
static float_t x_eval_directions[AMOUNT_OF_SLICES] = {
        0.8660254037844387, 0.8660254037844387, 0.8660254037844387, 0.8660254037844387, 0.8660254037844387,
        0.8660254037844387, 0.8660254037844387, 0.8660254037844387, 0.9659258262890683, 0.9659258262890683,
        0.9659258262890683, 0.9659258262890683, 0.9659258262890683, 0.9659258262890683, 0.9659258262890683,
        0.9659258262890683, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9659258262890683, 0.9659258262890683,
        0.9659258262890683, 0.9659258262890683, 0.9659258262890683, 0.9659258262890683, 0.9659258262890683,
        0.9659258262890683, 0.8660254037844387, 0.8660254037844387, 0.8660254037844387, 0.8660254037844387,
        0.8660254037844387, 0.8660254037844387, 0.8660254037844387, 0.8660254037844387, -0.8660254037844387,
        -0.8660254037844387, -0.8660254037844387, -0.8660254037844387, -0.8660254037844387, -0.8660254037844387,
        -0.8660254037844387, -0.8660254037844387, -0.9659258262890682, -0.9659258262890682, -0.9659258262890682,
        -0.9659258262890682, -0.9659258262890682, -0.9659258262890682, -0.9659258262890682, -0.9659258262890682,
        -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -0.9659258262890682, -0.9659258262890682, -0.9659258262890682,
        -0.9659258262890682, -0.9659258262890682, -0.9659258262890682, -0.9659258262890682, -0.9659258262890682,
        -0.8660254037844387, -0.8660254037844387, -0.8660254037844387, -0.8660254037844387, -0.8660254037844387,
        -0.8660254037844387, -0.8660254037844387, -0.8660254037844387, 0.9271838545667874, 0.9271838545667874,
        0.9271838545667874, 0.9271838545667874, 0.9271838545667874, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        0.9271838545667874, 0.9271838545667874, 0.9271838545667874, 0.9271838545667874, 0.9271838545667874,
        -0.9271838545667873, -0.9271838545667873, -0.9271838545667873, -0.9271838545667873, -0.9271838545667873,
        -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -0.9271838545667873, -0.9271838545667873,
        -0.9271838545667873, -0.9271838545667873, -0.9271838545667873, 0.42261826174069944, 0.42261826174069944,
        0.42261826174069944, 0.0, 0.0, 0.0, -0.42261826174069933, -0.42261826174069933, -0.42261826174069933,
        0.7071067811865476, 0.7071067811865476, 0.7071067811865476, 0.5, 0.5, 0.5, 0.25881904510252074,
        0.25881904510252074, 0.25881904510252074, 0.0, 0.0, 0.0, -0.25881904510252085, -0.25881904510252085,
        -0.25881904510252085, -0.5, -0.5, -0.5, -0.7071067811865475, -0.7071067811865475, -0.7071067811865475,
        1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0,
        1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0
  };

// y-components of direction unit vector
static float_t y_eval_directions[AMOUNT_OF_SLICES] = {
        -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.25881904510252074, -0.25881904510252074,
        -0.25881904510252074, -0.25881904510252074, -0.25881904510252074, -0.25881904510252074, -0.25881904510252074,
        -0.25881904510252074, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25881904510252074, 0.25881904510252074,
        0.25881904510252074, 0.25881904510252074, 0.25881904510252074, 0.25881904510252074, 0.25881904510252074,
        0.25881904510252074, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
        0.258819045102521, 0.258819045102521, 0.258819045102521, 0.258819045102521, 0.258819045102521,
        0.258819045102521, 0.258819045102521, 0.258819045102521, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        -0.258819045102521, -0.258819045102521, -0.258819045102521, -0.258819045102521, -0.258819045102521,
        -0.258819045102521, -0.258819045102521, -0.258819045102521, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5,
        -0.374606593415912, -0.374606593415912, -0.374606593415912, -0.374606593415912, -0.374606593415912,
        -0.12186934340514748, -0.12186934340514748, -0.12186934340514748, -0.12186934340514748, -0.12186934340514748,
        0.12186934340514748, 0.12186934340514748, 0.12186934340514748, 0.12186934340514748, 0.12186934340514748,
        0.374606593415912, 0.374606593415912, 0.374606593415912, 0.374606593415912, 0.374606593415912,
        0.37460659341591224, 0.37460659341591224, 0.37460659341591224, 0.37460659341591224, 0.37460659341591224,
        0.12186934340514755, 0.12186934340514755, 0.12186934340514755, 0.12186934340514755, 0.12186934340514755,
        -0.12186934340514755, -0.12186934340514755, -0.12186934340514755, -0.12186934340514755, -0.12186934340514755,
        -0.37460659341591224, -0.37460659341591224, -0.37460659341591224, -0.37460659341591224, -0.37460659341591224,
        0.9063077870366499, 0.9063077870366499, 0.9063077870366499, 1.0, 1.0, 1.0, 0.90630778703665, 0.90630778703665,
        0.90630778703665, -0.7071067811865476, -0.7071067811865476, -0.7071067811865476, -0.8660254037844386,
        -0.8660254037844386, -0.8660254037844386, -0.9659258262890683, -0.9659258262890683, -0.9659258262890683,
        -1.0, -1.0, -1.0, -0.9659258262890683, -0.9659258262890683, -0.9659258262890683, -0.8660254037844387,
        -0.8660254037844387, -0.8660254037844387, -0.7071067811865476, -0.7071067811865476, -0.7071067811865476,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0
  };

// Which zone does this marker belong to?
static uint8_t eval_zones[AMOUNT_OF_SLICES] = {
        2, 3, 3, 3, 3, 3, 4, 4, 2, 3, 3, 3, 3, 4, 4, 4, 2, 3, 3, 3, 4, 4, 4, 4, 2, 3, 3, 3, 3, 4, 4, 4, 2, 3, 3, 3, 3,
        3, 4, 4, 2, 1, 1, 1, 1, 1, 0, 0, 2, 1, 1, 1, 1, 0, 0, 0, 2, 1, 1, 1, 1, 0, 0, 0, 2, 1, 1, 1, 1, 0, 0, 0, 2, 1,
        1, 1, 1, 1, 0, 0, 3, 3, 4, 4, 4, 3, 3, 4, 4, 4, 3, 3, 4, 4, 4, 3, 3, 4, 4, 4, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1,
        1, 0, 0, 0, 1, 1, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
        2, 1, 2, 2, 2, 2, 2, 3, 1, 3, 1, 4, 0, 4, 0, 2, 2, 2, 2, 2, 3, 1, 3, 1, 4, 0, 4, 0, 2, 2, 2, 2, 2, 3, 1, 3, 1,
        4, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
  };

// Heading command for each zone
static float zone_headings[5] = {-M_PI/10, -M_PI/14, 0, M_PI/14, M_PI/10};
// Normalisation for amount of markers per zone
static float zone_amount_of_markers[5] = {29, 37, 75, 36, 30};

// Struct to save image for one frame
struct image_t previous_image;

// The current image is the image of the current frame given by the image pipeline. The buffer of this image is copied
// to previous image at the end of the pipeline function. The location of the current image moves between about
// 5-6 memory locations. The location of the previous image is set by us and stays at the same place, so only needs one
// cache. To speed up calculations, the pointer to left top pixel of every slice in the buffer is precalculated
// and stored, for both the current image and the previous image. We check which pointer to the image we get for both
// images and use the array of pointers associated with that one. If it is new, we recalculate for that location and
// remove the oldest cache.

// Array of pointers to an array of pointers to locations in buffer of the current image
uint8_t **current_buf_locations_p[AMOUNT_OF_IMAGE_BUFFERS];
// Which buffer slot was last used
uint8_t latest_added_buf_i;

// Array of pointers to locations in the previous image
uint8_t **previous_buf_locations_p;

// The previously used pointer to the buffer of the image
uint8_t *previous_current_image_buf_p[AMOUNT_OF_IMAGE_BUFFERS];
uint8_t *previous_previous_image_buf_p;

// Pointer to array of steps that are not aligned in YUV and need to be shifted
bool *color_shift_p;

// Mutex for writing the results
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

uint8_t *find_current_buf_locations(struct image_t *image_p, uint8_t **local_buf_locations_p[AMOUNT_OF_IMAGE_BUFFERS],
                                    uint8_t slot);
uint8_t *find_previous_buf_locations(struct image_t *image_p, uint8_t **local_buf_locations_p);

uint8_t *find_current_buf_locations(struct image_t *image_p, uint8_t **local_buf_locations_p[AMOUNT_OF_IMAGE_BUFFERS],
                                    uint8_t slot) {
  /* This function saves the pointers to the left top pixels of all
   * necessary windows and steps from these windows of the current
   * image in an array, such that these don't have to be recomputed
   * every frame.
   * @return - Pointer to the image buffer
   * */
  VERBOSE_PRINT("Reconstructing new image buffer pointer array\n");
  uint8_t *img_buf = (uint8_t *) image_p->buf;

  // 8bits are enough because Slices and Steps are < 256
  for (uint8_t slice_i = 0; slice_i < AMOUNT_OF_SLICES; slice_i++) {
    for (uint8_t step_i = 0; step_i < AMOUNT_OF_STEPS; step_i++) {
      // x and y are swapped here because the image is rotated
      uint16_t x_center = y_eval_locations[slice_i] + (uint16_t)(step_i * y_eval_directions[slice_i]);
      uint16_t y_center = x_eval_locations[slice_i] + (uint16_t)(step_i * x_eval_directions[slice_i]);

      local_buf_locations_p[slot][slice_i * step_i] =
              img_buf + 2 * image_p->w * (y_center - slice_extend) + 2 * (x_center - slice_extend);
    }
  }

  return img_buf;
}

uint8_t *find_previous_buf_locations(struct image_t *image_p, uint8_t **local_buf_locations_p) {
  /* This function saves the pointers to the left top pixels of all
   * necessary windows of the previous image in an array, such that
   * these don't have to be recomputed every frame
   * @return - Pointer to the image buffer
   * */
  VERBOSE_PRINT("Reconstructing saved image buffer pointer array\n");
  uint8_t *img_buf = (uint8_t *) image_p->buf;

  for (uint8_t slice_i = 0; slice_i < AMOUNT_OF_SLICES; slice_i++) {
    // x and y are swapped here because the image is rotated
    local_buf_locations_p[slice_i] =
            img_buf + 2 * image_p->w * (x_eval_locations[slice_i] - slice_extend)
                    + 2 * (y_eval_locations[slice_i] - slice_extend);
  }

  return img_buf;
}


static struct image_t *corr_depth_finder(struct image_t *current_image_p) {
  /*
   * corr_depth_finder
   * This function is inserted into the image pipeline and runs every frame.
   * It computes the correlation between a window of the previous image and
   * multiple windows in the current image. Then it calculates the st.dev. of
   * the correlation over these windows and determines if this is large enough
   * to consider this window for the next step. In the next step, the window
   * with the largest correlation is chosen as the most likely candidate, and
   * the amount of steps is used to give an indication how far this point has
   * moved. This is then used to see how close/busy 5 regions divided over the
   * image are. If an area is sufficiently busy, a steering impulse is given to
   * the drone to avoid this region.
   * @param img  - The input image to process
   * @return img - The processed image. The (very rough) depth estimates of the
   *               evaluated windows are added as drawn pixels, where white
   *               means close.
   */

  clock_t start = clock();

  // make shortcut to image buffers
  uint8_t *current_buf = (uint8_t *) current_image_p->buf;
  uint8_t *previous_buf = (uint8_t *) previous_image.buf;

  // if the image is at a different spot in the memory compared to last time, search for the new pointers,
  // otherwise assume that all pointers to the pixels are still valid.
  uint8_t current_prev_curr_image_buf_i;
  bool buffer_found = false;
  for (uint8_t buf_i = 0; buf_i < AMOUNT_OF_IMAGE_BUFFERS; buf_i++) {
    if (current_buf == previous_current_image_buf_p[buf_i]) {
      current_prev_curr_image_buf_i = buf_i;
      buffer_found = true;
    }
  }

  if (!buffer_found) {
    // Select new slot in the buffer
    latest_added_buf_i++;
    if (latest_added_buf_i > AMOUNT_OF_IMAGE_BUFFERS - 1) {
      latest_added_buf_i = 0;
    }
    VERBOSE_PRINT("Adding new buffer in slot %d\n", latest_added_buf_i);
    previous_current_image_buf_p[latest_added_buf_i] =
            find_current_buf_locations(current_image_p, current_buf_locations_p, latest_added_buf_i);
    current_prev_curr_image_buf_i = latest_added_buf_i;
  }

  // Check same thing for previous image
  if (previous_buf != previous_previous_image_buf_p) {
    previous_previous_image_buf_p = find_previous_buf_locations(&previous_image, previous_buf_locations_p);
  }

  // Array to save the depth estimates for every window
  float depths[AMOUNT_OF_SLICES];
  memset(&depths, 0.0f, AMOUNT_OF_SLICES * sizeof(float));

  // Array to save the correlations for each step of a single window
  float correlations[AMOUNT_OF_STEPS];
  float mean, std;

  for (uint8_t slice_i = 0; slice_i < AMOUNT_OF_SLICES; slice_i++) {
    memset(&correlations, 0.0f, AMOUNT_OF_STEPS * sizeof(float));
    mean = std = 0.0f;

    #if SIMD_ENABLED == TRUE  // === SIMD VARIANT OF THE FUNCTION ===
      // Load the window from the previous image that stays the same with each step

      // Four arrays of 16 vectors of each 8 times an uint8 (when slize size = 16)
      // Four arrays are needed because every pixel has a Y and U/V value.
      // Total dimension is 16x(4x8) uint8's for 16x32 values for 16x16 pixels
      uint8x8_t previous_buf_vec_1[SLICE_SIZE], previous_buf_vec_2[SLICE_SIZE],
              previous_buf_vec_3[SLICE_SIZE], previous_buf_vec_4[SLICE_SIZE];
      // The alternate windows are shifted with one pixel, to fix alignment issue
      // with windows that start with either U or V.
      uint8x8_t previous_buf_vec_1_alt[SLICE_SIZE], previous_buf_vec_2_alt[SLICE_SIZE],
              previous_buf_vec_3_alt[SLICE_SIZE], previous_buf_vec_4_alt[SLICE_SIZE];

      for (uint8_t slice_line = 0; slice_line < 16; slice_line++) {
        uint16_t buffer_offset = 2 * current_image_p->w * slice_line;

        // Load 4x8 values from the image buffer
        previous_buf_vec_1[slice_line] = vld1_u8(previous_buf_locations_p[slice_i] + buffer_offset);
        previous_buf_vec_2[slice_line] = vld1_u8(previous_buf_locations_p[slice_i] + buffer_offset + 8);
        previous_buf_vec_3[slice_line] = vld1_u8(previous_buf_locations_p[slice_i] + buffer_offset + 16);
        previous_buf_vec_4[slice_line] = vld1_u8(previous_buf_locations_p[slice_i] + buffer_offset + 24);

        // Load 4x8 values from the image buffer, shifted by one pixel
        previous_buf_vec_1_alt[slice_line] = vld1_u8(previous_buf_locations_p[slice_i] + buffer_offset + 2);
        previous_buf_vec_2_alt[slice_line] = vld1_u8(previous_buf_locations_p[slice_i] + buffer_offset + 2 + 8);
        previous_buf_vec_3_alt[slice_line] = vld1_u8(previous_buf_locations_p[slice_i] + buffer_offset + 2 + 16);
        previous_buf_vec_4_alt[slice_line] = vld1_u8(previous_buf_locations_p[slice_i] + buffer_offset + 2 + 24);
      }
    #endif

    for (uint8_t step_i = 0; step_i < AMOUNT_OF_STEPS; step_i++) {

      // Y,UV = color coordinates
      // x,y  = pixel coordinates

      #if SIMD_ENABLED == TRUE  // === SIMD VARIANT OF THE FUNCTION ===
        // Load the window from the current image for this specific step and perform calculations

        for (uint8_t slice_line = 0; slice_line < SLICE_SIZE; slice_line++) {
          // Window multiplied line by line using NEON Intrinsics

          uint16_t buffer_offset = 2 * current_image_p->w * slice_line;

          uint8x8_t current_buf_vec_1, current_buf_vec_2, current_buf_vec_3, current_buf_vec_4;
          // Load 4x8 values from the image buffer.
          current_buf_vec_1  = vld1_u8(current_buf_locations_p[current_prev_curr_image_buf_i][slice_i * step_i]
                                                                                                  + buffer_offset);
          current_buf_vec_2  = vld1_u8(current_buf_locations_p[current_prev_curr_image_buf_i][slice_i * step_i]
                                                                                                  + buffer_offset + 8);
          current_buf_vec_3  = vld1_u8(current_buf_locations_p[current_prev_curr_image_buf_i][slice_i * step_i]
                                                                                                  + buffer_offset + 16);
          current_buf_vec_4  = vld1_u8(current_buf_locations_p[current_prev_curr_image_buf_i][slice_i * step_i]
                                                                                                  + buffer_offset + 24);

          uint16x8_t multiplied_1, multiplied_2, multiplied_3, multiplied_4;
          // Multiply entire rows of the slices element-wise and widen
          if (color_shift_p[slice_i * step_i]) {
            multiplied_1 = vmull_u8(current_buf_vec_1, previous_buf_vec_1[slice_line]);
            multiplied_2 = vmull_u8(current_buf_vec_2, previous_buf_vec_2[slice_line]);
            multiplied_3 = vmull_u8(current_buf_vec_3, previous_buf_vec_3[slice_line]);
            multiplied_4 = vmull_u8(current_buf_vec_4, previous_buf_vec_4[slice_line]);
          } else {
            multiplied_1 = vmull_u8(current_buf_vec_1, previous_buf_vec_1_alt[slice_line]);
            multiplied_2 = vmull_u8(current_buf_vec_2, previous_buf_vec_2_alt[slice_line]);
            multiplied_3 = vmull_u8(current_buf_vec_3, previous_buf_vec_3_alt[slice_line]);
            multiplied_4 = vmull_u8(current_buf_vec_4, previous_buf_vec_4_alt[slice_line]);
          }

          // split into half for Horizontal Add
          uint16x4_t multiplied_1_low  =  vget_low_u16(multiplied_1);
          uint16x4_t multiplied_1_high = vget_high_u16(multiplied_1);
          uint16x4_t multiplied_2_low  =  vget_low_u16(multiplied_2);
          uint16x4_t multiplied_2_high = vget_high_u16(multiplied_2);
          uint16x4_t multiplied_3_low  =  vget_low_u16(multiplied_3);
          uint16x4_t multiplied_3_high = vget_high_u16(multiplied_3);
          uint16x4_t multiplied_4_low  =  vget_low_u16(multiplied_4);
          uint16x4_t multiplied_4_high = vget_high_u16(multiplied_4);

          // HADD Tree step 1
          uint32x4_t added_1_1 = vaddl_u16(multiplied_1_low,  multiplied_2_low);
          uint32x4_t added_1_2 = vaddl_u16(multiplied_3_low,  multiplied_4_low);
          uint32x4_t added_1_3 = vaddl_u16(multiplied_1_high, multiplied_2_high);
          uint32x4_t added_1_4 = vaddl_u16(multiplied_3_high, multiplied_4_high);

          // HADD Tree step 2
          uint32x4_t added_2_1 = vaddq_u32(added_1_1,  added_1_2);
          uint32x4_t added_2_2 = vaddq_u32(added_1_3,  added_1_4);

          // HADD Tree step 3
          uint32x4_t added_3_1 = vaddq_u32(added_2_1,  added_2_2);

          // Unload from SIMD registers
          for (uint8_t vec_i = 0; vec_i < 4; vec_i++) {
            correlations[step_i] += (float) added_3_1[vec_i];
          }
        }

      #else  // === NORMAL VARIANT OF THE FUNCTION ===

        uint8_t *previous_Y_slice_p, *previous_UV_slice_p, *current_Y_slice_p, *current_UV_slice_p;

        // Just simply multiply pixel by pixels and accumulate
        for (uint32_t x_slice = 0; x_slice < SLICE_SIZE; x_slice++) {
          for (uint32_t y_slice = 0; y_slice < SLICE_SIZE; y_slice++) {
            uint32_t buffer_offset = 2 * current_image_p->w * y_slice + 2 * x_slice;
            // Y_eval_e
            previous_Y_slice_p  = previous_buf_locations_p[slice_i] + buffer_offset + 1;
            current_Y_slice_p   = current_buf_locations_p[current_prev_curr_image_buf_i][slice_i * step_i]
                                                                                                    + buffer_offset + 1;
            correlations[step_i] += (float) (*current_Y_slice_p * *previous_Y_slice_p);
            // UV
            previous_UV_slice_p = previous_buf_locations_p[slice_i] + buffer_offset;
            current_UV_slice_p  = current_buf_locations_p[current_prev_curr_image_buf_i][slice_i * step_i]
                                                                                                        + buffer_offset;
            correlations[step_i] += (float) (*(current_UV_slice_p + 2 * color_shift_p[slice_i * step_i])
                    * *previous_UV_slice_p);
          }
        }

      #endif

      // Change range to 0...1 instead of 0...(2*16*16*255*255)
      correlations[step_i] /= 2 * SLICE_SIZE * SLICE_SIZE * 255 * 255;

      // Also keep track total sum for mean
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

      // Output a normalised "depth" estimate
      depths[slice_i] = (float) max_i / (float) AMOUNT_OF_STEPS;

      if (draw) {
        current_buf[2 * current_image_p->w * x_eval_locations[slice_i] + 2 * y_eval_locations[slice_i] + 1] =
                (uint8_t)(depths[slice_i] * 255);
      }
    }
  }

  // Add all windows to their corresponding zones
  float zone_busyness[5] = {0, 0, 0, 0, 0};
  for (uint8_t slice_i = 0; slice_i < AMOUNT_OF_SLICES; slice_i++) {
    zone_busyness[eval_zones[slice_i]] += depths[slice_i];
  }

  // Find best zone to go to if the zone is busy enough
  uint8_t min_zone_i = 2;
  for (uint8_t zone_i = 0; zone_i < 5; zone_i++) {
    zone_busyness[zone_i] /= zone_amount_of_markers[zone_i];
    if (zone_busyness[zone_i] < zone_busyness[min_zone_i] && zone_busyness[zone_i] > cdf_threshold) {
      // ^ "Flown" mistake: It now only checks the busyness of the minimum zone, which should not be busy, it should
      //                    have checked the busyness of the other zones
      min_zone_i = zone_i;
    }
  }

  // Write findings to mutex
  pthread_mutex_lock(&mutex);
  global_corr_heading_object.best_heading = zone_headings[min_zone_i];
  global_corr_heading_object.zone_busyness_ll = zone_busyness[0];
  global_corr_heading_object.zone_busyness_l = zone_busyness[1];
  global_corr_heading_object.zone_busyness_m = zone_busyness[2];
  global_corr_heading_object.zone_busyness_r = zone_busyness[3];
  global_corr_heading_object.zone_busyness_rr = zone_busyness[4];
  global_corr_heading_object.updated = true;
  pthread_mutex_unlock(&mutex);

  // Copy current image to previous image
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
  /*
   * This function initialises everything needed for Corr Depth Finder and subscribes the main function to the image
   * pipeline
   */
  register_periodic_telemetry(DefaultPeriodic, PPRZ_MSG_ID_CORR_DEPTH_FINDER, send_corr_depth_finder);

  // Default values
  cdf_max_std = 0.008;
  cdf_threshold = 0.6f;

  // Allocate space for all pointer arrays
  for (uint8_t buf_i = 0; buf_i < AMOUNT_OF_IMAGE_BUFFERS; buf_i++) {
    current_buf_locations_p[buf_i] = malloc(AMOUNT_OF_SLICE_STEPS * sizeof(uint8_t *));
  }
  previous_buf_locations_p = malloc(AMOUNT_OF_STEPS * sizeof(uint8_t*));
  latest_added_buf_i = 0;

  // Allocate space for the LUT to save which windows need to shift to correct UV-misalignment
  color_shift_p = malloc(AMOUNT_OF_SLICE_STEPS * sizeof(bool));

  // And save which windows need this
  for (uint8_t slice_i = 0; slice_i < AMOUNT_OF_SLICES; slice_i++) {
    for (uint8_t step_i = 0; step_i < AMOUNT_OF_STEPS; step_i++) {
      // Shift color to correct for U and V unevenness
      if ((step_i % 2 == 0 && slice_i % 2 == 0) || (step_i % 2 != 0 && slice_i % 2 != 0)) {
        color_shift_p[slice_i * step_i] = false;
      } else {
        color_shift_p[slice_i * step_i] = true;
      }
    }
  }

  memset(&global_corr_heading_object, 0, sizeof(struct heading_object_t));

  pthread_mutex_init(&mutex, NULL);

  // Set all saved image pointers to null pointers since we did not save any yet
  for (uint8_t buf_i = 0; buf_i < AMOUNT_OF_IMAGE_BUFFERS; buf_i++) {
    previous_current_image_buf_p[buf_i] = NULL;
  }
  previous_previous_image_buf_p = NULL;

  // Allocate space for the previous image
  image_create(&previous_image, 240, 520, IMAGE_YUV422);

  // Subscribe our function to the image pipeline
  cv_add_to_device(&DEPTHFINDER_CAMERA, corr_depth_finder1, DEPTHFINDER_FPS, 0);
}

void corr_depth_finder_periodic(void) {
  /*
   * This function sends the results of the Corr Depth Finder to the Green Follower
   */
  static struct heading_object_t local_heading_object;

  pthread_mutex_lock(&mutex);
  memcpy(&local_heading_object, &global_corr_heading_object, sizeof(struct heading_object_t));
  pthread_mutex_unlock(&mutex);

  if(local_heading_object.updated) {
    AbiSendMsgDEPTH_FINDER_HEADING(CORR_DEPTH_ID, local_heading_object.best_heading);
    local_heading_object.updated = false;
  }
}
