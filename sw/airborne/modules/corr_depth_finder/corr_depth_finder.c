
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

#ifndef DEPTHFINDER_FPS
#define DEPTHFINDER_FPS 0       ///< Default FPS (zero means run at camera fps)
#endif
PRINT_CONFIG_VAR(DEPTHFINDER_FPS)

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

// Predefined evaluation location, direction and dependency on forward or sideways motion
const struct eval_locations {
    // x-components of location in pixels
    uint32_t[207] x_locations = {285, 311, 337, 363, 381, 398, 415, 288, 317, 346, 375, 395, 414, 433, 462, 290, 320,
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
    uint32_t[207] y_locations = {105,  90,  75,  60,  50,  40,  30, 112, 104,  96,  88,  83,  78,  73,  65, 120, 120,
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
    float_t[207] x_directions = {8.66025404e-01, 8.66025404e-01, 8.66025404e-01, 8.66025404e-01,
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
    float_t[207] y_directions = {-5.00000000e-01, -5.00000000e-01, -5.00000000e-01, -5.00000000e-01,
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
    bool[207] dependencies = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                              1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                              1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
};
const struct eval_locations global_eval_locations;

struct image_t previous_image;
struct image_t *previous_image_p;

static pthread_mutex_t mutex;

struct depth_object_t {
    float[207] depth;
    bool updated;
};
struct depth_object_t global_depth_object;



//void apply_threshold(struct image_t *img, uint32_t *green_pixels,
//                     uint8_t lum_min, uint8_t lum_max,
//                     uint8_t cb_min, uint8_t cb_max,
//                     uint8_t cr_min, uint8_t cr_max);
//
//float get_radial(struct image_t *img, float angle, uint8_t radius);
//
//void get_direction(struct image_t *img, uint8_t resolution, float* best_heading, float* safe_length);

/*
 * object_detector
 * @param img - input image to process
 * @return img
 */
static struct image_t *corr_depth_finder(struct image_t *current_image_p)
{
  struct depth_object_t local_depth_object;
  struct image_t previous_slice, current_slice;

  struct point_t slice_center, step_center;

  for (uint8_t slice_i = 0; slice_i < 207; slice_i++) {
    slice_center.x = global_eval_locations.x_directions[slice_i];
    slice_center.y = global_eval_locations.y_directions[slice_i];

    step_center = slice_center;

    image_window(previous_image_p, &previous_slice, &slice_center);

    uint32_t delta_x = ;
    uint32_t delta_y = ;


    for (uint8_t step_i = 0; step_i < 207; step_i++) {
      step_center.x +=
      image_window(current_image_p, &current_slice, &step_center);

    }

  }

  // TODO Do stuff here

  pthread_mutex_lock(&mutex);
  // TODO Write to mutex here
  global_depth_object.updated = true;
  pthread_mutex_unlock(&mutex);

  // Possible optimisation: Only save necessary slices?
  memcpy(previous_image_p, img_p, sizeof(struct image_t));

  return img_p;
}

struct image_t *corr_depth_finder1(struct image_t *img, uint8_t camera_id);
struct image_t *corr_depth_finder1(struct image_t *img, uint8_t camera_id __attribute__((unused)))
{
  return corr_depth_finder(img);
}

void corr_depth_finder_init(void) {
  memset(&global_depth_object, 0, sizeof(struct depth_object_t));
  pthread_mutex_init(&mutex, NULL);

  slice->h = slice->w = slice_size;

  previous_image_p = malloc(previous_image);

  //#ifdef GREEN_DETECTOR_LUM_MIN
  //  gd_lum_min = GREEN_DETECTOR_LUM_MIN;
  //        gd_lum_max = GREEN_DETECTOR_LUM_MAX;
  //        gd_cb_min = GREEN_DETECTOR_CB_MIN;
  //        gd_cb_max = GREEN_DETECTOR_CB_MAX;
  //        gd_cr_min = GREEN_DETECTOR_CR_MIN;
  //        gd_cr_max = GREEN_DETECTOR_CR_MAX;
  //#endif

  cv_add_to_device(&CORR_DEPTH_FINDER_CAMERA, corr_depth_finder1, DEPTHFINDER_FPS, 0);
}

void corr_depth_finder_periodic(void) {
  static struct depth_object_t local_depth_object;
  pthread_mutex_lock(&mutex);
  memcpy(&local_depth_object, &global_depth_object, sizeof(struct depth_object_t));
  pthread_mutex_unlock(&mutex);

  if(local_depth_object.updated){
    AbiSendMsgCORR_DEPTH_FINDER(CORR_DEPTH_ID, <message here>);
    local_depth_object.updated = false;
  }
}



//
//
//void apply_threshold(struct image_t *img, uint32_t* green_pixels,
//                     uint8_t lum_min, uint8_t lum_max,
//                     uint8_t cb_min, uint8_t cb_max,
//                     uint8_t cr_min, uint8_t cr_max)
//{
//  uint8_t *buffer = img->buf;
//  uint32_t local_green_pixels = 0;
//
//  // Go through all the pixels
//  for (uint16_t y = 0; y < img->h; y++) {
//    for (uint16_t x = 0; x < img->w; x ++) {
//      // Check if the color is inside the specified values
//      uint8_t *yp, *up, *vp;
//      if (x % 2 == 0) {
//        // Even x
//        up = &buffer[y * 2 * img->w + 2 * x];      // U
//        yp = &buffer[y * 2 * img->w + 2 * x + 1];  // Y1
//        vp = &buffer[y * 2 * img->w + 2 * x + 2];  // V
//        //yp = &buffer[y * 2 * img->w + 2 * x + 3]; // Y2
//      } else {
//        // Uneven x
//        up = &buffer[y * 2 * img->w + 2 * x - 2];  // U
//        //yp = &buffer[y * 2 * img->w + 2 * x - 1]; // Y1
//        vp = &buffer[y * 2 * img->w + 2 * x];      // V
//        yp = &buffer[y * 2 * img->w + 2 * x + 1];  // Y2
//      }
//      if ( (*yp >= lum_min) && (*yp <= lum_max) &&
//           (*up >= cb_min ) && (*up <= cb_max ) &&
//           (*vp >= cr_min ) && (*vp <= cr_max )) {
//        local_green_pixels++;
//        *yp = 255;  // make pixel white
//      }
//      else {
//        *yp = 0; // make pixel black
//      }
//    }
//  }
//  *green_pixels = local_green_pixels;
//}
//
//float get_radial(struct image_t *img, float angle, uint8_t radius) {
//  uint8_t *buffer = img->buf;
//
//  uint32_t sum = 0;
//  uint16_t x, y;
//
//  for (double i = 0; i < radius; i++) {
//    y = (uint16_t)((double)img->h - i * sin(angle));
//    x = (uint16_t)((double)img->w / 2 + i * cos(angle));
//    if (buffer[y * 2 * img->w + 2 * x + 1] == 255) {
//      sum++;
//    }
//  }
//
//  return (float)sum; // * (sin(angle) + 0.2) ;
//}
//
//void get_direction(struct image_t *img, uint8_t resolution, float* best_heading, float* safe_length) {
//
//  float step_size = M_PI / (float)resolution;
//  *best_heading = 0;
//  *safe_length = 0;
//
//  for (float angle = 0.001; angle < M_PI; angle += step_size) {
//    float radial = get_radial(img, angle, img->h - 20);
//
//    // VERBOSE_PRINT("GF: Radial %f is %f\n", angle, radial);
//
//
//    if (radial >= *safe_length) {
//      *best_heading = angle;
//      *safe_length = radial;
//    }
//  }
//
//  *best_heading = M_PI/2 - *best_heading;
//}
