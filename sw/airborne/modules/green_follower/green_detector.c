//
// Created by lilly on 6-3-24.
//

#include "green_detector.h"
#include "modules/computer_vision/cv.h"
#include "modules/core/abi.h"

#include <stdbool.h>
#include <math.h>
#include "pthread.h"

static pthread_mutex_t mutex;

struct heading_object_t {
    float best_heading;
    float safe_length;
    bool updated;
};

struct heading_object_t global_heading_object;

void apply_threshold(struct image_t *img,
                     uint8_t lum_min, uint8_t lum_max,
                     uint8_t cb_min, uint8_t cb_max,
                     uint8_t cr_min, uint8_t cr_max);

float get_radial(struct image_t *img, float angle, uint8_t radius);

void get_direction(struct image_t *img, uint8_t resolution);

/*
 * object_detector
 * @param img - input image to process
 * @param filter - which detection filter to process
 * @return img
 */
static struct image_t *green_heading_finder(struct image_t *img)
{
    uint8_t lum_min, lum_max;
    uint8_t cb_min, cb_max;
    uint8_t cr_min, cr_max;
    bool draw;

    float best_heading, safe_length;

    lum_min = 60;
    lum_max = 130;
    cb_min = 75;
    cb_max = 110;
    cr_min = 120;
    cr_max = 140;
    draw = true;
    uint8_t scan_resolution = 50;

    // Filter the image so that all green pixels have a y value of 255 and all others a y value of 0
    apply_threshold(img, lum_min, lum_max, cb_min, cb_max, cr_min, cr_max);
    // Scan in radials from the centre bottom of the image to find the direction with the most green pixels
    get_direction(img, scan_resolution, &best_heading, &safe_length)

    // Filter and find centroid
    pthread_mutex_lock(&mutex);
    global_heading_object.best_heading = best_heading;
    global_heading_object.safe_length = safe_length;
    global_heading_object.updated = true;
    pthread_mutex_unlock(&mutex);
    return img;
}

void green_detector_init(void) {
    memset(global_heading_object, 0, sizeof(struct heading_object_t));
    pthread_mutex_init(&mutex, NULL);

    cv_add_to_device(?, green_heading_finder, 0, 0);

}

void green_detector_periodic(void) {

}


void apply_threshold(struct image_t *img,
                     uint8_t lum_min, uint8_t lum_max,
                     uint8_t cb_min, uint8_t cb_max,
                     uint8_t cr_min, uint8_t cr_max)
{
  uint8_t *buffer = img->buf;

  // Go through all the pixels
  for (uint16_t y = 0; y < img->h; y++) {
    for (uint16_t x = 0; x < img->w; x ++) {
      // Check if the color is inside the specified values
      uint8_t *yp, *up, *vp;
      if (x % 2 == 0) {
        // Even x
        up = &buffer[y * 2 * img->w + 2 * x];      // U
        yp = &buffer[y * 2 * img->w + 2 * x + 1];  // Y1
        vp = &buffer[y * 2 * img->w + 2 * x + 2];  // V
        //yp = &buffer[y * 2 * img->w + 2 * x + 3]; // Y2
      } else {
        // Uneven x
        up = &buffer[y * 2 * img->w + 2 * x - 2];  // U
        //yp = &buffer[y * 2 * img->w + 2 * x - 1]; // Y1
        vp = &buffer[y * 2 * img->w + 2 * x];      // V
        yp = &buffer[y * 2 * img->w + 2 * x + 1];  // Y2
      }
      if ( (*yp >= lum_min) && (*yp <= lum_max) &&
           (*up >= cb_min ) && (*up <= cb_max ) &&
           (*vp >= cr_min ) && (*vp <= cr_max )) {
        *yp = 255;  // make pixel white
        }
      else {
        *yp = 0; // make pixel black
      }
    }
  }
}

float get_radial(struct image_t *img, float angle, uint8_t radius) {
    uint8_t *buffer = img->buf;

    uint8_t sum;
    for (uint8_t i = 0; i < radius; i++) {
        y = img->h - i * (uint8_t)sin(angle);
        x = img->w / 2 + i * (uint8_t)cos(angle);
        sum = sum + &buffer[y * 2 * img->w + 2 * x + 1]
    }

    return (float)sum * (sin(angle) + 0.2)
}

void get_direction(struct image_t *img, uint8_t resolution, float* best_heading, float* safe_length) {

    float step_size = M_PI / (float)resolution;
    *best_heading = 0;
    *safe_length = 0;

    for (float angle = 0; angle < M_PI; angle += step_size) {
        float radial = get_radial(img, angle, img->w / 2);

        if (radial >= *safe_length) {
            *best_heading = angle;
            *safe_length = radial;
        }
    }
}

