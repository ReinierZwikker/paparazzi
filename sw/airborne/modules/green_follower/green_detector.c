//
// Created by lilly on 6-3-24.
//

#include "green_detector.h"
#include "modules/computer_vision/cv.h"
#include "modules/core/abi.h"

#include <stdbool.h>
#include "pthread.h"

static pthread_mutex_t mutex;

struct heading_object_t {
    float best_heading;
    float safe_length;
    bool updated;
};

struct heading_object_t global_heading_object;


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

    lum_min = 0;
    lum_max = 255;
    cb_min = 0;
    cb_max = 255;
    cr_min = 0;
    cr_max = 255;
    draw = true;

    // Filter and find centroid
    uint32_t best_heading = find_object_centroid(img, &x_c, &y_c, draw, lum_min, lum_max, cb_min, cb_max, cr_min, cr_max);

    pthread_mutex_lock(&mutex);
    global_heading_object.best_heading = 0;
    global_heading_object.safe_length = 0;
    global_heading_object.updated = true;
    pthread_mutex_unlock(&mutex);

    return img;
}

void green_detector_init(void) {
    memset(global_filters, 0, 2*sizeof(struct color_object_t));
    pthread_mutex_init(&mutex, NULL);

    cv_add_to_device(&COLOR_OBJECT_DETECTOR_CAMERA1, green_heading_finder, 0, 0);

}

void green_detector_periodic(void) {

}
