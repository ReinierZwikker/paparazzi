#include "modules/green_follower/green_detector.h"
#include "modules/computer_vision/video_capture.h"
#include "modules/computer_vision/cv.h"
#include "modules/core/abi.h"
#include "modules/datalink/telemetry.h"
#include "std.h"

#include <stdio.h>
#include <stdbool.h>
#include <math.h>
#include "pthread.h"

#define SIMD_ENABLED FALSE

#if SIMD_ENABLED == TRUE
#include "arm_neon.h"
#endif

#ifndef GREENFILTER_FPS
#define GREENFILTER_FPS 0       ///< Default FPS (zero means run at camera fps)
#endif
PRINT_CONFIG_VAR(COLORFILTER_FPS)

#define GREEN_DETECTOR_VERBOSE TRUE

#define PRINT(string,...) fprintf(stderr, "[green_follower->%s()] " string,__FUNCTION__ , ##__VA_ARGS__)
#if GREEN_DETECTOR_VERBOSE
#define VERBOSE_PRINT PRINT
#else
#define VERBOSE_PRINT(...)
#endif


// TODO Make auto-select based on build target
#define CYBERZOO_FILTER FALSE
#if CYBERZOO_FILTER
// Filter Settings CYBERZOO
uint8_t gd_lum_min = 60;
uint8_t gd_lum_max = 140;
uint8_t gd_cb_min = 24;
uint8_t gd_cr_min = 100;
uint8_t gd_cb_max = 28;
uint8_t gd_cr_max = 160;
#else
// Filter Settings NPS/GAZEBO
uint8_t gd_lum_min = 60;
uint8_t gd_lum_max = 110;
uint8_t gd_cb_min = 75;
uint8_t gd_cb_max = 110;
uint8_t gd_cr_min = 110;
uint8_t gd_cr_max = 130;
#endif

float weight_function = 0.8;

int scan_resolution = 100; // Amount of radials
clock_t start_cycle_counter = 0; // Start timer for cycles_since_update
clock_t end_cycle_counter = 0; // End timer for cycles_since_update
static pthread_mutex_t mutex;

// Struct with relevant information for the navigation
struct heading_object_t {
    float best_heading;
    float safe_length;
    uint32_t green_pixels;
    float cycle_time;
    float cycles_since_update;
    bool updated;
};
struct heading_object_t global_heading_object;

#if SIMD_ENABLED == TRUE
struct threshold_object_t {
    uint8x16_t zero_array = vdup_n_u8(0);
    uint8x16_t one_array = vdup_n_u8(1);
    uint8x16_t array_224 = vdup_n_u8(224); // 224 = 11100000
    uint8x16_t array_14  = vdup_n_u8(14);  // 14  = 00001110
    uint8x16_t array_176 = vdup_n_u8(176); // 176 = 10110000
    uint8x16_t array_11  = vdup_n_u8(11);  // 11  = 00001011
    uint8_t select = {1, 2, 4, 8, 16, 32, 64, 128};
    uint16x8_t min_thresh;
    uint16x8_t max_thresh;
};
struct threshold_object_t gto;
#endif

void apply_threshold(struct image_t *img, uint32_t *green_pixels,
                     uint8_t lum_min, uint8_t lum_max,
                     uint8_t cb_min, uint8_t cb_max,
                     uint8_t cr_min, uint8_t cr_max);

float get_radial(struct image_t *img, float angle, uint8_t radius);

void get_direction(struct image_t *img, int resolution, float* best_heading, float* safe_length);

// Create telemetry message
static void send_green_follower(struct transport_tx *trans, struct link_device *dev) {
  static struct heading_object_t local_heading_object;
  pthread_mutex_lock(&mutex);
  memcpy(&local_heading_object, &global_heading_object, sizeof(struct heading_object_t));
  pthread_mutex_unlock(&mutex);

  pprz_msg_send_GREEN_FOLLOWER(trans, dev, AC_ID,
                               &local_heading_object.best_heading,
                               &local_heading_object.safe_length,
                               &local_heading_object.green_pixels,
                               &local_heading_object.cycle_time,
                               &local_heading_object.cycles_since_update);
}

/*
 * object_detector
 * @param img - input image to process
 * @return img
 */
static struct image_t *green_heading_finder(struct image_t *img)
{
    clock_t start = clock();

    uint8_t lum_min, lum_max;
    uint8_t cb_min, cb_max;
    uint8_t cr_min, cr_max;

    float best_heading, safe_length;

    uint32_t green_pixels;

    lum_min = gd_lum_min;
    lum_max = gd_lum_max;
    cb_min = gd_cb_min;
    cb_max = gd_cb_max;
    cr_min = gd_cr_min;
    cr_max = gd_cr_max;

    // Filter the image so that all green pixels have a y value of 255 and all others a y value of 0
    apply_threshold(img, &green_pixels, lum_min, lum_max, cb_min, cb_max, cr_min, cr_max);
    // Scan in radials from the centre bottom of the image to find the direction with the most green pixels
    get_direction(img, scan_resolution, &best_heading, &safe_length);

    pthread_mutex_lock(&mutex);
    global_heading_object.best_heading = best_heading;
    global_heading_object.safe_length = safe_length;
    global_heading_object.green_pixels = green_pixels;
    global_heading_object.updated = true;

    clock_t end = clock();
    global_heading_object.cycle_time = (end - start);
    pthread_mutex_unlock(&mutex);

    return img;
}

void apply_threshold(struct image_t *img, uint32_t* green_pixels,
                     uint8_t lum_min, uint8_t lum_max,
                     uint8_t cb_min, uint8_t cb_max,
                     uint8_t cr_min, uint8_t cr_max)
{
    uint8_t *buffer = img->buf;
    uint32_t local_green_pixels = 0;

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
                 (*vp >= cr_min ) && (*vp <= cr_max )){

                *yp = 255;  // make pixel brighter
                local_green_pixels++;

            }
            else {
                *yp = 0; // make pixel darker
            }

            //if (y == (img->h)/2+2*x || y == (img->h)/2-2*x){
            //*yp = 0; // make pixel red
            //*up = 0;
            //*vp = 255;
            //}
        }
    }
    *green_pixels = local_green_pixels;
}

float get_radial(struct image_t *img, float angle, uint8_t radius) {
    uint8_t *buffer = img->buf;

    uint32_t sum = 0;
    uint16_t x, y;

    for (double i = 0; i < radius; i++) {
        y = (uint16_t)((double)img->h/2 + i * cos(angle));
        x = (uint16_t)(i * sin(angle));
        if (buffer[y * 2 * img->w + 2 * x + 1] == 255) {
            sum += 5;
        } else {
            sum--;
        }
    }

    return (float)sum; // * (sin(angle) + 0.2) ;
}

#if SIMD_ENABLED == TRUE
void get_regions_opt(struct image_t *img, uint32_t* green_pixel) {
    uint8_t *buffer = img->buf;
    uint8_t band_sum[img->h]; // Sum for every vertical band in the image

    for (uint16_t y = 0; y < img->h; y++) {
        uint8 sum_vector[8];
        for uint8_t i = 0; i < 2; i++) {
            uint8x16_t greater_combined = gto.zero_array; // A uint8 vector with 16 values of which every bit represents a y, u or v value
            uint8x16_t smaller_combined = gto.zero_array; // A uint8 vector with 16 values of which every bit represents a y, u or v value
            for (uint16_t j = 0; j < 15; j++) {
                // Get a slice from the image buffer
                uint8x16_t slice = vld1q_u8(buffer + j * 8);

                // Get the bounds
                uint8x16_t greater = vcgtq_u8(slice, gto.min_thresh); // Sets all YUV values greater than min thresh to 1
                uint8x16_t smaller = vcgtq_u8(slice, gto.max_thresh); // Sets all YUV values smaller than max thresh to 1

                // Combine the boolean integers into one integer to be more efficient
                uint8x16_t greater_combined = vbslq_u8(vdup_n_u8(gto.select[j]), greater, greater_combined);
                uint8x16_t smaller_combined = vbslq_u8(vdup_n_u8(gto.select[j]), smaller, smaller_combined);
            }
            // Get the bitwise union between the greater_combined and smaller_combined vectors
            // The result is an array in which every bit represents a y, u or v value. If the bit is 1, the value is within the threshold.
            uint8x16_t bounded = vandq_u8(greater_combined, smaller_combined);

            // See if yuv pixel groups are all 1
            uint8x16_t even_left = vceqq_u8(gto.array_244, greater_combined);  // 244 = 11100000, thus the first uyv of all sets of 8 yuv are checked.
            uint8x16_t even_right = vceqq_u8(gto.array_14, greater_combined);  // 14 = 00001110, thus the second uyv of all sets of 8 yuv are checked.
            uint8x16_t uneven_left = vceqq_u8(gto.array_176, greater_combined);  // 176 = 10110000, thus the first uvy of all sets of 8 yuv are checked.
            uint8x16_t uneven_right = vceqq_u8(gto.array_11, greater_combined);  // 11 = 00001011, thus the second uvy of all sets of 8 yuv are checked.

            // Make it so the value is between 0 and 1 instead of 0 and 255, so that addition is possible within an uint8
            uint8x16_t even_left_filtered = vandq_u8(gto.one_array, even_left);
            uint8x16_t even_right_filtered = vandq_u8(gto.one_array, even_right);
            uint8x16_t uneven_left_filtered = vandq_u8(gto.one_array, uneven_left);
            uint8x16_t uneven_right_filtered = vandq_u8(gto.one_array, uneven_right);

            // Sum every pixel group
            sum_vector[0 + 4*i] = vaddvq_u8(even_left_filtered);
            sum_vector[1 + 4*i] = vaddvq_u8(even_right_filtered);
            sum_vector[2 + 4*i] = vaddvq_u8(uneven_left_filtered);
            sum_vector[3 + 4*i] = vaddvq_u8(uneven_right_filtered);
        }
        uint8x8_t values_to_sum = vld1_u8(*sum_vector);
        band_sum = vaddv_u8(values_to_sum); // The sum of green pixels across one vertical row of the image
    }
}

#else
void get_direction(struct image_t *img, int resolution, float *best_heading, float *safe_length) {

    float step_size = M_PI / (float)resolution;
    *best_heading = 0;
    *safe_length = 0;

    int counter = 0; //Initialize the counter variable
    int number_steps_average = 31;
    int radial_memory[31] = {0};

    for (float angle = 0.001; angle < M_PI; angle += step_size) {
        float radial = get_radial(img, angle, img->w);

        if (counter + 1 > number_steps_average){
            // Move elements one position up and discard the first element
            for (int i = 0; i < number_steps_average-2; ++i){
                radial_memory[i] = radial_memory[i+1];
            }
            // Store value in the last position
            radial_memory[number_steps_average - 1] = radial;
        } else {
            // Store value in the current position
            radial_memory[counter] = radial;
        }


        //VERBOSE_PRINT("GF: Radial %f is %f\n", angle, radial);

        if (counter >= number_steps_average-1){
            if (counter == number_steps_average - 1){
                float average_radial = 0;
                int steps_used = 0;
                for (int i = 0; i < (number_steps_average - 1)/2; ++i) {
                    if (i == 0){
                        average_radial += radial_memory[i]*sin((i+steps_used)*step_size);
                        //VERBOSE_PRINT("GF: check right %f\n", radial_memory[i]/240);
                    } else {
                    average_radial += (radial_memory[i+steps_used]*((1-weight_function)+weight_function*sin((i+steps_used)*step_size)) + radial_memory[i+steps_used+1]*((1-weight_function)+weight_function*sin((i+steps_used+1)*step_size)));
                    average_radial = average_radial/(2*i+1);
                    steps_used += 1;
                    }
                    if (average_radial > *safe_length) {
                        *best_heading = i*step_size;
                        *safe_length = average_radial;
                    }
                }
            }

            float average_radial = 0;
            float angle_in_middle = angle - (number_steps_average-1)*step_size/2;
            for (int i = 0; i < number_steps_average; ++i) {
                average_radial += radial_memory[i]*((1-weight_function)+weight_function*sin(angle - (number_steps_average-1-i)*step_size));
            }
            //average_radial = average_radial*sin((angle_in_middle-M_PI/6)*M_PI/(5*M_PI/6-M_PI/6))/number_steps_average;
            average_radial = average_radial/number_steps_average;
            // VERBOSE_PRINT("%f  %f  %f\n",angle_in_middle,average_radial,*safe_length);
            if (average_radial > *safe_length) {
                *best_heading = angle_in_middle;
                *safe_length = average_radial;
            }
            if(counter == resolution-1){
                average_radial = average_radial*number_steps_average;
                int steps_used = 0;
                for (int i = 0; i < (number_steps_average - 1)/2; ++i) {
                    average_radial += -(radial_memory[i+steps_used]*((1-weight_function)+weight_function*sin(angle-(number_steps_average-1-2*steps_used)*step_size)) + radial_memory[i+steps_used+1]*((1-weight_function)+weight_function*sin(angle-(number_steps_average-1-2*steps_used-1)*step_size)));
                    average_radial = average_radial/(number_steps_average-2*(i+1));
                    steps_used += 1;
                    if (average_radial > *safe_length) {
                        *best_heading = angle-((number_steps_average-1)/2-i)*step_size;
                        *safe_length = average_radial;
                    }
                }
            }
        }

        ++counter;
    }
    uint8_t *buffer = img->buf;
    for (double i = 0; i < img->w; i++) {
        uint16_t y = (uint16_t)((double)img->h/2 + i * cos(*best_heading));
        uint16_t x = (uint16_t)(i * sin(*best_heading));
        buffer[y * 2 * img->w + 2 * x] = 0;      // U
        buffer[y * 2 * img->w + 2 * x + 1] = 0;  // Y1
        buffer[y * 2 * img->w + 2 * x + 2] = 255;  // V
    }
    *best_heading = M_PI/2-*best_heading;
    // VERBOSE_PRINT("GF: Angle %f and length %f\n", *best_heading,*safe_length);



    //*best_heading = 0;
}

#endif

/*
 * Init and Periodic code
 */
struct image_t *green_heading_finder1(struct image_t *img, uint8_t camera_id);
struct image_t *green_heading_finder1(struct image_t *img, uint8_t camera_id __attribute__((unused)))
{
    return green_heading_finder(img);
}

void green_detector_init(void) {
    register_periodic_telemetry(DefaultPeriodic, PPRZ_MSG_ID_GREEN_FOLLOWER, send_green_follower);

    memset(&global_heading_object, 0, sizeof(struct heading_object_t));
    pthread_mutex_init(&mutex, NULL);

    #ifdef GREEN_DETECTOR_LUM_MIN
        gd_lum_min = GREEN_DETECTOR_LUM_MIN;
        gd_lum_max = GREEN_DETECTOR_LUM_MAX;
        gd_cb_min = GREEN_DETECTOR_CB_MIN;
        gd_cb_max = GREEN_DETECTOR_CB_MAX;
        gd_cr_min = GREEN_DETECTOR_CR_MIN;
        gd_cr_max = GREEN_DETECTOR_CR_MAX;
    #endif

    cv_add_to_device(&GREENFILTER_CAMERA, green_heading_finder1, GREENFILTER_FPS, 0);
}

void green_detector_periodic(void) {
    static struct heading_object_t local_heading_object;
    pthread_mutex_lock(&mutex);
    memcpy(&local_heading_object, &global_heading_object, sizeof(struct heading_object_t));
    pthread_mutex_unlock(&mutex);

    if(local_heading_object.updated){
        end_cycle_counter = clock();
        AbiSendMsgGREEN_DETECTION(GREEN_DETECTION_ID, local_heading_object.best_heading, local_heading_object.safe_length, local_heading_object.green_pixels);
        local_heading_object.updated = false;

        global_heading_object.cycles_since_update = (end_cycle_counter - start_cycle_counter);
        start_cycle_counter = clock();
    }
}
