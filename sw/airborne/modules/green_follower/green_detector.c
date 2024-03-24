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

#define SIMD_ENABLED TRUE

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
#define CYBERZOO_FILTER TRUE
#if CYBERZOO_FILTER
// Filter Settings CYBERZOO
uint8_t gd_lum_min = 60;
uint8_t gd_lum_max = 140;
uint8_t gd_cb_min = 24;
uint8_t gd_cb_max = 105;
uint8_t gd_cr_min = 28;
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

float weight_function = 0.85;

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
    uint8x16_t zero_array;
    uint8x8_t zero_array_8;
    uint8x16_t one_array;
    uint8_t select[8];
    uint8x16_t min_thresh;
    uint8x16_t max_thresh;
};
struct threshold_object_t gto;
#endif

void apply_threshold(struct image_t *img, uint32_t *green_pixels,
                     uint8_t lum_min, uint8_t lum_max,
                     uint8_t cb_min, uint8_t cb_max,
                     uint8_t cr_min, uint8_t cr_max);

float get_radial(struct image_t *img, float angle, uint8_t radius);

void get_direction(struct image_t *img, int resolution, float* best_heading, float* safe_length);

void get_lines(struct image_t *img, uint8_t* band_sum);
void add_band_sums(uint8_t* band_sum, uint32_t* green_pixel);
void average_regions_32(uint8_t* band_sum, uint8_t* average_array);

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
    uint8_t lum_min, lum_max;
    uint8_t cb_min, cb_max;
    uint8_t cr_min, cr_max;

    float best_heading, safe_length;

    uint32_t green_pixels_simd = 0;
    uint32_t green_pixels_norm = 0;

    uint32_t green_pixels;

    lum_min = gd_lum_min;
    lum_max = gd_lum_max;
    cb_min = gd_cb_min;
    cb_max = gd_cb_max;
    cr_min = gd_cr_min;
    cr_max = gd_cr_max;


    #if SIMD_ENABLED == TRUE
    clock_t start_simd = clock();

//    // Thresholds start
//
//    uint8_t min_thresh_array[16] = {cb_min, lum_min, cr_min, lum_min,
//                                    cb_min, lum_min, cr_min, lum_min,
//                                    cb_min, lum_min, cr_min, lum_min,
//                                    cb_min, lum_min, cr_min, lum_min};
//    uint8_t *min_thresh_pointer = min_thresh_array;
//    uint8_t max_thresh_array[16] = {cb_max, lum_max, cr_max, lum_max,
//                                    cb_max, lum_max, cr_max, lum_max,
//                                    cb_max, lum_max, cr_max, lum_max,
//                                    cb_max, lum_max, cr_max, lum_max};
//    uint8_t *max_thresh_pointer = max_thresh_array;
//
//    gto.min_thresh = vld1q_u8(min_thresh_pointer);
//    gto.max_thresh = vld1q_u8(max_thresh_pointer);
//    // Thresholds end

    uint8_t band_sum[520];
    memset(&band_sum, 0, 520*sizeof(uint8_t));
    uint8_t average_array[504];

    get_lines(img, &band_sum[0]);
    add_band_sums(&band_sum[0], &green_pixels);
    average_regions_32(&band_sum[0], &average_array[0]);

    green_pixels_simd = green_pixels;
    clock_t end_simd = clock();

    clock_t start_norm = clock();
    apply_threshold(img, &green_pixels, lum_min, lum_max, cb_min, cb_max, cr_min, cr_max);
    green_pixels_norm = green_pixels;
    clock_t end_norm = clock();
    #else
    // Filter the image so that all green pixels have a y value of 255 and all others a y value of 0
    apply_threshold(img, &green_pixels, lum_min, lum_max, cb_min, cb_max, cr_min, cr_max);
    // Scan in radials from the centre bottom of the image to find the direction with the most green pixels
    get_direction(img, scan_resolution, &best_heading, &safe_length);
    #endif

    pthread_mutex_lock(&mutex);
    global_heading_object.best_heading = (end_norm - start_norm);
    global_heading_object.safe_length = (float)green_pixels_simd;
    global_heading_object.green_pixels = green_pixels_norm;
    global_heading_object.updated = true;

    global_heading_object.cycle_time = (end_simd - start_simd);
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

                //*yp = 255;  // make pixel brighter
                local_green_pixels++;

            }
            else {
                // *yp = 0; // make pixel darker
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
            sum += 1;
        }
    }

    return (float)sum; // * (sin(angle) + 0.2) ;
}

#if SIMD_ENABLED == TRUE
void get_lines(struct image_t *img, uint8_t* band_sum) {
    uint8_t *buffer = img->buf;
    uint8_t bound = 0;

    for (uint16_t y = 0; y < 520; y++) {
        bound = 0;
        for (uint8_t i = 0; i < 4; i++) {
            uint8x16_t greater_combined = gto.zero_array; // A uint8 vector with 16 values of which every bit represents a y, u or v value
            uint8x16_t smaller_combined = gto.zero_array; // A uint8 vector with 16 values of which every bit represents a y, u or v value

            // In the last loop only 6 sliced should be added for summation to round nicely to 240
            if (i == 3) {
              bound = 2;
            }

            for (uint8_t j = 0; j < (8 - bound); j++) {
              // Get a slice from the image buffer
                uint8x16_t slice = vld1q_u8(buffer + j * 16 + i * 128 + y * 480);

                // Get the bounds
                uint8x16_t greater = vcgeq_u8(slice, gto.min_thresh); // Sets all YUV values greater than min thresh to 1
                uint8x16_t smaller = vcleq_u8(slice, gto.max_thresh); // Sets all YUV values smaller than max thresh to 1

              // Combine the boolean integers into one integer to be more efficient
              uint8x16_t selection_array = vdupq_n_u8(gto.select[j]); // TODO: initialize this value
              greater_combined = vbslq_u8(selection_array, greater, greater_combined);
              smaller_combined = vbslq_u8(selection_array, smaller, smaller_combined);
            }
            // Get the bitwise union between the greater_combined and smaller_combined vectors
            // The result is an array in which every bit represents a y, u or v value. If the bit is 1, the value is within the threshold.
            uint8x16_t bounded = vandq_u8(greater_combined, smaller_combined);

            // Set last value to zero, to make sure iterations fit nicely in 240
            // bounded = vsetq_lane_u8(0, bounded, 15);

            // Naar links!
            uint8x16_t shifted_l = vld1q_u8(&bounded[1]);
            uint8x16_t first_sum = vandq_u8(bounded, shifted_l);

            // Tuh tuh tuh tuh
            uint8x16_t shifted_ll = vld1q_u8(&shifted_l[1]);
            uint8x16_t even_sum = vandq_u8(first_sum, shifted_ll);
            uint8x16_t even_pop = vcntq_u8(even_sum);

            // Naar rechts!
            uint8x16_t shifted_rr = vld1q_u8(&bounded[-2]);
            uint8x16_t uneven_sum = vandq_u8(first_sum, shifted_rr);
            uint8x16_t uneven_pop = vcntq_u8(uneven_sum);

            // Sum the even and uneven parts
            uint8x16_t uneven_pop_ll = vld1q_u8(&uneven_pop[2]);
            uint8x16_t sum_pop = vaddq_u8(even_pop, uneven_pop_ll);

            // Retrieve the sum
            band_sum[y] += (uint32_t)sum_pop[0] + (uint32_t)sum_pop[4] + (uint32_t)sum_pop[8] + (uint32_t)sum_pop[12];
        }
    }
}

void add_band_sums(uint8_t* band_sum, uint32_t* green_pixels) {
  uint32_t total_sum = 0;
  for (uint16_t i = 0; i < 520; i++) {
    total_sum += (uint32_t)band_sum[i];
  }

  *green_pixels = total_sum;
}

void average_regions_32(uint8_t* band_sum, uint8_t* average_array) {
  uint8x16_t average_16;
  uint8x16_t average_16_last;
  uint8x16_t current_average = gto.zero_array;
  for (uint32_t i = 0; i < 504; i++) {
    uint8x16_t slice = vld1q_u8(&band_sum[i]);
    current_average = vrhaddq_u8(current_average, slice);
    if (i % 16 == 0 && i != 0) {
      if (i > 30) {
        uint8x16_t average_32 = vrhaddq_u8(average_16, current_average);
        for (uint8_t j = 0; j < 16; j++){
          average_array[i - 31 + j] = average_32[j];
        }

        // Last averages
        if (i == 472) {
          average_16_last = current_average;
        }
        if (i == 503) {
          uint8x16_t average_32_last = vrhaddq_u8(average_16_last, current_average);
          for (uint8_t j = 0; j < 16; j++){
            average_array[i - 31 + j] = average_32_last[j];
          }
        }
      }
      average_16 = current_average;
    }
  }
}

#endif

void get_direction(struct image_t *img, int resolution, float *best_heading, float *safe_length) {

    float step_size = M_PI / (float)resolution;
    *best_heading = 0;
    *safe_length = 0;

    int counter = 0; //Initialize the counter variable
    int number_steps_average = 31;
    int radial_memory[31] = {0};

    for (float angle = 0.001; angle < M_PI; angle += step_size) {
        float radial = get_radial(img, angle, img->w);

        if (counter > number_steps_average - 1){
            // Move elements one position up and discard the first element
            for (int i = 0; i < number_steps_average-1; ++i){
                radial_memory[i] = radial_memory[i+1];
            }
            // Store value in the last position
            radial_memory[number_steps_average - 1] = radial;
        } else {
            // Store value in the current position
            radial_memory[counter] = radial;
        }



        if (counter >= number_steps_average-1){
            float correction_weight = 0.2*(1-2*global_heading_object.green_pixels/(520.0 * 240.0));
            //float correction_weight = 0;
            // VERBOSE_PRINT("GF: correction weight %f\n", correction_weight);
            if (counter == number_steps_average - 1){
                float average_radial = 0;
                int steps_used = 0;
                for (int i = 0; i < (number_steps_average - 1)/2; ++i) {
                    if (i == 0){
                        //average_radial += radial_memory[i]*((1-weight_function+correction_weight)+(weight_function - correction_weight)*sin((i+steps_used)*step_size));
                        average_radial += radial_memory[i]*(1.5 - (weight_function-correction_weight)*fabs((i+steps_used)*step_size-M_PI/2)-0.5*sin((i+steps_used)*step_size));
                        //VERBOSE_PRINT("GF: check right %f\n", radial_memory[i]/240);
                    } else {

                    //average_radial += (radial_memory[i+steps_used]*((1-weight_function+correction_weight)+(weight_function - correction_weight)*sin((i+steps_used)*step_size)) + radial_memory[i+steps_used+1]*((1-weight_function+correction_weight)+(weight_function - correction_weight)*sin((i+steps_used+1)*step_size)));
                    average_radial += (radial_memory[i+steps_used]*(1.5 - (weight_function-correction_weight)*fabs((i+steps_used)*step_size-M_PI/2)-0.5*sin((i+steps_used)*step_size)) + radial_memory[i+steps_used+1]*(1.5 - (weight_function-correction_weight)*fabs((i+steps_used+1)*step_size-M_PI/2)-0.5*sin((i+steps_used+1)*step_size)));
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
                //average_radial += radial_memory[i]*((1-weight_function+correction_weight)+(weight_function - correction_weight)*sin(angle - (number_steps_average-1-i)*step_size));
                average_radial += (radial_memory[i]*(1.5 - (weight_function-correction_weight)*fabs(angle-(number_steps_average-1-i)*step_size-M_PI/2)-0.5*sin(angle-(number_steps_average-1-i)*step_size)));
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
                    //average_radial += -(radial_memory[i+steps_used]*((1-weight_function+correction_weight)+(weight_function - correction_weight)*sin(angle-(number_steps_average-1-2*steps_used)*step_size)) + radial_memory[i+steps_used+1]*((1-weight_function+correction_weight)+(weight_function - correction_weight)*sin(angle-(number_steps_average-1-2*steps_used-1)*step_size)));
                    average_radial += -(radial_memory[i+steps_used]*(1.5 - (weight_function-correction_weight)*fabs(angle-(number_steps_average-1-2*steps_used)*step_size-M_PI/2)-0.5*sin(angle-(number_steps_average-1-2*steps_used)*step_size))+radial_memory[i+steps_used+1]*(1.5 - (weight_function-correction_weight)*fabs(angle-(number_steps_average-1-2*steps_used-1)*step_size-M_PI/2)-0.5*sin(angle-(number_steps_average-1-2*steps_used-1)*step_size)));
                    average_radial = average_radial/(number_steps_average-2*(i+1));
                    steps_used += 1;

                    if (average_radial > *safe_length) {
                        *best_heading = angle-((number_steps_average-1)/2-i)*step_size;
                        *safe_length = average_radial;
                    }
                    average_radial = average_radial*(number_steps_average-2*(i+1));
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

    // Initialize the SIMD parameters
    #if SIMD_ENABLED == TRUE
        // Set Threshold arrays
        uint8_t min_thresh_array[16] = {gd_cb_min, gd_lum_min, gd_cr_min, gd_lum_min,
                                        gd_cb_min, gd_lum_min, gd_cr_min, gd_lum_min,
                                        gd_cb_min, gd_lum_min, gd_cr_min, gd_lum_min,
                                        gd_cb_min, gd_lum_min, gd_cr_min, gd_lum_min};

        uint8_t max_thresh_array[16] = {gd_cb_max, gd_lum_max, gd_cr_max, gd_lum_max,
                                        gd_cb_max, gd_lum_max, gd_cr_max, gd_lum_max,
                                        gd_cb_max, gd_lum_max, gd_cr_max, gd_lum_max,
                                        gd_cb_max, gd_lum_max, gd_cr_max, gd_lum_max};

        gto.min_thresh = vld1q_u8(min_thresh_array);
        gto.max_thresh = vld1q_u8(max_thresh_array);

        // Set standard arrays
        gto.zero_array = vdupq_n_u8(0);
        gto.zero_array_8 = vdup_n_u8(0);
        gto.one_array = vdupq_n_u8(1);

        // Set selector to powers of 2 (so 1, 2, 4, 8, etc.) which in binary matches the right bits.
        for (int i = 0; i < 8; i++) {
          gto.select[i] = pow(2, i);
        }
    #endif
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
