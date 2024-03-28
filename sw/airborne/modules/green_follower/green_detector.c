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

#ifndef IMAGE_BOOL
#define IMAGE_BOOL 5
#endif

// Only one should be active at a time!!
#define PAINT_OVER_IMAGE_NORMAL FALSE
#define PAINT_OVER_IMAGE_AVERAGED TRUE

// Enables vector optimization
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

float *hysteresis_template_p;

// TODO Make auto-select based on build target
#define CYBERZOO_FILTER TRUE
#if CYBERZOO_FILTER
// Filter Settings CYBERZOO
uint8_t gd_lum_min = 60;
uint8_t gd_lum_max = 220;
uint8_t gd_cb_min = 20;
uint8_t gd_cb_max = 105;
uint8_t gd_cr_min = 14;
uint8_t gd_cr_max = 150;
#else
// Filter Settings NPS/GAZEBO
uint8_t gd_lum_min = 60;
uint8_t gd_lum_max = 110;
uint8_t gd_cb_min = 75;
uint8_t gd_cb_max = 110;
uint8_t gd_cr_min = 110;
uint8_t gd_cr_max = 130;
#endif

uint8_t local_lum_min;
uint8_t local_lum_max;
uint8_t local_cb_min;
uint8_t local_cb_max;
uint8_t local_cr_min;
uint8_t local_cr_max;

float gain_centre = 0.6;
float gain_previous_heading = 0.8;
float hysteresis_width = 5;
float hysteresis_sides = 0.5;
uint8_t visualize = 0;

// Define constants
float weight_function = 0.85;
int scan_resolution = 100; // Amount of radials
clock_t start_cycle_counter = 0; // Start timer for cycles_since_update
clock_t end_cycle_counter = 0; // End timer for cycles_since_update

const uint8_t kernel_size_w = 16;	// Note: Needs to be an integer divider of the input image pixel width
const uint8_t kernel_size_h = 8; // Note: Needs to be an integer divider of the input image pixel height
// NOTE [Aaron]: If we make it such that the number of pixels in the reduced image (after the filter) is divisible by 8,
// we could do bitwise boolean allocation to save on memory, but waaay overkill for now
const float ray_weights[] = {0.1f, 0.5f, 0.85f, 1.0f, 0.85f, 0.5f, 0.1f};

const float ray_angles[] = {M_PI/6.0f,
                            M_PI/6.0f + 			 M_PI/9.0f,
                            M_PI/6.0f + 2.0f * M_PI/9.0f,
                            M_PI/6.0f + 3.0f * M_PI/9.0f,
                            M_PI/6.0f + 4.0f * M_PI/9.0f,
                            M_PI/6.0f + 5.0f * M_PI/9.0f,
                            M_PI/6.0f + 6.0f * M_PI/9.0f};

static pthread_mutex_t mutex;

// Struct with relevant information for the navigation
struct heading_object_t {
    float best_heading;
    uint16_t old_direction;
    float safe_length;
    uint32_t green_pixels;
    float cycle_time;
    float cycles_since_update;
    bool updated;
};
struct heading_object_t global_heading_object;

#if SIMD_ENABLED == TRUE
// Struct with relevant information for SIMD Thresholding
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

void set_hysteresis_template(float *local_hysteresis_template_p, uint16_t width);
void get_direction(float* regions, float* local_hysteresis_template_p,
                        uint16_t old_direction, uint16_t* new_direction, float* safe_length, uint32_t* green_pixels);
#if SIMD_ENABLED
uint8x16_t average_block(struct image_t *img, uint32_t location);
void get_regions(struct image_t *img, float* regions);
static void green_filter(struct image_t* original_image, struct image_t* filtered_image);
#else
void get_regions(struct image_t* original_image, float* regions);
#endif

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

    float best_heading, safe_length;

    uint32_t green_pixels;
    uint16_t old_direction = global_heading_object.old_direction;
    uint16_t new_direction = 0;

    clock_t start = clock();
    #if SIMD_ENABLED
      // Update threshold arrays, when thresholds change
      if (local_lum_min + local_lum_max + local_cb_min +
          local_cb_max + local_cr_min + local_cr_max !=
          gd_lum_min + gd_lum_max + gd_cb_min + gd_cb_max + gd_cr_min + gd_cr_max) {

        local_lum_min = gd_lum_min;
        local_lum_max = gd_lum_max;
        local_cb_min = gd_cb_min;
        local_cb_max = gd_cb_max;
        local_cr_min = gd_cr_min;
        local_cr_max = gd_cr_max;

        uint8_t min_thresh_array[16] = {local_cb_min, local_cb_min, local_lum_min, local_lum_min,
                                        local_cr_min, local_cr_min, local_lum_min, local_lum_min,
                                        local_cb_min, local_cb_min, local_lum_min, local_lum_min,
                                        local_cr_min, local_cr_min, local_lum_min, local_lum_min};
        uint8_t *min_thresh_pointer = min_thresh_array;
        uint8_t max_thresh_array[16] = {local_cb_max, local_cb_max, local_lum_max, local_lum_max,
                                        local_cr_max, local_cr_max, local_lum_max, local_lum_max,
                                        local_cb_max, local_cb_max, local_lum_max, local_lum_max,
                                        local_cr_max, local_cr_max, local_lum_max, local_lum_max};
        uint8_t *max_thresh_pointer = max_thresh_array;

        gto.min_thresh = vld1q_u8(min_thresh_pointer);
        gto.max_thresh = vld1q_u8(max_thresh_pointer);
      }
    #endif

      float regions[16];
      memset(&regions, 0, 16*sizeof(float));

      get_regions(img, regions);
      get_direction(regions, hysteresis_template_p,
                         old_direction, &new_direction, &safe_length, &green_pixels);

      old_direction = new_direction;
      best_heading = ((float)new_direction - 8.0f) * 0.065;

    #if SIMD_ENABLED
      // Visualize
      if (visualize == 1) {
        struct image_t filtered_image;
        image_create(&filtered_image, img->w / kernel_size_w, img->h / kernel_size_h, IMAGE_BOOL);
        green_filter(img, &filtered_image);
        img = &filtered_image;
      }
      else if (visualize == 2) {
        uint32_t dummy_green_pixels;
        apply_threshold(img, &dummy_green_pixels,
                        local_lum_min, local_lum_max,
                        local_cb_min, local_cb_max,
                        local_cr_min, local_cr_max);
      }
    #endif

    clock_t end = clock();

    pthread_mutex_lock(&mutex);
    global_heading_object.best_heading = best_heading;
    global_heading_object.old_direction = new_direction;
    global_heading_object.safe_length = safe_length;
    global_heading_object.green_pixels = green_pixels;
    global_heading_object.updated = true;

    global_heading_object.cycle_time = (end - start);
    pthread_mutex_unlock(&mutex);

    return img;
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

  hysteresis_template_p = malloc(2*16*sizeof(float));
  set_hysteresis_template(hysteresis_template_p, hysteresis_width);

  // Initialize the SIMD parameters
#if SIMD_ENABLED
  // Set Threshold arrays
  uint8_t min_thresh_array[16] = {gd_cb_min, gd_cb_min, gd_lum_min, gd_lum_min,
                                  gd_cr_min, gd_cr_min, gd_lum_min, gd_lum_min,
                                  gd_cb_min, gd_cb_min, gd_lum_min, gd_lum_min,
                                  gd_cr_min, gd_cr_min, gd_lum_min, gd_lum_min};

  uint8_t max_thresh_array[16] = {gd_cb_max, gd_cb_max, gd_lum_max, gd_lum_max,
                                  gd_cr_max, gd_cr_max, gd_lum_max, gd_lum_max,
                                  gd_cb_max, gd_cb_max, gd_lum_max, gd_lum_max,
                                  gd_cr_max, gd_cr_max, gd_lum_max, gd_lum_max};

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

/*
 * Thresholding code for if the green pixels should be drawn on the image
 * Also returns total amount of green pixels
 */
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
        }
    }
    *green_pixels = local_green_pixels;
}

static void green_filter(struct image_t* original_image, struct image_t* filtered_image) {

  uint8_t *original_buffer = original_image->buf;
  uint8_t *filtered_buffer = filtered_image->buf;

  // Perform mean pooling on the original image
  for (uint16_t row=0; row<filtered_image->h; row++) {
    for (uint16_t col=0; col<filtered_image->w; col++) {

      uint32_t yuv_y = 0;
      uint32_t yuv_u = 0;
      uint32_t yuv_v = 0;

      // Sum color channels of pixels in the kernel
      for (uint8_t row_in_kernel=0; row_in_kernel<kernel_size_h; row_in_kernel++) {
        for (int8_t col_in_kernel=0; col_in_kernel<kernel_size_w; col_in_kernel++) {
          // Parse depending on even or uneven col nr
          if ((col * kernel_size_w + col_in_kernel) % 2 == 0) {
            // Even col nr
            yuv_u += original_buffer[(row * kernel_size_h + row_in_kernel) * 2 * original_image->w + 2 * (col * kernel_size_w + col_in_kernel)];      // U
            yuv_y += original_buffer[(row * kernel_size_h + row_in_kernel) * 2 * original_image->w + 2 * (col * kernel_size_w + col_in_kernel) + 1];  // Y1
            yuv_v += original_buffer[(row * kernel_size_h + row_in_kernel) * 2 * original_image->w + 2 * (col * kernel_size_w + col_in_kernel) + 2];  // V
          }
          else {
            // Uneven col nr
            yuv_u += original_buffer[(row * kernel_size_h + row_in_kernel) * 2 * original_image->w + 2 * (col * kernel_size_w + col_in_kernel) - 2];  // U
            yuv_v += original_buffer[(row * kernel_size_h + row_in_kernel) * 2 * original_image->w + 2 * (col * kernel_size_w + col_in_kernel)];      // V
            yuv_y += original_buffer[(row * kernel_size_h + row_in_kernel) * 2 * original_image->w + 2 * (col * kernel_size_w + col_in_kernel) + 1];  // Y2
          }
        }
      }

      // Divide by number of pixels in the kernel to get the mean values
      yuv_y = yuv_y / (kernel_size_w * kernel_size_h);
      yuv_u = yuv_u / (kernel_size_w * kernel_size_h);
      yuv_v = yuv_v / (kernel_size_w * kernel_size_h);

      // Perform limit checks and assign value to filter's pixel
      if (	(yuv_y >= gd_lum_min) && (yuv_y < gd_lum_max) &&
            (yuv_u >= gd_cb_min) 	&& (yuv_u < gd_cb_max)  &&
            (yuv_v >= gd_cr_min) 	&& (yuv_v < gd_cr_max)) {

        filtered_buffer[row * filtered_image->w + col] = true;

#if PAINT_OVER_IMAGE_AVERAGED && !SIMD_ENABLED
        // Go over all pixels in kernel and adjust them color to resemble valid pixels
        for (uint8_t row_in_kernel=0; row_in_kernel<kernel_size_h; row_in_kernel++) {
          for (int8_t col_in_kernel=0; col_in_kernel<kernel_size_w; col_in_kernel++) {
            // Parse depending on even or uneven col nr
            if ((col * kernel_size_w + col_in_kernel) % 2 == 0) {
              // Even col nr
              original_buffer[(row * kernel_size_h + row_in_kernel) * 2 * original_image->w + 2 * (col * kernel_size_w + col_in_kernel)] = 0;      // U
              original_buffer[(row * kernel_size_h + row_in_kernel) * 2 * original_image->w + 2 * (col * kernel_size_w + col_in_kernel) + 1] = 127;  // Y1
              original_buffer[(row * kernel_size_h + row_in_kernel) * 2 * original_image->w + 2 * (col * kernel_size_w + col_in_kernel) + 2] = 0;  // V
            }
            else {
              // Uneven col nr
              original_buffer[(row * kernel_size_h + row_in_kernel) * 2 * original_image->w + 2 * (col * kernel_size_w + col_in_kernel) - 2] = 0;  // U
              original_buffer[(row * kernel_size_h + row_in_kernel) * 2 * original_image->w + 2 * (col * kernel_size_w + col_in_kernel)] = 0;      // V
              original_buffer[(row * kernel_size_h + row_in_kernel) * 2 * original_image->w + 2 * (col * kernel_size_w + col_in_kernel) + 1] = 127;  // Y2
            }
          }
        }
#endif

      }
      else {
        filtered_buffer[row * filtered_image->w + col] = false;
      }
    }
  }
}

void set_hysteresis_template(float *local_hysteresis_template_p, uint16_t width) {
  // This function creates a weighing function
  memset(local_hysteresis_template_p, 0, 2 * 16 * sizeof(float));
  float amplitude = (1 - hysteresis_sides)/2;
  float average = 1 - amplitude;
  for (uint16_t i = 0; i < 16; i++) {
    local_hysteresis_template_p[i] = hysteresis_sides;
  }
  for (uint16_t i = 0; i < width; i++) {
    // local_hysteresis_template_p[16 - (uint16_t) (M_PI * i/2)] = 0.75 + (uint16_t) (0.25 * cos(2 * (i - 16) / width));
    uint8_t measure_point = 16 - width/2 + i;
    local_hysteresis_template_p[measure_point] = average + (amplitude * cos(4 * ((float)measure_point - 16) / (float)width));
  }
}

void get_direction(float* regions, float* local_hysteresis_template_p,
                        uint16_t old_direction, uint16_t* new_direction, float* safe_length, uint32_t* green_pixels) {
  // Calculates the direction the drone should go using the weighing function and regions
  uint16_t local_new_direction = 0;
  uint32_t local_green_pixels = 0;
  float weighted_regions[16];
  float local_safe_length = 0;

  for (uint16_t i = 0; i < 16; i++) {
    // Apply weights
    uint16_t array_location = i + 16 - old_direction;
    float average_weighting = (gain_previous_heading * local_hysteresis_template_p[array_location] +
                               gain_centre * local_hysteresis_template_p[i + 8]) / (gain_previous_heading + gain_centre);
    weighted_regions[i] = (regions[i] * average_weighting);

    // Count up all the green pixels
    if (regions[i] >= 0) {  // NOTE: [Aaron] Only cast to uint if the region has positive value (otherwise would underflow)
      local_green_pixels += (uint32_t)regions[i];
    }

    // Find maximum
    if (weighted_regions[i] > weighted_regions[local_new_direction]) {
      local_new_direction = i; // Set new direction (not the same as heading)
      local_safe_length = regions[i]; // Set new safe length
    }
  }
  *new_direction = local_new_direction;
  *safe_length = local_safe_length;
  *green_pixels = local_green_pixels * 60; // 60 to make it comparable to the old green follower!!
}

#if SIMD_ENABLED == TRUE
uint8x16_t average_block(struct image_t *img, uint32_t location) {
  // This function computes a 8x8 pixel average
  uint8_t *buffer = img->buf;
  uint8x16_t averages_1[4];
  uint8x16_t averages_2[2];

  for (uint8_t row = 0; row < 8; row += 2) {
    uint8x16_t slice_1 = vld1q_u8(buffer + location + 480 * row);
    uint8x16_t slice_2 = vld1q_u8(buffer + location + 480 * (row + 1));
    averages_1[row/2] = vhaddq_u8(slice_1, slice_2);
  }
  for (uint8_t row = 0; row < 4; row += 2) {
    averages_2[row/2] = vhaddq_u8(averages_1[row], averages_1[row + 1]);
  }
  return vhaddq_u8(averages_2[0], averages_2[1]);
}

void get_regions(struct image_t *img, float* regions) {
  // This function gets 16 regions from the image and counts the amount of averaged green pixels within these regions
  uint8x8_t first_add_low_array[4];
  uint8x8_t first_add_high_array[4];
  uint8x16_t second_add_array[2];
  uint8x16_t region_array[2];

  uint32_t location_1, location_2;

  for (uint8_t region_id = 0; region_id < 16; region_id++) {
    for (uint8_t k = 0; k < 2; k++) {
      uint8x16_t greater_combined = gto.zero_array; // A uint8 vector with 16 values of which every bit represents a y, u or v value
      uint8x16_t smaller_combined = gto.zero_array; // A uint8 vector with 16 values of which every bit represents a y, u or v value
      for (uint8_t i = 0; i < 8; i++) {
        for (uint8_t j = 0; j < 4; j++) {
          if (i == 1 && j == 3) { // Cut-off the 8th value of the first and second block
            first_add_low_array[j] = gto.zero_array_8; // Maintain constant average
            first_add_high_array[j] = gto.zero_array_8; // Maintain constant average
          }
          else {
            if (i == 0 || i == 1 || i == 4 || i == 5){
              location_1 = 16*j + 64*(i%2) + 8*480*(i/3) + 480*16*k + 480*32*region_id;
              location_2 = location_1 + 7*16;
            } else {
              location_1 = 16*j + 64*(i%2) + 8*480*(i/5) + 14*16 + 480*16*k + 480*32*region_id;
              location_2 = location_1 + 8*16;
            }

            // uint8_t test_vector_1[16] = {80, 70, 15, 70, 80, 70, 15, 70, 80, 70, 15, 70, 80, 70, 15, 70};
            // uint8_t test_vector_2[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

            // Take slices from the image
            // Starts with arrays in form [u00, y00, v00, y01, u01, y02, v01, y03, ... , y07],
            //                            [u10, y10, v10, y11, u11, y12, v11, y13, ... , y17], ...
            uint8x16_t slice_1 = average_block(img, location_1); // Load in a slice
            uint8x16_t slice_2 = average_block(img, location_2); // Load in a slice 8 addresses further than slice 1

            // Zip and add arrays
            // Gets array in form [(u00+u02)/2, (u10+u12)/2, (y00+y04)/2, (y10+y14)/2, .. (y115+y137)/2], ...
            uint8x16x2_t zipped_slices = vzipq_u8(slice_1, slice_2); // Zip the slices together so [a0, b0, a1, b1, ...]
            uint8x16_t first_add = vhaddq_u8(zipped_slices.val[0],
                                             zipped_slices.val[1]); // Add the first zip array to the second

            // Separate arrays
            first_add_low_array[j] = vget_low_u8(first_add); // Get the first 8 bytes of first_add_1
            first_add_high_array[j] = vget_high_u8(first_add); // Get the last 8 bytes of first_add_1
          }

          // Start adding
          if (j == 1) {
            // Combine and add
            // Gets array in form [u0_t, u1_t, y0_t, ... y7_t]
            uint8x16_t first_add_low_comb = vcombine_u8(first_add_low_array[0], first_add_low_array[1]);
            uint8x16_t first_add_high_comb = vcombine_u8(first_add_high_array[0], first_add_high_array[1]);
            second_add_array[0] = vhaddq_u8(first_add_low_comb, first_add_high_comb);
          } else if (j == 3) {
            // Combine and add
            // Gets array in form [u0_t, u1_t, y0_t, ... y7_t]
            uint8x16_t first_add_low_comb = vcombine_u8(first_add_low_array[2], first_add_low_array[3]);
            uint8x16_t first_add_high_comb = vcombine_u8(first_add_high_array[2], first_add_high_array[3]);
            second_add_array[1] = vhaddq_u8(first_add_low_comb, first_add_high_comb);
          }
        }
        // Average of 4 windows [u0_av, u1_av, y00_av, y10_av, v0_av, v1_av, y01_av, y11_av, u2_av, ..., y31_av]
        uint8x16_t third_add_array = vhaddq_u8(second_add_array[0], second_add_array[1]);

        // Threshold
        uint8x16_t greater = vcgeq_u8(third_add_array, gto.min_thresh);
        uint8x16_t smaller = vcleq_u8(third_add_array, gto.max_thresh);

        // Put into bit vector
        uint8x16_t selection_array = vdupq_n_u8(gto.select[i]); // TODO: initialize this value
        greater_combined = vbslq_u8(selection_array, greater, greater_combined);
        smaller_combined = vbslq_u8(selection_array, smaller, smaller_combined);
      }
      // Get the bitwise union between the greater_combined and smaller_combined vectors
      // The result is an array in which every bit represents a y, u or v value. If the bit is 1, the value is within the threshold.
      uint8x16_t bounded = vandq_u8(greater_combined, smaller_combined);

      // Rotate matrices
      uint8x16_t reverse_64 = vrev64q_u8(bounded);
      uint8x16_t reverse_32 = vrev32q_u8(bounded);
      uint8x16_t reverse_16 = vrev16q_u8(bounded);

      // Check if correct
      uint8x16_t check_1 = vandq_u8(reverse_64, reverse_32);
      uint8x16_t check_2 = vandq_u8(reverse_32, reverse_16);
      uint8x16_t check_3 = vandq_u8(check_1, check_2);

      // Pop count
      region_array[k] = vcntq_u8(check_3);
    }
    // Add together green pixels
    uint8x16_t first_region_add = vaddq_u8(region_array[0], region_array[1]);
    uint8x16_t first_region_add_r16 = vrev64q_u8(first_region_add);
    uint8x16_t sra = vaddq_u8(first_region_add_r16, first_region_add);

    regions[region_id] = (float)(sra[2] + sra[6] + sra[10] + sra[14]);
  }
}
                        
#else
void get_regions(struct image_t* original_image, float* regions) {

	// Array which will contain the filtered image
  struct image_t filtered_image;
  image_create(&filtered_image, original_image->w / kernel_size_w, original_image->h / kernel_size_h, IMAGE_BOOL);
  // Ptr to the filtered buffer
  uint8_t *filtered_buffer = filtered_image.buf;

	// Apply filter to original image
	green_filter(original_image, &filtered_image);

  // Go through each pixel in the filtered image
	// and determine the number of valid large pixels in each vertical band of 4 large pixels wide (=region?)
  memset(regions, 0, 16*sizeof(float));
  for (uint16_t region=0; region<filtered_image.h / 4; region++) {
    for (uint16_t row_in_region=0; row_in_region<4; row_in_region++) {
		  for (uint16_t col=0; col<11; col++) {
        if (filtered_buffer[(region*4 + row_in_region) * filtered_image.w + col] == true) {
          // Add a count to the corresponding region
          regions[region] += 1.0f;
        }
		  }
      for (uint16_t col=11; col<filtered_image.w; col++) {
        if (filtered_buffer[(region*4 + row_in_region) * filtered_image.w + col] == true) {
          // Subtract a count to the corresponding region
          regions[region] -= 1.0f;
        }
      }
    }
	}

	// Deallocate memory
  image_free(&filtered_image);

}

#endif
